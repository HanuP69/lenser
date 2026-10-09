import asyncio
import base64
import io
import urllib.request
import json
from pathlib import Path
from typing import Any, Optional
from PIL import Image, ImageChops
import numpy as np
from langchain_core.messages import HumanMessage
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool, run_in_process_pool
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import SlideCaptured


def capture_screen_png_bytes(monitor_index: int) -> bytes:
    import mss
    with mss.mss() as sct:
        monitors = sct.monitors
        target_monitor = monitors[0]
        if monitor_index < len(monitors):
            target_monitor = monitors[monitor_index]
        screenshot = sct.grab(target_monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG")
        return byte_io.getvalue()


def compute_image_difference(current_png_bytes: bytes, previous_png_bytes: bytes) -> float:
    current_image = Image.open(io.BytesIO(current_png_bytes)).convert("L")
    previous_image = Image.open(io.BytesIO(previous_png_bytes)).convert("L")

    if current_image.size != previous_image.size:
        previous_image = previous_image.resize(current_image.size)

    diff_image = ImageChops.difference(current_image, previous_image)
    diff_array = np.array(diff_image, dtype=np.float32)
    mean_diff = float(np.mean(diff_array))
    normalized_diff = mean_diff / 255.0
    return normalized_diff


def run_tesseract_ocr_worker(image_png_bytes: bytes, tesseract_path: str) -> str:
    import pytesseract
    if tesseract_path and Path(tesseract_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    image = Image.open(io.BytesIO(image_png_bytes))
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()


def run_docker_ocr_request(image_png_bytes: bytes, docker_url: str = "http://localhost:8000/ocr") -> str:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="frame.png"\r\n')
    body.extend(b"Content-Type: image/png\r\n\r\n")
    body.extend(image_png_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        docker_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response_data = json.loads(response.read().decode("utf-8"))
        return str(response_data.get("text", "")).strip()


def save_slide_image_file(image_bytes: bytes, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open(destination_path, "wb") as file_handle:
        file_handle.write(image_bytes)


class ScreenAgent:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str,
        llm_router: Any = None
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.session_clock: SessionClock = session_clock
        self.session_id: str = session_id
        self.llm_router: Any = llm_router
        self.stream_name: str = f"lecturelens:{session_id}:stream:slide"
        self.slide_counter: int = 0
        self.previous_frame_bytes: Optional[bytes] = None
        self.is_running: bool = False

    async def run(self) -> None:
        self.is_running = True
        session_dir = self.settings.create_session_directories(self.session_id)
        raw_slides_dir = session_dir / "raw" / "slides"

        while self.is_running:
            try:
                current_frame_bytes = await run_in_thread_pool(
                    capture_screen_png_bytes,
                    self.settings.monitor_index
                )
            except Exception:
                await asyncio.sleep(self.settings.screenshot_interval_seconds)
                continue

            should_process = False
            if self.previous_frame_bytes is None:
                should_process = True
            else:
                difference_score = compute_image_difference(
                    current_frame_bytes,
                    self.previous_frame_bytes
                )
                if difference_score >= self.settings.slide_diff_threshold:
                    should_process = True

            if not should_process:
                await asyncio.sleep(self.settings.screenshot_interval_seconds)
                continue

            self.previous_frame_bytes = current_frame_bytes
            self.slide_counter += 1
            timestamp_epoch = self.session_clock.get_timestamp()

            slide_file_name = f"slide_{self.slide_counter:04d}.png"
            slide_file_path = raw_slides_dir / slide_file_name
            save_slide_image_file(current_frame_bytes, slide_file_path)

            extracted_text = ""
            try:
                extracted_text = await run_in_process_pool(
                    run_tesseract_ocr_worker,
                    current_frame_bytes,
                    self.settings.tesseract_path
                )
            except Exception:
                extracted_text = ""

            if len(extracted_text) == 0:
                try:
                    extracted_text = run_docker_ocr_request(current_frame_bytes)
                except Exception:
                    extracted_text = ""

            if len(extracted_text) == 0 and self.llm_router is not None:
                try:
                    base64_encoded = base64.b64encode(current_frame_bytes).decode("utf-8")
                    prompt_message = HumanMessage(content=[
                        {
                            "type": "text",
                            "text": "Extract all slide titles, bullet points, text, equations, and describe any visual diagrams or charts shown in this lecture frame."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_encoded}"
                        }
                    ])
                    response = await self.llm_router.ainvoke([prompt_message])
                    extracted_text = str(getattr(response, "content", response)).strip()
                except Exception:
                    extracted_text = ""

            slide_event = SlideCaptured(
                session_id=self.session_id,
                timestamp_epoch=timestamp_epoch,
                slide_number=self.slide_counter,
                monitor_index=self.settings.monitor_index,
                image_file_path=str(slide_file_path),
                extracted_text=extracted_text
            )

            await self.redis_bus.publish(self.stream_name, slide_event)
            await asyncio.sleep(self.settings.screenshot_interval_seconds)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    args, _ = parser.parse_known_args()

    settings = Settings()
    if args.session is not None and len(args.session) > 0:
        settings.session_name = args.session

    session_id = settings.session_name or "default_session"
    clock_file = settings.get_session_output_dir(session_id) / "clock.json"
    if clock_file.exists():
        session_clock = SessionClock.load_from_file(clock_file)
    else:
        session_clock = SessionClock()

    redis_bus = RedisBus(settings.redis_url)

    llm_router = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        router_config = load_router_config(settings.router_config_path)
        llm_router = build_router_for_task(
            task_type="board_ocr",
            redis_client=redis_bus.client,
            router_config=router_config,
            ollama_url=settings.ollama_url
        )
    except Exception:
        llm_router = None

    agent = ScreenAgent(
        settings=settings,
        redis_bus=redis_bus,
        session_clock=session_clock,
        session_id=session_id,
        llm_router=llm_router
    )
    try:
        await agent.run()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
