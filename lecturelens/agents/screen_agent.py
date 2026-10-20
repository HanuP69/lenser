import asyncio
import io
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import AgentControl, RuntimeContext, execute_agent_main, run_standard_agent
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_process_pool, run_in_thread_pool
from lecturelens.core.fileio import atomic_write_bytes, next_file_index
from lecturelens.core.image_utils import (
    SlideChangeGate,
    compute_changed_fraction,
    compute_difference_hash,
    compute_spatial_block_similarity,
    hamming_distance
)
from lecturelens.core.drift_filter import is_refusal_text
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.vision_embedder import VisionEmbedder, get_shared_vision_embedder
from lecturelens.router.messages import build_image_message, response_to_text
from lecturelens.schemas.events import SlideCaptured

logger = get_logger("lecturelens.screen_agent")

TESSERACT_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
MAX_REMEMBERED_SLIDES = 200
REVISIT_THUMBNAIL_WIDTH = 192
REVISIT_MAX_CHANGED_FRACTION = 0.01

SLIDE_VISION_PROMPT = (
    "Transcribe all text visible on this presentation slide exactly. "
    "Preserve code indentation. If there is a diagram, add one short line describing it. "
    "Output only the transcription."
)


@dataclass
class ScreenFrame:
    raw_bgra: bytes
    width: int
    height: int
    thumbnail: np.ndarray


def build_gray_thumbnail(image: Image.Image, thumbnail_width: int) -> np.ndarray:
    original_width, original_height = image.size
    scale = thumbnail_width / float(max(1, original_width))
    thumbnail_height = max(1, int(original_height * scale))
    gray_image = image.convert("L").resize((thumbnail_width, thumbnail_height), Image.BILINEAR)
    return np.array(gray_image, dtype=np.uint8)


def shrink_thumbnail(thumbnail: np.ndarray, target_width: int) -> np.ndarray:
    height, width = thumbnail.shape
    scale = target_width / float(max(1, width))
    target_height = max(1, int(height * scale))
    image = Image.fromarray(thumbnail)
    return np.array(image.resize((target_width, target_height), Image.BILINEAR), dtype=np.uint8)


def capture_screen_frame(monitor_index: int, thumbnail_width: int) -> ScreenFrame:
    import mss
    with mss.mss() as screen_capturer:
        monitors = screen_capturer.monitors
        selected_index = monitor_index
        if selected_index >= len(monitors):
            selected_index = len(monitors) - 1
        if selected_index < 0:
            selected_index = 0
        screenshot = screen_capturer.grab(monitors[selected_index])
        raw_bgra = bytes(screenshot.bgra)
        width = screenshot.width
        height = screenshot.height
    image = Image.frombuffer("RGBA", (width, height), raw_bgra, "raw", "BGRA", 0, 1)
    thumbnail = build_gray_thumbnail(image, thumbnail_width)
    return ScreenFrame(raw_bgra=raw_bgra, width=width, height=height, thumbnail=thumbnail)


def encode_frame_as_png(frame: ScreenFrame) -> bytes:
    image = Image.frombuffer(
        "RGBA", (frame.width, frame.height), frame.raw_bgra, "raw", "BGRA", 0, 1
    ).convert("RGB")
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def build_thumbnail_from_png(png_bytes: bytes, thumbnail_width: int) -> np.ndarray:
    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    return build_gray_thumbnail(image, thumbnail_width)





def save_slide_image(image_bytes: bytes, output_dir: str, slide_number: int) -> str:
    slides_dir = Path(output_dir)
    slides_dir.mkdir(parents=True, exist_ok=True)
    file_path = slides_dir / f"slide_{slide_number:04d}.png"
    atomic_write_bytes(file_path, image_bytes)
    return str(file_path)


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
        self.is_running: bool = False

        session_dir = settings.create_session_directories(session_id)
        self.slides_dir: Path = session_dir / "raw" / "slides"
        self.stream_name: str = f"lecturelens:{session_id}:stream:slide"
        self.control: AgentControl = AgentControl(session_dir, "screen_agent")
        self.slide_counter: int = next_file_index(self.slides_dir, "slide_", ".png") - 1
        self.vision_embedder = get_shared_vision_embedder()
        self.gate: SlideChangeGate = SlideChangeGate(
            change_threshold=settings.slide_diff_threshold,
            pixel_delta=settings.slide_pixel_delta,
            min_interval_seconds=settings.slide_min_interval_seconds,
            embedder=self.vision_embedder
        )
        self.slide_hashes: list[tuple[int, int, np.ndarray]] = []
        self.slide_texts: dict[int, str] = {}
        self.consecutive_capture_errors: int = 0
        self.retry_tasks: set[asyncio.Task] = set()

    def find_revisited_slide(self, difference_hash: int, small_thumbnail: np.ndarray) -> int:
        for slide_number, stored_hash, stored_thumbnail in reversed(self.slide_hashes):
            distance = hamming_distance(difference_hash, stored_hash)
            if distance > self.settings.slide_revisit_hamming_distance:
                continue
            changed_fraction = compute_changed_fraction(
                stored_thumbnail, small_thumbnail, self.settings.slide_pixel_delta
            )
            if changed_fraction <= REVISIT_MAX_CHANGED_FRACTION:
                spatial_sim = compute_spatial_block_similarity(stored_thumbnail, small_thumbnail)
                if spatial_sim >= 0.85:
                    return slide_number
        return 0

    def remember_slide(
        self,
        slide_number: int,
        difference_hash: int,
        small_thumbnail: np.ndarray,
        text: str
    ) -> None:
        self.slide_hashes.append((slide_number, difference_hash, small_thumbnail))
        self.slide_texts[slide_number] = text
        while len(self.slide_hashes) > MAX_REMEMBERED_SLIDES:
            removed_number, _, _ = self.slide_hashes.pop(0)
            if removed_number in self.slide_texts:
                del self.slide_texts[removed_number]

    async def run_ocr(self, png_bytes: bytes) -> str:
        return await self.run_vision_fallback(png_bytes)

    async def run_vision_fallback(self, png_bytes: bytes, max_retries: int = 2) -> str:
        if self.llm_router is None or self.control.stop_requested():
            return ""
        message = build_image_message(SLIDE_VISION_PROMPT, png_bytes, "image/png")
        for attempt in range(max_retries):
            try:
                response = await self.llm_router.ainvoke([message])
                text = response_to_text(response).strip()
                if is_refusal_text(text):
                    return ""
                return text
            except Exception as error:
                logger.error("slide vision fallback failed (attempt %d/%d): %s", attempt + 1, max_retries, error)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return ""

    async def accept_png(
        self,
        png_bytes: bytes,
        thumbnail: np.ndarray,
        capture_epoch: Optional[float] = None
    ) -> SlideCaptured:
        capture_timestamp = capture_epoch if capture_epoch is not None else self.session_clock.get_timestamp()
        self.slide_counter += 1
        slide_number = self.slide_counter
        image_path = await run_in_thread_pool(
            save_slide_image, png_bytes, str(self.slides_dir), slide_number
        )

        difference_hash = compute_difference_hash(thumbnail)
        small_thumbnail = shrink_thumbnail(thumbnail, REVISIT_THUMBNAIL_WIDTH)
        revisited_number = self.find_revisited_slide(difference_hash, small_thumbnail)
        extracted_text = ""
        if revisited_number > 0 and revisited_number in self.slide_texts:
            extracted_text = self.slide_texts[revisited_number]
        elif not self.control.stop_requested():
            extracted_text = await self.run_ocr(png_bytes)
        if len(extracted_text.strip()) > 0:
            self.remember_slide(slide_number, difference_hash, small_thumbnail, extracted_text)

        slide_event = SlideCaptured(
            session_id=self.session_id,
            timestamp_epoch=capture_timestamp,
            slide_number=slide_number,
            monitor_index=self.settings.monitor_index,
            image_file_path=image_path,
            extracted_text=extracted_text,
            revisited_slide_number=revisited_number
        )
        published_id = await self.redis_bus.publish_resilient(self.stream_name, slide_event)
        if published_id is not None:
            self.control.increment_stat("slides_published")
        else:
            self.control.increment_stat("slides_buffered")
        needs_text_retry = (
            len(extracted_text.strip()) == 0
            and revisited_number == 0
            and self.llm_router is not None
            and not self.control.stop_requested()
        )
        if needs_text_retry:
            self.schedule_text_retry(slide_number, image_path, capture_timestamp, difference_hash, small_thumbnail)
        return slide_event

    def schedule_text_retry(
        self,
        slide_number: int,
        image_path: str,
        capture_timestamp: float,
        difference_hash: int,
        small_thumbnail: np.ndarray
    ) -> None:
        task = asyncio.create_task(
            self.retry_slide_text(slide_number, image_path, capture_timestamp, difference_hash, small_thumbnail)
        )
        self.retry_tasks.add(task)
        task.add_done_callback(self.retry_tasks.discard)

    async def retry_slide_text(
        self,
        slide_number: int,
        image_path: str,
        capture_timestamp: float,
        difference_hash: int,
        small_thumbnail: np.ndarray
    ) -> None:
        for delay_seconds in self.settings.vision_retry_delays_seconds:
            await asyncio.sleep(delay_seconds)
            if self.control.stop_requested():
                return
            try:
                png_bytes = await run_in_thread_pool(Path(image_path).read_bytes)
            except OSError:
                return
            recovered_text = await self.run_vision_fallback(png_bytes, max_retries=1)
            if len(recovered_text.strip()) == 0:
                continue
            self.remember_slide(slide_number, difference_hash, small_thumbnail, recovered_text)
            update_event = SlideCaptured(
                session_id=self.session_id,
                timestamp_epoch=capture_timestamp,
                slide_number=slide_number,
                monitor_index=self.settings.monitor_index,
                image_file_path=image_path,
                extracted_text=recovered_text,
                revisited_slide_number=0
            )
            await self.redis_bus.publish_resilient(self.stream_name, update_event)
            self.control.increment_stat("slide_text_recovered")
            return
        logger.warning("slide %d text could not be recovered after retries", slide_number)

    async def ingest_png_bytes(
        self,
        png_bytes: bytes,
        capture_epoch: Optional[float] = None,
        enforce_gate: bool = False
    ) -> Optional[SlideCaptured]:
        thumbnail = await run_in_thread_pool(
            build_thumbnail_from_png, png_bytes, self.settings.slide_thumbnail_width
        )
        if enforce_gate:
            gate_accepted = await run_in_thread_pool(
                self.gate.evaluate, thumbnail, capture_epoch or time.monotonic()
            )
            if not gate_accepted:
                return None
        return await self.accept_png(png_bytes, thumbnail, capture_epoch)

    async def run(self) -> None:
        self.is_running = True
        interval = self.settings.screenshot_interval_seconds
        while self.is_running and not self.control.stop_requested():
            if self.control.is_paused():
                await asyncio.sleep(interval)
                continue
            try:
                frame = await run_in_thread_pool(
                    capture_screen_frame,
                    self.settings.monitor_index,
                    self.settings.slide_thumbnail_width
                )
                self.consecutive_capture_errors = 0
            except Exception as error:
                self.consecutive_capture_errors += 1
                if self.consecutive_capture_errors in (1, 10, 100):
                    logger.error("screen capture failed (%d): %s", self.consecutive_capture_errors, error)
                await asyncio.sleep(min(10.0, interval * self.consecutive_capture_errors))
                continue

            if self.control.stop_requested():
                break

            gate_accepted = await run_in_thread_pool(
                self.gate.evaluate, frame.thumbnail, time.monotonic()
            )
            if gate_accepted:
                if self.control.stop_requested():
                    break
                try:
                    detection_epoch = self.session_clock.get_timestamp()
                    png_bytes = await run_in_thread_pool(encode_frame_as_png, frame)
                    if self.control.stop_requested():
                        break
                    await self.accept_png(png_bytes, frame.thumbnail, capture_epoch=detection_epoch)
                except Exception as error:
                    logger.error("slide processing failed: %s", error)
                    self.control.increment_stat("slide_errors")
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self.is_running = False
        for task in list(self.retry_tasks):
            task.cancel()


async def main() -> None:
    def build_agent(context: RuntimeContext) -> ScreenAgent:
        router = None
        try:
            from lecturelens.router.llm_router import build_router_for_task, load_router_config
            router_config = load_router_config(context.settings.router_config_path)
            router = build_router_for_task(
                "screen_ocr",
                context.redis_bus.client,
                router_config,
                context.settings.ollama_url,
                context.settings.llm_mode,
                context.settings.llm_timeout_seconds
            )
        except Exception as error:
            logger.warning("vision router unavailable, OCR only: %s", error)
        return ScreenAgent(
            settings=context.settings,
            redis_bus=context.redis_bus,
            session_clock=context.session_clock,
            session_id=context.session_id,
            llm_router=router
        )

    await run_standard_agent(build_agent)


if __name__ == "__main__":
    execute_agent_main(main)
