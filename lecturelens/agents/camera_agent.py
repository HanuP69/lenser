import asyncio
import base64
import io
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from langchain_core.messages import HumanMessage
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool, run_in_process_pool
from lecturelens.core.redis_bus import RedisBus
from lecturelens.router.llm_router import build_router_for_task, load_router_config
from lecturelens.schemas.events import BoardCaptured


def capture_camera_frame_bytes(device_index: int) -> bytes:
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video capture device: {device_index}")
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise RuntimeError("Failed to read frame from camera")
    success_encode, encoded_buffer = cv2.imencode(".jpg", frame)
    if not success_encode:
        raise RuntimeError("Failed to encode frame as JPEG")
    return encoded_buffer.tobytes()


def compute_motion_score(current_jpeg_bytes: bytes, previous_jpeg_bytes: bytes) -> float:
    current_array = np.frombuffer(current_jpeg_bytes, dtype=np.uint8)
    previous_array = np.frombuffer(previous_jpeg_bytes, dtype=np.uint8)

    current_frame = cv2.imdecode(current_array, cv2.IMREAD_GRAYSCALE)
    previous_frame = cv2.imdecode(previous_array, cv2.IMREAD_GRAYSCALE)

    if current_frame.shape != previous_frame.shape:
        previous_frame = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))

    diff = cv2.absdiff(current_frame, previous_frame)
    mean_diff = float(np.mean(diff))
    normalized_diff = mean_diff / 255.0
    return normalized_diff


def compute_blur_variance(jpeg_bytes: bytes) -> float:
    buffer_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    gray_frame = cv2.imdecode(buffer_array, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def detect_board_contour_and_warp(jpeg_bytes: bytes) -> bytes:
    buffer_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quadrilateral_contour = None
    for contour in sorted_contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            quadrilateral_contour = approximation
            break

    if quadrilateral_contour is None:
        return jpeg_bytes

    points = quadrilateral_contour.reshape(4, 2).astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(differences)]
    bottom_left = points[np.argmax(differences)]

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    max_height = int(max(height_left, height_right))

    if max_width <= 10 or max_height <= 10:
        return jpeg_bytes

    source_coordinates = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    destination_coordinates = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(source_coordinates, destination_coordinates)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    success_encode, encoded_buffer = cv2.imencode(".jpg", warped)
    if not success_encode:
        return jpeg_bytes
    return encoded_buffer.tobytes()


def save_jpeg_file(jpeg_bytes: bytes, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open(destination_path, "wb") as file_handle:
        file_handle.write(jpeg_bytes)


class CameraAgent:
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
        self.stream_name: str = f"lecturelens:{session_id}:stream:board"
        self.capture_counter: int = 0
        self.previous_frame_bytes: Optional[bytes] = None
        self.stillness_counter: int = 0
        self.is_running: bool = False

    async def run(self) -> None:
        self.is_running = True
        session_dir = self.settings.create_session_directories(self.session_id)
        raw_board_dir = session_dir / "raw" / "board"

        while self.is_running:
            try:
                current_frame_bytes = await run_in_thread_pool(
                    capture_camera_frame_bytes,
                    self.settings.camera_device_index
                )
            except Exception:
                await asyncio.sleep(self.settings.camera_poll_interval_seconds)
                continue

            if self.previous_frame_bytes is None:
                self.previous_frame_bytes = current_frame_bytes
                await asyncio.sleep(self.settings.camera_poll_interval_seconds)
                continue

            motion_score = compute_motion_score(current_frame_bytes, self.previous_frame_bytes)
            self.previous_frame_bytes = current_frame_bytes

            if motion_score < self.settings.motion_threshold:
                self.stillness_counter += 1
            else:
                self.stillness_counter = 0

            if self.stillness_counter < self.settings.motion_stillness_required_frames:
                await asyncio.sleep(self.settings.camera_poll_interval_seconds)
                continue

            self.stillness_counter = 0

            blur_variance = compute_blur_variance(current_frame_bytes)
            if blur_variance < self.settings.blur_variance_threshold:
                await asyncio.sleep(self.settings.camera_poll_interval_seconds)
                continue

            self.capture_counter += 1
            timestamp_epoch = self.session_clock.get_timestamp()

            raw_file_name = f"board_raw_{self.capture_counter:04d}.jpg"
            raw_file_path = raw_board_dir / raw_file_name
            save_jpeg_file(current_frame_bytes, raw_file_path)

            try:
                warped_bytes = await run_in_process_pool(
                    detect_board_contour_and_warp,
                    current_frame_bytes
                )
            except Exception:
                warped_bytes = current_frame_bytes

            warped_file_name = f"board_warped_{self.capture_counter:04d}.jpg"
            warped_file_path = raw_board_dir / warped_file_name
            save_jpeg_file(warped_bytes, warped_file_path)

            extracted_text = ""
            provider_used = "none"

            if self.llm_router is not None:
                base64_encoded = base64.b64encode(warped_bytes).decode("utf-8")
                prompt_message = HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Transcribe and describe all handwriting, equations, diagrams, and written text visible on this whiteboard or board frame."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_encoded}"
                    }
                ])
                try:
                    response = await self.llm_router.ainvoke([prompt_message])
                    if hasattr(response, "content"):
                        extracted_text = str(response.content)
                    else:
                        extracted_text = str(response)
                    provider_used = "router_chain"
                except Exception:
                    extracted_text = ""

            board_event = BoardCaptured(
                session_id=self.session_id,
                timestamp_epoch=timestamp_epoch,
                image_file_path=str(warped_file_path),
                raw_ocr_text=extracted_text,
                confidence=0.85 if len(extracted_text) > 0 else 0.0,
                provider_used=provider_used
            )

            await self.redis_bus.publish(self.stream_name, board_event)
            await asyncio.sleep(self.settings.camera_poll_interval_seconds)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    settings = Settings()
    session_id = settings.session_name or "default_session"
    clock_file = settings.get_session_output_dir(session_id) / "clock.json"
    if clock_file.exists():
        session_clock = SessionClock.load_from_file(clock_file)
    else:
        session_clock = SessionClock()

    redis_bus = RedisBus(settings.redis_url)

    router_config = load_router_config(settings.router_config_path)
    router = build_router_for_task(
        task_type="board_ocr",
        redis_client=redis_bus.client,
        router_config=router_config,
        ollama_url=settings.ollama_url
    )

    agent = CameraAgent(
        settings=settings,
        redis_bus=redis_bus,
        session_clock=session_clock,
        session_id=session_id,
        llm_router=router
    )
    try:
        await agent.run()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
