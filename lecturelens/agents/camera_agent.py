import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import AgentControl, RuntimeContext, execute_agent_main, run_standard_agent
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_process_pool, run_in_thread_pool
from lecturelens.core.fileio import atomic_write_bytes, next_file_index
from lecturelens.core.image_utils import compute_changed_fraction
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.router.messages import (
    build_image_message,
    extract_provider_name,
    parse_confidence_footer,
    response_to_text
)
from lecturelens.schemas.events import BoardCaptured

logger = get_logger("lecturelens.camera_agent")

BOARD_THUMBNAIL_SIZE = (320, 240)
MAX_WARP_DIMENSION = 2000
MIN_BOARD_ASPECT = 0.5
MAX_BOARD_ASPECT = 4.0
MAX_BOARD_AREA_RATIO = 0.92

BOARD_VISION_PROMPT = (
    "Transcribe everything written or drawn on this whiteboard or paper. "
    "Preserve equations, lists and structure. Describe diagrams in one short line each. "
    "Output only the transcription. On a final separate line write CONFIDENCE: followed by "
    "a number between 0 and 1 for how legible the content was."
)


class CameraSource:
    def __init__(self, device_index: int, warmup_frames: int = 5) -> None:
        self.device_index: int = device_index
        self.warmup_frames: int = warmup_frames
        self.capture: Any = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open camera {self.device_index}")
        for _ in range(self.warmup_frames):
            capture.read()
        self.capture = capture

    def read_jpeg(self) -> bytes:
        if self.capture is None:
            self.open()
        success, frame = self.capture.read()
        if not success or frame is None:
            self.close()
            raise RuntimeError("camera read failed")
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encoded_ok:
            raise RuntimeError("jpeg encode failed")
        return encoded.tobytes()

    def close(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception as error:
                logger.warning("camera release failed: %s", error)
            self.capture = None


def decode_gray(frame_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("cannot decode frame")
    return gray


def compute_motion_score(frame_bytes_a: bytes, frame_bytes_b: bytes) -> float:
    gray_a = decode_gray(frame_bytes_a)
    gray_b = decode_gray(frame_bytes_b)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    absolute_difference = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(absolute_difference)) / 255.0


def compute_blur_score(frame_bytes: bytes) -> float:
    gray = decode_gray(frame_bytes)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_board_thumbnail(frame_bytes: bytes) -> np.ndarray:
    gray = decode_gray(frame_bytes)
    return cv2.resize(gray, BOARD_THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)


def order_quadrilateral_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def find_board_quadrilateral(image: np.ndarray, min_area_ratio: float) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    frame_area = float(height * width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in ranked:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approximation = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(approximation) != 4:
            continue
        area_ratio = cv2.contourArea(approximation) / frame_area
        if area_ratio < min_area_ratio or area_ratio > MAX_BOARD_AREA_RATIO:
            continue
        ordered = order_quadrilateral_points(approximation)
        top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
        left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
        if top_width < 1.0 or left_height < 1.0:
            continue
        aspect = top_width / left_height
        if aspect < MIN_BOARD_ASPECT or aspect > MAX_BOARD_ASPECT:
            continue
        return ordered
    return None


def detect_board_contour_and_warp(image_bytes: bytes, min_area_ratio: float = 0.2) -> bytes:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return image_bytes
    ordered = find_board_quadrilateral(image, min_area_ratio)
    if ordered is None:
        return image_bytes

    top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom_width = float(np.linalg.norm(ordered[2] - ordered[3]))
    left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
    right_height = float(np.linalg.norm(ordered[2] - ordered[1]))
    output_width = int(max(top_width, bottom_width))
    output_height = int(max(left_height, right_height))
    largest = max(output_width, output_height)
    if largest > MAX_WARP_DIMENSION:
        scale = MAX_WARP_DIMENSION / float(largest)
        output_width = int(output_width * scale)
        output_height = int(output_height * scale)
    if output_width < 32 or output_height < 32:
        return image_bytes

    destination = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, transform, (output_width, output_height))
    encoded_ok, encoded = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded_ok:
        return image_bytes
    return encoded.tobytes()


def save_board_image(image_bytes: bytes, output_dir: str, capture_index: int, suffix: str = "corrected") -> str:
    board_dir = Path(output_dir)
    board_dir.mkdir(parents=True, exist_ok=True)
    file_path = board_dir / f"board_{suffix}_{capture_index:04d}.jpg"
    atomic_write_bytes(file_path, image_bytes)
    return str(file_path)


class CameraAgent:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str,
        llm_router: Any = None,
        camera_source: Any = None
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.session_clock: SessionClock = session_clock
        self.session_id: str = session_id
        self.llm_router: Any = llm_router
        self.is_running: bool = False

        session_dir = settings.create_session_directories(session_id)
        self.board_dir: Path = session_dir / "raw" / "board"
        self.stream_name: str = f"lecturelens:{session_id}:stream:board"
        self.control: AgentControl = AgentControl(session_dir, "camera_agent")
        self.capture_counter: int = next_file_index(self.board_dir, "board_raw_", ".jpg") - 1
        self.camera_source: Any = camera_source
        if self.camera_source is None:
            self.camera_source = CameraSource(settings.camera_device_index, settings.camera_warmup_frames)
        self.capture_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        self.previous_frame_bytes: Optional[bytes] = None
        self.stillness_counter: int = 0
        self.last_published_thumbnail: Optional[np.ndarray] = None
        self.last_publish_time: float = 0.0
        self.consecutive_capture_errors: int = 0

    def has_board_changed(self, thumbnail: np.ndarray, now_seconds: float) -> bool:
        if self.last_published_thumbnail is None:
            return True
        if (now_seconds - self.last_publish_time) < self.settings.board_min_interval_seconds:
            return False
        fraction = compute_changed_fraction(self.last_published_thumbnail, thumbnail, 25)
        return fraction >= self.settings.board_change_threshold

    async def read_vision_text(self, image_bytes: bytes) -> tuple[str, float, str]:
        if self.llm_router is None or self.control.stop_requested():
            return "", 0.0, "none"
        message = build_image_message(BOARD_VISION_PROMPT, image_bytes, "image/jpeg")
        try:
            response = await self.llm_router.ainvoke([message])
        except Exception as error:
            logger.error("board vision call failed: %s", error)
            return "", 0.0, "none"
        raw_text = response_to_text(response)
        cleaned_text, confidence = parse_confidence_footer(raw_text)
        provider_name = extract_provider_name(response, "unknown")
        return cleaned_text, confidence, provider_name

    async def process_still_frame(self, frame_bytes: bytes) -> Optional[BoardCaptured]:
        thumbnail = await run_in_thread_pool(build_board_thumbnail, frame_bytes)
        now_seconds = time.monotonic()
        if not self.has_board_changed(thumbnail, now_seconds):
            return None

        self.capture_counter += 1
        capture_index = self.capture_counter
        await run_in_thread_pool(
            save_board_image, frame_bytes, str(self.board_dir), capture_index, "raw"
        )
        try:
            corrected_bytes = await run_in_process_pool(
                detect_board_contour_and_warp, frame_bytes, self.settings.board_min_area_ratio
            )
        except Exception as error:
            logger.error("perspective correction failed, using raw frame: %s", error)
            corrected_bytes = frame_bytes
        corrected_path = await run_in_thread_pool(
            save_board_image, corrected_bytes, str(self.board_dir), capture_index, "corrected"
        )

        if self.control.stop_requested():
            return None

        text, confidence, provider_name = await self.read_vision_text(corrected_bytes)
        self.last_published_thumbnail = thumbnail
        self.last_publish_time = time.monotonic()

        board_event = BoardCaptured(
            session_id=self.session_id,
            timestamp_epoch=self.session_clock.get_timestamp(),
            image_file_path=corrected_path,
            raw_ocr_text=text,
            confidence=confidence,
            provider_used=provider_name
        )
        await self.redis_bus.publish_resilient(self.stream_name, board_event)
        self.control.increment_stat("boards_published")
        return board_event

    async def ingest_jpeg_bytes(self, frame_bytes: bytes) -> Optional[BoardCaptured]:
        blur_score = await run_in_thread_pool(compute_blur_score, frame_bytes)
        if blur_score < self.settings.blur_variance_threshold:
            return None
        return await self.process_still_frame(frame_bytes)

    async def run(self) -> None:
        self.is_running = True
        loop = asyncio.get_running_loop()
        poll_interval = self.settings.camera_poll_interval_seconds
        while self.is_running and not self.control.stop_requested():
            if self.control.is_paused():
                await asyncio.sleep(poll_interval)
                continue
            try:
                frame_bytes = await loop.run_in_executor(
                    self.capture_executor, self.camera_source.read_jpeg
                )
                self.consecutive_capture_errors = 0
            except Exception as error:
                self.consecutive_capture_errors += 1
                if self.consecutive_capture_errors in (1, 10, 100):
                    logger.error("camera capture failed (%d): %s", self.consecutive_capture_errors, error)
                await asyncio.sleep(min(15.0, poll_interval * 2.0 * self.consecutive_capture_errors))
                continue

            if self.control.stop_requested():
                break

            try:
                await self.evaluate_frame(frame_bytes)
            except Exception as error:
                logger.error("camera frame processing failed: %s", error)
                self.control.increment_stat("camera_errors")
            await asyncio.sleep(poll_interval)

        await loop.run_in_executor(self.capture_executor, self.camera_source.close)
        self.capture_executor.shutdown(wait=False)

    async def evaluate_frame(self, frame_bytes: bytes) -> None:
        previous_bytes = self.previous_frame_bytes
        self.previous_frame_bytes = frame_bytes
        if previous_bytes is None:
            return

        motion_score = await run_in_thread_pool(compute_motion_score, previous_bytes, frame_bytes)
        if motion_score > self.settings.motion_threshold:
            self.stillness_counter = 0
            return

        self.stillness_counter += 1
        if self.stillness_counter < self.settings.motion_stillness_required_frames:
            return

        blur_score = await run_in_thread_pool(compute_blur_score, frame_bytes)
        if blur_score < self.settings.blur_variance_threshold:
            return

        await self.process_still_frame(frame_bytes)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    def build_agent(context: RuntimeContext) -> CameraAgent:
        router = None
        try:
            from lecturelens.router.llm_router import build_router_for_task, load_router_config
            router_config = load_router_config(context.settings.router_config_path)
            router = build_router_for_task(
                "board_ocr",
                context.redis_bus.client,
                router_config,
                context.settings.ollama_url,
                context.settings.llm_mode,
                context.settings.llm_timeout_seconds
            )
        except Exception as error:
            logger.warning("vision router unavailable, boards saved without text: %s", error)
        return CameraAgent(
            settings=context.settings,
            redis_bus=context.redis_bus,
            session_clock=context.session_clock,
            session_id=context.session_id,
            llm_router=router
        )

    await run_standard_agent(build_agent)


if __name__ == "__main__":
    execute_agent_main(main)
