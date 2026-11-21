import asyncio
import hashlib
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from lecturelens.agents.base_agent import BaseAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import AgentControl, RuntimeContext, execute_agent_main, run_standard_agent
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.fileio import atomic_write_bytes, next_file_index
from lecturelens.core.image_utils import (
    SlideChangeGate,
    compute_changed_fraction,
    compute_difference_hash,
    compute_edge_count,
    compute_spatial_block_similarity,
    hamming_distance,
    is_content_building_on_top
)
from lecturelens.core.drift_filter import is_refusal_text
from lecturelens.core.handwriting import BOARD_VISION_PROMPT, detect_board_surface, filter_chrome_lines
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.vision_embedder import get_shared_vision_embedder
from lecturelens.router.messages import build_image_message, response_to_text
from lecturelens.router.usage import stage as usage_stage
from lecturelens.schemas.events import SlideCaptured

logger = get_logger("lecturelens.screen_agent")

MAX_REMEMBERED_SLIDES = 200
REVISIT_THUMBNAIL_WIDTH = 192
FIGURE_MAX_WIDTH = 900
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
    scale = thumbnail_width / float(max(1, image.size[0]))
    target_height = max(1, int(image.size[1] * scale))
    return np.array(image.convert("L").resize((thumbnail_width, target_height), Image.BILINEAR), dtype=np.uint8)


def shrink_thumbnail(thumbnail: np.ndarray, target_width: int) -> np.ndarray:
    h, w = thumbnail.shape
    target_height = max(1, int(h * (target_width / float(max(1, w)))))
    return np.array(Image.fromarray(thumbnail).resize((target_width, target_height), Image.BILINEAR), dtype=np.uint8)


def capture_screen_frame(
    monitor_index: int,
    thumbnail_width: int,
    capture_target: str = "screen",
    window_handle: Optional[int] = None
) -> ScreenFrame:
    if capture_target == "window" and window_handle:
        from lecturelens.core.window_capture import capture_window_bgra
        raw_bgra, width, height = capture_window_bgra(window_handle)
    else:
        import mss
        with mss.mss() as capturer:
            monitors = capturer.monitors
            idx = max(0, min(len(monitors) - 1, monitor_index))
            shot = capturer.grab(monitors[idx])
            raw_bgra, width, height = bytes(shot.bgra), shot.width, shot.height
    image = Image.frombuffer("RGBA", (width, height), raw_bgra, "raw", "BGRA", 0, 1)
    return ScreenFrame(raw_bgra=raw_bgra, width=width, height=height, thumbnail=build_gray_thumbnail(image, thumbnail_width))


def encode_frame_as_png(frame: ScreenFrame) -> bytes:
    return encode_image_as_png(Image.frombuffer("RGBA", (frame.width, frame.height), frame.raw_bgra, "raw", "BGRA", 0, 1))


def encode_image_as_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def crop_of_the_change(frame: ScreenFrame, changed_region: Optional[tuple[int, int, int, int]]) -> Optional[bytes]:
    if changed_region is None or frame.thumbnail.shape[1] <= 0 or frame.width <= 0 or frame.height <= 0:
        return None
    scale = frame.width / float(frame.thumbnail.shape[1])
    l, t, r, b = changed_region
    left = max(0, min(frame.width - 1, int(l * scale)))
    top = max(0, min(frame.height - 1, int(t * scale)))
    right = max(left + 1, min(frame.width, int(r * scale)))
    bottom = max(top + 1, min(frame.height, int(b * scale)))
    if (right - left) < 32 or (bottom - top) < 32:
        return None
    return encode_image_as_png(Image.frombuffer("RGBA", (frame.width, frame.height), frame.raw_bgra, "raw", "BGRA", 0, 1).crop((left, top, right, bottom)))


def build_thumbnail_from_png(png_bytes: bytes, thumbnail_width: int) -> np.ndarray:
    return build_gray_thumbnail(Image.open(io.BytesIO(png_bytes)), thumbnail_width)


def save_slide_image(image_bytes: bytes, output_dir: str, slide_number: int) -> str:
    path = Path(output_dir) / f"slide_{slide_number:04d}.png"
    atomic_write_bytes(path, image_bytes)
    return str(path)


class ScreenAgent(BaseAgent):
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str,
        llm_router: Any = None
    ) -> None:
        super().__init__(settings, redis_bus, session_clock, session_id, "screen_agent")
        self.llm_router = llm_router
        self.slides_dir = self.session_dir / "raw" / "slides"
        self.figures_dir = self.session_dir / "assets"
        self.stream_name = f"lecturelens:{session_id}:stream:slide"
        self.slide_counter = next_file_index(self.slides_dir, "slide_", ".png") - 1
        self.vision_embedder = get_shared_vision_embedder()
        self.gate = SlideChangeGate(
            change_threshold=settings.slide_diff_threshold,
            pixel_delta=settings.slide_pixel_delta,
            min_interval_seconds=settings.slide_min_interval_seconds,
            embedder=self.vision_embedder
        )
        self.slide_hashes: list[tuple[int, int, np.ndarray]] = []
        self.slide_texts: dict[int, str] = {}
        self.consecutive_capture_errors: int = 0
        self.retry_tasks: set[asyncio.Task] = set()
        self._pending_text_extractions: int = 0
        self.capture_target: str = getattr(settings, "capture_target", "screen") or "screen"
        self.window_handle: Optional[int] = getattr(settings, "window_handle", None)
        self._figure_box_history: list[tuple[float, float, float, float]] = []
        self.last_figure_phash: Optional[int] = None
        self.last_figure_id: str = ""
        self.last_figure_path: str = ""
        self.slide_figures: dict[int, dict[str, str]] = {}
        self.vision_semaphore = asyncio.Semaphore(1)
        self.last_board_phash: Optional[int] = None
        self.last_board_text: str = ""
        self.current_slide_text: str = ""
        self.current_slide_edges: int = 0
        self.current_slide_emb: Optional[np.ndarray] = None
        self.current_slide_number: int = 0
        self.has_active_slide: bool = False

    def resolve_window_handle(self) -> Optional[int]:
        if self.capture_target != "window":
            return None
        from lecturelens.core.window_capture import find_window, is_window_alive
        if self.window_handle and is_window_alive(self.window_handle):
            return self.window_handle
        title = getattr(self.settings, "window_title", None)
        if not title:
            self.capture_target = "screen"
            return None
        win = find_window(title)
        if win is None:
            self.capture_target = "screen"
            return None
        self.window_handle = win.handle
        return win.handle

    def find_revisited_slide(self, difference_hash: int, small_thumbnail: np.ndarray) -> int:
        for slide_num, stored_hash, stored_thumb in reversed(self.slide_hashes):
            if hamming_distance(difference_hash, stored_hash) <= self.settings.slide_revisit_hamming_distance:
                if compute_changed_fraction(stored_thumb, small_thumbnail, self.settings.slide_pixel_delta) <= REVISIT_MAX_CHANGED_FRACTION:
                    if compute_spatial_block_similarity(stored_thumb, small_thumbnail) >= 0.85:
                        return slide_num
        return 0

    def remember_slide(self, slide_number: int, difference_hash: int, small_thumbnail: np.ndarray, text: str) -> None:
        self.slide_hashes.append((slide_number, difference_hash, small_thumbnail))
        self.slide_texts[slide_number] = text
        while len(self.slide_hashes) > MAX_REMEMBERED_SLIDES:
            rem_num, _, _ = self.slide_hashes.pop(0)
            self.slide_texts.pop(rem_num, None)

    def _is_repeating_chrome(self, box: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
        w, h = image_size
        if w <= 0 or h <= 0:
            return False
        norm = (box[0] / float(w), box[1] / float(h), box[2] / float(w), box[3] / float(h))
        matches = 0
        for prev in self._figure_box_history:
            ix, iy = max(norm[0], prev[0]), max(norm[1], prev[1])
            rx, by = min(norm[2], prev[2]), min(norm[3], prev[3])
            if rx > ix and by > iy:
                inter = (rx - ix) * (by - iy)
                union = (norm[2] - norm[0]) * (norm[3] - norm[1]) + (prev[2] - prev[0]) * (prev[3] - prev[1]) - inter
                if union > 0 and (inter / union) >= 0.80:
                    matches += 1
        self._figure_box_history.append(norm)
        if len(self._figure_box_history) > 100:
            self._figure_box_history.pop(0)
        return matches >= 2

    def save_the_figure(self, png_bytes: bytes, regions: list[Any], slide_number: int) -> str:
        from lecturelens.core.layout_model import KIND_FIGURE
        figs = [r for r in regions if getattr(r, "kind", "") == KIND_FIGURE]
        if not figs:
            return ""
        try:
            img = Image.open(io.BytesIO(png_bytes))
            img.load()
        except Exception:
            return ""
        total_area = img.size[0] * img.size[1]
        cands = [
            f for f in figs
            if f.width >= 120 and f.height >= 120 and (total_area * 0.04 <= f.area <= total_area * 0.60)
            and getattr(f, "score", 1.0) >= 0.65 and not self._is_repeating_chrome(f.box, img.size)
        ]
        if not cands:
            return ""
        fig = max(cands, key=lambda r: r.area)
        if fig.area / float(total_area) > 0.30:
            arr = np.array(img.crop(fig.box).resize((32, 32), Image.LANCZOS)).astype(np.float32) / 255.0
            if float(np.var(arr)) > 0.05 or int(len(np.unique((arr * 15).astype(np.int32).reshape(-1, 3), axis=0))) > 100:
                self.control.increment_stat("figure_rejected_photo")
                return ""
        try:
            crop = img.crop(fig.box)
            if crop.width > FIGURE_MAX_WIDTH:
                crop = crop.resize((FIGURE_MAX_WIDTH, max(1, int(crop.height * FIGURE_MAX_WIDTH / crop.width))), Image.LANCZOS)
            crop_bytes = encode_image_as_png(crop)
            fig_id = f"fig_{hashlib.sha256(crop_bytes).hexdigest()[:16]}"
            curr_phash = compute_difference_hash(build_gray_thumbnail(crop, 32))
            if self.last_figure_phash is not None and self.last_figure_path:
                if hamming_distance(self.last_figure_phash, curr_phash) <= 4:
                    self.control.increment_stat("slide_figure_deduped")
                    return self.last_figure_path
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            path = self.figures_dir / f"figure_{slide_number:04d}.png"
            id_path = self.figures_dir / f"{fig_id}.png"
            atomic_write_bytes(path, crop_bytes)
            if not id_path.exists():
                atomic_write_bytes(id_path, crop_bytes)
            self.last_figure_phash, self.last_figure_path, self.last_figure_id = curr_phash, str(path), fig_id
            return str(path)
        except Exception as error:
            logger.warning("failed to save figure: %s", error)
            return ""

    def caption_of_the_figure(self, regions: list[Any], lines: Optional[list[dict[str, Any]]] = None, slide_text: str = "") -> str:
        from lecturelens.core.layout_model import KIND_FIGURE_TITLE, KIND_TITLE
        titles = [r for r in regions if r.kind == KIND_TITLE] or [r for r in regions if r.kind == KIND_FIGURE_TITLE]
        caption = self._words_of_the_titles(titles, lines) or slide_text.strip()
        return caption[:157].rstrip() + "..." if len(caption) > 160 else caption

    def _words_of_the_titles(self, titles: list[Any], lines: Optional[list[dict[str, Any]]]) -> str:
        if not titles:
            return ""
        top = min(titles, key=lambda r: r.box[1])
        band_bottom = top.box[1] + max(top.height, 40)
        wanted = [r for r in titles if r.box[1] <= band_bottom]
        words = []
        for line in lines or []:
            box = line.get("box")
            if box:
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                if any(r.box[0] <= cx <= r.box[2] and r.box[1] <= cy <= r.box[3] for r in wanted):
                    t = str(line.get("text", "")).strip()
                    if t:
                        words.append(t)
        return " ".join(words).strip()

    async def read_text_locally(self, png_bytes: bytes, regions: Optional[list[Any]] = None) -> dict[str, Any]:
        if not self.settings.screen_local_ocr:
            return {"text": "", "lines": []}
        try:
            from lecturelens.core.local_ocr import read_text
            res = await run_in_thread_pool(read_text, np.array(Image.open(io.BytesIO(png_bytes))), regions)
            res["text"] = str(res.get("text", "")).strip()
            return res
        except Exception:
            return {"text": "", "lines": []}

    async def find_the_parts(self, png_bytes: bytes) -> list[Any]:
        if not self.settings.screen_layout_model:
            return []
        try:
            from lecturelens.core.layout_model import get_shared_layout_model
            model = get_shared_layout_model()
            return await run_in_thread_pool(model.detect, np.array(Image.open(io.BytesIO(png_bytes)))) if model.is_available() else []
        except Exception:
            return []

    async def run_vision_fallback(self, png_bytes: bytes, max_retries: int = 2, prompt: Optional[str] = None) -> str:
        if self.llm_router is None or self.control.stop_requested():
            return ""
        msg = build_image_message(prompt or SLIDE_VISION_PROMPT, png_bytes, "image/png")
        for attempt in range(max_retries):
            try:
                stage_name = "board_vision" if prompt == BOARD_VISION_PROMPT else "slide_vision"
                with usage_stage(stage_name):
                    res = await self.llm_router.ainvoke([msg])
                text = response_to_text(res).strip()
                return "" if is_refusal_text(text) else text
            except Exception as error:
                logger.error("vision fallback failed (attempt %d/%d): %s", attempt + 1, max_retries, error)
                if attempt < max_retries - 1:
                    await asyncio.sleep(20.0 * (attempt + 1))
        return ""

    def can_work_in_the_background(self) -> bool:
        return self.llm_router is not None or self.settings.screen_local_ocr or self.settings.screen_layout_model

    async def accept_png(
        self,
        png_bytes: bytes,
        thumbnail: np.ndarray,
        capture_epoch: Optional[float] = None,
        changed_region: Optional[tuple[int, int, int, int]] = None,
        crop_png: Optional[bytes] = None
    ) -> SlideCaptured:
        capture_timestamp = capture_epoch if capture_epoch is not None else self.session_clock.get_timestamp()
        self.slide_counter += 1
        slide_number = self.slide_counter
        image_path = await run_in_thread_pool(save_slide_image, png_bytes, str(self.slides_dir), slide_number)

        diff_hash = compute_difference_hash(thumbnail)
        small_thumb = shrink_thumbnail(thumbnail, REVISIT_THUMBNAIL_WIDTH)
        revisited_number = self.find_revisited_slide(diff_hash, small_thumb)
        extracted_text = self.slide_texts.get(revisited_number, "") if revisited_number > 0 else ""

        is_board, _ = detect_board_surface(thumbnail)
        if is_board and self.last_board_phash is not None and hamming_distance(self.last_board_phash, diff_hash) <= 3:
            extracted_text = self.last_board_text or extracted_text

        regions = await self.find_the_parts(png_bytes)
        figure_path = await run_in_thread_pool(self.save_the_figure, png_bytes, regions, slide_number) if regions else ""
        local = await self.read_text_locally(png_bytes, regions)
        if is_board and local.get("lines"):
            local["lines"] = filter_chrome_lines(local["lines"], thumbnail.shape[0], thumbnail.shape[1])
            local["text"] = " ".join(str(l.get("text", "")) for l in local["lines"]).strip()
        local_text = str(local.get("text", "")).strip()

        if not extracted_text.strip() and local_text and (not is_board or len(local_text) >= 12):
            extracted_text = local_text
            self.control.increment_stat("slide_text_read_locally")

        figure_caption = self.caption_of_the_figure(regions, local.get("lines"), extracted_text) if figure_path else ""
        figure_id = getattr(self, "last_figure_id", "") or (f"fig_{slide_number:04d}" if figure_path else "")
        if figure_path:
            self.control.increment_stat("slide_figure_kept")

        self.slide_figures[slide_number] = {"path": figure_path, "caption": figure_caption, "figure_id": figure_id}
        if extracted_text.strip():
            self.remember_slide(slide_number, diff_hash, small_thumb, extracted_text)

        slide_event = SlideCaptured(
            session_id=self.session_id,
            timestamp_epoch=capture_timestamp,
            slide_number=slide_number,
            monitor_index=self.settings.monitor_index,
            image_file_path=image_path,
            extracted_text=extracted_text,
            revisited_slide_number=revisited_number,
            figure_file_path=figure_path,
            figure_caption=figure_caption,
            figure_id=figure_id
        )

        needs_text = (
            (not extracted_text.strip() or (is_board and not extracted_text.strip()))
            and (revisited_number == 0 or revisited_number not in self.slide_texts)
            and self.can_work_in_the_background() and not self.control.stop_requested()
        )
        if needs_text:
            if figure_path:
                await self.redis_bus.publish_resilient(self.stream_name, slide_event)
            self.schedule_text_extraction(
                slide_number, image_path, capture_timestamp, diff_hash,
                small_thumb, crop_png if crop_png is not None else png_bytes,
                revisited_slide_number=revisited_number, is_board=is_board
            )
            self._pending_text_extractions += 1
            self.control.set_stat("slides_awaiting_text", self._pending_text_extractions)
            return slide_event

        await self.redis_bus.publish_resilient(self.stream_name, slide_event)
        return slide_event

    def schedule_text_extraction(
        self,
        slide_number: int,
        image_path: str,
        capture_timestamp: float,
        difference_hash: int,
        small_thumbnail: np.ndarray,
        png_bytes: bytes,
        revisited_slide_number: int = 0,
        is_board: bool = False
    ) -> None:
        task = asyncio.create_task(
            self.extract_and_publish_slide_text(
                slide_number, image_path, capture_timestamp, difference_hash,
                small_thumbnail, png_bytes, revisited_slide_number, is_board
            )
        )
        self.retry_tasks.add(task)
        task.add_done_callback(self.retry_tasks.discard)

    async def extract_and_publish_slide_text(
        self,
        slide_number: int,
        image_path: str,
        capture_timestamp: float,
        difference_hash: int,
        small_thumbnail: np.ndarray,
        png_bytes: bytes,
        revisited_slide_number: int = 0,
        is_board: bool = False
    ) -> None:
        async with self.vision_semaphore:
            recovered_text = ""
            fig_info = self.slide_figures.get(slide_number, {})
            figure_path = fig_info.get("path", "")
            figure_caption = fig_info.get("caption", "")
            figure_id = fig_info.get("figure_id", "")
            try:
                if self.llm_router is not None:
                    self.control.increment_stat("slide_text_needs_model")
                    for delay in [0.0] + list(self.settings.vision_retry_delays_seconds):
                        if delay > 0:
                            await asyncio.sleep(delay)
                        if self.control.stop_requested():
                            return
                        recovered_text = await self.run_vision_fallback(png_bytes, max_retries=1, prompt=BOARD_VISION_PROMPT if is_board else None)
                        if recovered_text.strip():
                            break
                if is_board and recovered_text.strip():
                    self.last_board_phash, self.last_board_text = difference_hash, recovered_text
                if recovered_text.strip():
                    self.remember_slide(slide_number, difference_hash, small_thumbnail, recovered_text)
                    if revisited_slide_number > 0 and revisited_slide_number not in self.slide_texts:
                        self.slide_texts[revisited_slide_number] = recovered_text
                    self.control.increment_stat("slide_text_recovered")
                else:
                    self.control.increment_stat("slide_text_unrecovered")
            finally:
                self._pending_text_extractions = max(0, self._pending_text_extractions - 1)
                self.control.set_stat("slides_awaiting_text", self._pending_text_extractions)

        if figure_path and not figure_caption and recovered_text:
            figure_caption = recovered_text.strip()[:157]

        event = SlideCaptured(
            session_id=self.session_id,
            timestamp_epoch=capture_timestamp,
            slide_number=slide_number,
            monitor_index=self.settings.monitor_index,
            image_file_path=image_path,
            extracted_text=recovered_text,
            revisited_slide_number=revisited_slide_number,
            figure_file_path=figure_path,
            figure_caption=figure_caption,
            figure_id=figure_id
        )
        await self.redis_bus.publish_resilient(self.stream_name, event)

    async def ingest_png_bytes(
        self,
        png_bytes: bytes,
        capture_epoch: Optional[float] = None,
        enforce_gate: bool = False,
        gate_now: Optional[float] = None
    ) -> Optional[SlideCaptured]:
        thumbnail = await run_in_thread_pool(build_thumbnail_from_png, png_bytes, self.settings.slide_thumbnail_width)
        if not enforce_gate:
            return await self.accept_png(png_bytes, thumbnail, capture_epoch)

        if not await run_in_thread_pool(self.gate.evaluate, thumbnail, time.monotonic() if gate_now is None else gate_now):
            return None

        curr_edges = compute_edge_count(thumbnail)
        curr_emb = self.vision_embedder.embed(thumbnail) if self.vision_embedder else None

        if not self.has_active_slide:
            event = await self.accept_png(png_bytes, thumbnail, capture_epoch)
            self.has_active_slide = True
            self.current_slide_number = self.slide_counter
            self.current_slide_edges, self.current_slide_emb = curr_edges, curr_emb
            local = await self.read_text_locally(png_bytes)
            self.current_slide_text = str(local.get("text", "")).strip()
            return event

        sim = float(self.vision_embedder.similarity(self.current_slide_emb, curr_emb)) if (self.current_slide_emb is not None and curr_emb is not None) else 1.0
        if sim >= 0.94:
            local = await self.read_text_locally(png_bytes)
            curr_text = str(local.get("text", "")).strip()
            building, _ = is_content_building_on_top(self.current_slide_text, curr_text, self.current_slide_edges, curr_edges)
            if building:
                await run_in_thread_pool(atomic_write_bytes, Path(self.slides_dir / f"slide_{self.current_slide_number:04d}.png"), png_bytes)
                self.current_slide_edges = curr_edges
                if curr_text:
                    self.current_slide_text = curr_text
                    self.slide_texts[self.current_slide_number] = curr_text
                self.current_slide_emb = curr_emb
                return None

        event = await self.accept_png(png_bytes, thumbnail, capture_epoch)
        self.current_slide_number = self.slide_counter
        self.current_slide_edges, self.current_slide_emb = curr_edges, curr_emb
        local = await self.read_text_locally(png_bytes)
        self.current_slide_text = str(local.get("text", "")).strip()
        return event

    async def run(self) -> None:
        self.is_running = True
        interval = self.settings.screenshot_interval_seconds
        if self.capture_target == "window":
            await run_in_thread_pool(self.resolve_window_handle)
        while self.is_running and not self.control.stop_requested():
            if self.control.is_paused():
                await asyncio.sleep(interval)
                continue
            try:
                frame = await run_in_thread_pool(
                    capture_screen_frame,
                    self.settings.monitor_index,
                    self.settings.slide_thumbnail_width,
                    self.capture_target,
                    self.window_handle
                )
                self.consecutive_capture_errors = 0
            except Exception as error:
                self.consecutive_capture_errors += 1
                await asyncio.sleep(min(10.0, interval * self.consecutive_capture_errors))
                continue

            if self.control.stop_requested():
                break

            if await run_in_thread_pool(self.gate.evaluate, frame.thumbnail, time.monotonic()):
                if self.control.stop_requested():
                    break
                try:
                    detection_epoch = self.session_clock.get_timestamp()
                    png_bytes = await run_in_thread_pool(encode_frame_as_png, frame)
                    changed_region = self.gate.last_changed_region
                    crop_png = await run_in_thread_pool(crop_of_the_change, frame, changed_region)
                    if not self.control.stop_requested():
                        await self.accept_png(png_bytes, frame.thumbnail, capture_epoch=detection_epoch, changed_region=changed_region, crop_png=crop_png)
                except Exception as error:
                    logger.error("slide processing failed: %s", error)
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
                "screen_ocr", context.redis_bus.client, router_config,
                context.settings.ollama_url, context.settings.llm_mode,
                context.settings.llm_timeout_seconds, context.settings.local_model
            )
        except Exception as error:
            logger.warning("vision router unavailable: %s", error)
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
