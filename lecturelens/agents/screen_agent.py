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

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import AgentControl, RuntimeContext, execute_agent_main, run_standard_agent
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
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
from lecturelens.core.vision_embedder import get_shared_vision_embedder
from lecturelens.router.messages import build_image_message, response_to_text
from lecturelens.schemas.events import SlideCaptured

logger = get_logger("lecturelens.screen_agent")

# There is no local OCR engine in the pipeline: the slide text comes from a
# vision model. The Tesseract configuration that used to live here was never
# referenced by anything.
MAX_REMEMBERED_SLIDES = 200
REVISIT_THUMBNAIL_WIDTH = 192
# The width of the picture of a diagram that goes into the notes. A
# picture that is wider is made smaller, so that the notes stay small.
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


def save_layout_debug_overlay(
    image_bytes: bytes,
    regions: list[Any],
    output_path: Path
) -> None:
    try:
        from PIL import ImageDraw

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        colors = {
            "figure": (0, 200, 0),
            "title": (0, 120, 255),
            "figure_title": (0, 180, 255),
            "chrome": (255, 0, 0),
            "text": (160, 160, 160),
            "code": (255, 165, 0),
            "formula": (180, 0, 180),
        }
        for region in regions:
            color = colors.get(getattr(region, "kind", ""), (120, 120, 120))
            draw.rectangle(region.box, outline=color, width=2)
            label = f"{getattr(region, 'label', '')}:{getattr(region, 'score', 0.0):.2f}"
            draw.text((region.box[0] + 2, max(0, region.box[1] - 12)), label, fill=color)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
    except Exception as error:
        logger.debug("saving debug overlay failed: %s", error)


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


def capture_screen_frame(
    monitor_index: int,
    thumbnail_width: int,
    capture_target: str = "screen",
    window_handle: Optional[int] = None
) -> ScreenFrame:
    """Take one frame of what is being recorded.

    With ``capture_target`` set to "window" the single window with that
    handle is recorded, and the rest of the desktop is left alone.
    """
    if capture_target == "window" and window_handle:
        from lecturelens.core.window_capture import capture_window_bgra

        raw_bgra, width, height = capture_window_bgra(window_handle)
    else:
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
    )
    return encode_image_as_png(image)


def encode_image_as_png(image: Image.Image) -> bytes:
    output_buffer = io.BytesIO()
    image.convert("RGB").save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def crop_of_the_change(
    frame: ScreenFrame,
    changed_region: Optional[tuple[int, int, int, int]]
) -> Optional[bytes]:
    """The part of the picture that changed, as a picture.

    The model is shown this part and not the whole screen. The request is
    then a small part of the size of the whole screen, and the small
    text of a slide is read properly, because the model does not have to
    make it small first.
    """
    if changed_region is None:
        return None
    thumbnail_width = frame.thumbnail.shape[1]
    if thumbnail_width <= 0 or frame.width <= 0 or frame.height <= 0:
        return None
    scale = frame.width / float(thumbnail_width)
    left, top, right, bottom = changed_region
    left = max(0, min(frame.width - 1, int(left * scale)))
    top = max(0, min(frame.height - 1, int(top * scale)))
    right = max(left + 1, min(frame.width, int(right * scale)))
    bottom = max(top + 1, min(frame.height, int(bottom * scale)))
    if (right - left) < 32 or (bottom - top) < 32:
        return None
    image = Image.frombuffer(
        "RGBA", (frame.width, frame.height), frame.raw_bgra, "raw", "BGRA", 0, 1
    )
    return encode_image_as_png(image.crop((left, top, right, bottom)))


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
        self.session_dir: Path = session_dir
        self.slides_dir: Path = session_dir / "raw" / "slides"
        # The pictures of the diagrams, which go into the notes.
        self.figures_dir: Path = session_dir / "assets"
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
        # How many slides are still waiting for their text.
        self._pending_text_extractions: int = 0
        # The window that is recorded, when a single window is recorded.
        self.capture_target: str = getattr(settings, "capture_target", "screen") or "screen"
        self.window_handle: Optional[int] = getattr(settings, "window_handle", None)
        self._figure_box_history: list[tuple[float, float, float, float]] = []
        self.last_figure_phash: Optional[int] = None
        self.last_figure_id: str = ""
        self.last_figure_path: str = ""
        self.slide_figures: dict[int, dict[str, str]] = {}

    def resolve_window_handle(self) -> Optional[int]:
        """Find the window that is recorded.

        The handle is used when the window is still there. Otherwise the
        title is looked up again, because the window may have been closed
        and opened again, which gives it a new handle.
        """
        if self.capture_target != "window":
            return None
        from lecturelens.core.window_capture import find_window, is_window_alive

        handle = self.window_handle
        if handle and is_window_alive(handle):
            return handle

        title = getattr(self.settings, "window_title", None)
        if not title:
            logger.error(
                "the window that was recorded is gone and no title is known, "
                "falling back to the whole screen"
            )
            self.capture_target = "screen"
            return None

        window = find_window(title)
        if window is None:
            logger.error(
                "the window %r is gone, falling back to the whole screen", title
            )
            self.capture_target = "screen"
            return None
        logger.info("the window %r was reopened, using its new handle", title)
        self.window_handle = window.handle
        return window.handle

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

    def _is_repeating_chrome(
        self,
        box: tuple[int, int, int, int],
        image_size: tuple[int, int]
    ) -> bool:
        width, height = image_size
        if width <= 0 or height <= 0:
            return False
        norm = (box[0] / float(width), box[1] / float(height), box[2] / float(width), box[3] / float(height))
        matches = 0
        for prev in self._figure_box_history:
            x_left = max(norm[0], prev[0])
            y_top = max(norm[1], prev[1])
            x_right = min(norm[2], prev[2])
            y_bottom = min(norm[3], prev[3])
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = (norm[2] - norm[0]) * (norm[3] - norm[1])
                area2 = (prev[2] - prev[0]) * (prev[3] - prev[1])
                union = area1 + area2 - intersection
                if union > 0 and (intersection / union) >= 0.80:
                    matches += 1
        self._figure_box_history.append(norm)
        if len(self._figure_box_history) > 100:
            self._figure_box_history.pop(0)
        return matches >= 2

    def save_the_figure(
        self,
        png_bytes: bytes,
        regions: list[Any],
        slide_number: int
    ) -> str:
        """The picture of the diagram on the screen, saved on its own.

        A description of a diagram teaches nothing. The picture itself is
        what the student needs, so it is kept next to the notes.
        """
        from lecturelens.core.layout_model import KIND_FIGURE

        figures = [region for region in regions if getattr(region, "kind", "") == KIND_FIGURE]
        if not figures:
            return ""

        try:
            image = Image.open(io.BytesIO(png_bytes))
            image.load()
        except Exception as error:
            logger.warning("failed to open slide image for figure extraction: %s", error)
            return ""

        width, height = image.size
        total_frame_area = width * height

        candidates = []
        for figure in figures:
            if figure.width < 120 or figure.height < 120:
                continue
            if figure.area < total_frame_area * 0.04:
                continue
            if getattr(figure, "score", 1.0) < 0.65:
                continue
            if self._is_repeating_chrome(figure.box, (width, height)):
                continue
            candidates.append(figure)

        if not candidates:
            return ""

        figure = max(candidates, key=lambda region: region.area)
        try:
            crop = image.crop(figure.box)
            if crop.width > FIGURE_MAX_WIDTH:
                crop_height = max(1, int(crop.height * FIGURE_MAX_WIDTH / crop.width))
                crop = crop.resize((FIGURE_MAX_WIDTH, crop_height), Image.LANCZOS)

            crop_io = io.BytesIO()
            crop.convert("RGB").save(crop_io, format="PNG")
            crop_bytes = crop_io.getvalue()
            sha_hash = hashlib.sha256(crop_bytes).hexdigest()[:16]
            figure_id = f"fig_{sha_hash}"
            self.last_figure_id = figure_id

            gray_crop = build_gray_thumbnail(crop, 32)
            current_phash = compute_difference_hash(gray_crop)

            assets_dir = self.figures_dir
            assets_dir.mkdir(parents=True, exist_ok=True)
            path = assets_dir / f"figure_{slide_number:04d}.png"
            id_path = assets_dir / f"{figure_id}.png"

            if self.last_figure_phash is not None:
                dist = hamming_distance(self.last_figure_phash, current_phash)
                if dist <= 4:
                    self.control.increment_stat("slide_figure_deduped")

            self.last_figure_phash = current_phash
            self.last_figure_path = str(path)

            atomic_write_bytes(path, crop_bytes)
            if not id_path.exists():
                atomic_write_bytes(id_path, crop_bytes)
        except Exception as error:
            logger.warning("the picture of the diagram could not be saved: %s", error)
            return ""
        return str(path)

    async def caption_figure_with_vision(
        self,
        crop_png: bytes,
        fallback_caption: str = ""
    ) -> str:
        if self.llm_router is None or self.control.stop_requested():
            return fallback_caption
        prompt = (
            "Provide a one-line concise title or caption (under 12 words) describing this diagram or chart. "
            "Do not include quotes or formatting. Return only the caption."
        )
        try:
            message = build_image_message(prompt, crop_png, "image/png")
            response = await self.llm_router.ainvoke([message])
            text = response_to_text(response).strip()
            if text and not is_refusal_text(text):
                cleaned = re.sub(r"[\r\n]+", " ", text).strip().strip('"\'')
                if len(cleaned) > 160:
                    cleaned = cleaned[:157].rstrip() + "..."
                return cleaned
        except Exception as error:
            logger.debug("vision captioning for diagram failed: %s", error)
        return fallback_caption

    def caption_of_the_figure(
        self,
        regions: list[Any],
        lines: Optional[list[dict[str, Any]]] = None,
        slide_text: str = ""
    ) -> str:
        """The caption of the picture: the title of the slide.

        A caption that is made up, such as "the picture on the screen",
        teaches the reader nothing. The title of the slide says what the
        diagram is about, so it is used as the caption.
        """
        from lecturelens.core.layout_model import KIND_FIGURE_TITLE, KIND_TITLE

        titles = [region for region in regions if region.kind == KIND_TITLE]
        if not titles:
            # A slide without a title still has a label under its
            # picture, which says what the picture is.
            titles = [region for region in regions if region.kind == KIND_FIGURE_TITLE]
        caption = self._words_of_the_titles(titles, lines)
        if not caption:
            # A slide with neither a title nor a label: the first words
            # of the slide say what it is about.
            caption = slide_text.strip()
        if len(caption) > 160:
            caption = caption[:157].rstrip() + "..."
        return caption

    def _words_of_the_titles(
        self,
        titles: list[Any],
        lines: Optional[list[dict[str, Any]]]
    ) -> str:
        """The words that were read inside the titles."""
        if not titles:
            return ""
        # The title is the one that is highest on the screen. A title that
        # sits on the same band belongs to it and is read with it.
        top = min(titles, key=lambda region: region.box[1])
        band_bottom = top.box[1] + max(top.height, 40)
        wanted = [
            region for region in titles
            if region.box[1] <= band_bottom
        ]

        words: list[str] = []
        for line in lines or []:
            box = line.get("box")
            if box is None:
                continue
            centre_x = (box[0] + box[2]) / 2.0
            centre_y = (box[1] + box[3]) / 2.0
            inside = any(
                region.box[0] <= centre_x <= region.box[2]
                and region.box[1] <= centre_y <= region.box[3]
                for region in wanted
            )
            if not inside:
                continue
            text = str(line.get("text", "")).strip()
            if text:
                words.append(text)
        return " ".join(words).strip()

    async def read_text_locally(
        self,
        png_bytes: bytes,
        regions: Optional[list[Any]] = None
    ) -> dict[str, Any]:
        """The text of a picture, read on the machine.

        The reader is not there on every machine, and then nothing is
        read and the model is asked as it was before.
        """
        if not self.settings.screen_local_ocr:
            return {"text": "", "lines": []}
        try:
            import io as _io

            from PIL import Image

            from lecturelens.core.local_ocr import read_text

            image = Image.open(_io.BytesIO(png_bytes))
            image.load()
            result = await run_in_thread_pool(read_text, np.array(image), regions)
        except Exception as error:
            logger.warning("the local reading of the text failed: %s", error)
            return {"text": "", "lines": []}
        result["text"] = str(result.get("text", "")).strip()
        return result

    async def find_the_parts(self, png_bytes: bytes) -> list[Any]:
        """The parts of the screen: the text, the pictures, the chrome.

        The model that finds them is not there on every machine, and then
        the whole screen is read as it was before.
        """
        if not self.settings.screen_layout_model:
            return []
        try:
            import io as _io

            from PIL import Image

            from lecturelens.core.layout_model import get_shared_layout_model

            model = get_shared_layout_model()
            if not model.is_available():
                return []
            image = Image.open(_io.BytesIO(png_bytes))
            image.load()
            return await run_in_thread_pool(model.detect, np.array(image))
        except Exception as error:
            logger.warning("the parts of the screen could not be found: %s", error)
            return []

    async def run_ocr(self, png_bytes: bytes) -> str:
        """Extract the text of a slide.

        This runs a vision model, not a local OCR engine, so the name is
        misleading. The local Tesseract configuration that used to sit next
        to it was never referenced by anything.
        """
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

    def can_work_in_the_background(self) -> bool:
        """Whether a slide is worth waiting for.

        The parts of the screen are found and the picture of the diagram
        is kept on the machine, so a slide is worth waiting for even when
        there is no model to ask. The model is only needed for a slide
        whose text the machine could not read.
        """
        if self.llm_router is not None:
            return True
        return self.settings.screen_local_ocr or self.settings.screen_layout_model

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
        image_path = await run_in_thread_pool(
            save_slide_image, png_bytes, str(self.slides_dir), slide_number
        )

        difference_hash = compute_difference_hash(thumbnail)
        small_thumbnail = shrink_thumbnail(thumbnail, REVISIT_THUMBNAIL_WIDTH)
        revisited_number = self.find_revisited_slide(difference_hash, small_thumbnail)
        extracted_text = ""
        if revisited_number > 0 and revisited_number in self.slide_texts:
            extracted_text = self.slide_texts[revisited_number]

        regions = await self.find_the_parts(png_bytes)
        figure_path = ""
        figure_caption = ""
        figure_id = ""
        if regions:
            figure_path = await run_in_thread_pool(
                self.save_the_figure, png_bytes, regions, slide_number
            )
            debug_path = self.session_dir / "debug" / "layout" / f"slide_{slide_number:04d}_layout.png"
            await run_in_thread_pool(
                save_layout_debug_overlay, png_bytes, regions, debug_path
            )

        local = await self.read_text_locally(png_bytes, regions)
        local_text = str(local.get("text", "")).strip()
        if len(extracted_text.strip()) == 0 and len(local_text) > 0:
            extracted_text = local_text
            self.control.increment_stat("slide_text_read_locally")

        if figure_path:
            figure_caption = self.caption_of_the_figure(
                regions, local.get("lines"), extracted_text
            )
            figure_id = getattr(self, "last_figure_id", "") or f"fig_{slide_number:04d}"
            self.control.increment_stat("slide_figure_kept")

        self.slide_figures[slide_number] = {
            "path": figure_path,
            "caption": figure_caption,
            "figure_id": figure_id
        }

        if len(extracted_text.strip()) > 0:
            self.remember_slide(slide_number, difference_hash, small_thumbnail, extracted_text)

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

        needs_text_extraction = (
            len(extracted_text.strip()) == 0
            and revisited_number == 0
            and self.can_work_in_the_background()
            and not self.control.stop_requested()
        )
        if needs_text_extraction:
            if figure_path:
                published_id = await self.redis_bus.publish_resilient(self.stream_name, slide_event)
                if published_id is not None:
                    self.control.increment_stat("slides_published")
                else:
                    self.control.increment_stat("slides_buffered")
            self.schedule_text_extraction(
                slide_number, image_path, capture_timestamp, difference_hash,
                small_thumbnail, crop_png if crop_png is not None else png_bytes
            )
            self._pending_text_extractions += 1
            self.control.set_stat("slides_awaiting_text", self._pending_text_extractions)
            return slide_event

        published_id = await self.redis_bus.publish_resilient(self.stream_name, slide_event)
        if published_id is not None:
            self.control.increment_stat("slides_published")
        else:
            self.control.increment_stat("slides_buffered")
        return slide_event

    def schedule_text_extraction(
        self,
        slide_number: int,
        image_path: str,
        capture_timestamp: float,
        difference_hash: int,
        small_thumbnail: np.ndarray,
        png_bytes: bytes
    ) -> None:
        task = asyncio.create_task(
            self.extract_and_publish_slide_text(
                slide_number, image_path, capture_timestamp, difference_hash,
                small_thumbnail, png_bytes
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
        png_bytes: bytes
    ) -> None:
        """Read a slide in the background and publish it once.

        Three things happen here, and all of them are done on the machine
        where they are free:

        1. The parts of the screen are found: the text, the pictures and
           the parts of the window around the lecture.
        2. The picture of the diagram is saved on its own. A description
           of a diagram teaches nothing, so the picture itself is kept.
        3. The text is read on the machine, and only the parts of the
           screen that hold text are read. The title bar and the bar at
           the bottom of the window are left out.

        The model is only asked when the machine could not read the text,
        which is the case for a picture or a formula.

        The image is read back from the file for every attempt instead of
        being held in memory. A backlog of pending slides would otherwise keep
        every screenshot alive for the whole retry schedule.
        """
        recovered_text = ""
        fig_info = self.slide_figures.get(slide_number, {})
        figure_path = fig_info.get("path", "")
        figure_caption = fig_info.get("caption", "")
        figure_id = fig_info.get("figure_id", "")

        if self.llm_router is not None:
            self.control.increment_stat("slide_text_needs_model")
            delays = [0.0] + list(self.settings.vision_retry_delays_seconds)
            for index, delay_seconds in enumerate(delays):
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                if self.control.stop_requested():
                    return
                recovered_text = await self.run_vision_fallback(png_bytes, max_retries=1)
                if len(recovered_text.strip()) > 0:
                    break
                try:
                    payload = await run_in_thread_pool(Path(image_path).read_bytes)
                    recovered_text = await self.run_vision_fallback(payload, max_retries=1)
                    if len(recovered_text.strip()) > 0:
                        break
                except OSError:
                    pass

        if len(recovered_text.strip()) > 0:
            self.remember_slide(slide_number, difference_hash, small_thumbnail, recovered_text)
            self.control.increment_stat("slide_text_recovered")
        else:
            logger.warning("slide %d text could not be recovered after retries", slide_number)
            self.control.increment_stat("slide_text_unrecovered")
        self._pending_text_extractions = max(0, self._pending_text_extractions - 1)
        self.control.set_stat("slides_awaiting_text", self._pending_text_extractions)

        if figure_path and not figure_caption and recovered_text:
            figure_caption = recovered_text.strip()[:157]

        update_event = SlideCaptured(
            session_id=self.session_id,
            timestamp_epoch=capture_timestamp,
            slide_number=slide_number,
            monitor_index=self.settings.monitor_index,
            image_file_path=image_path,
            extracted_text=recovered_text,
            revisited_slide_number=0,
            figure_file_path=figure_path,
            figure_caption=figure_caption,
            figure_id=figure_id
        )
        await self.redis_bus.publish_resilient(self.stream_name, update_event)

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
            # The gate is a rate limiter and must always use one clock base.
            # The capture epoch is a wall-clock timestamp used for the event
            # itself; feeding it into the gate froze slide capture for the
            # rest of the session.
            gate_accepted = await run_in_thread_pool(
                self.gate.evaluate, thumbnail, time.monotonic()
            )
            if not gate_accepted:
                return None
        return await self.accept_png(png_bytes, thumbnail, capture_epoch)

    async def run(self) -> None:
        self.is_running = True
        interval = self.settings.screenshot_interval_seconds
        if self.capture_target == "window":
            resolved = await run_in_thread_pool(self.resolve_window_handle)
            if resolved:
                logger.info("recording the window %s", resolved)
            else:
                logger.info("recording the whole screen")
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
                    # The part of the screen that changed. The model is
                    # shown that and not the whole screen.
                    changed_region = self.gate.last_changed_region
                    crop_png = await run_in_thread_pool(
                        crop_of_the_change, frame, changed_region
                    )
                    if self.control.stop_requested():
                        break
                    await self.accept_png(
                        png_bytes,
                        frame.thumbnail,
                        capture_epoch=detection_epoch,
                        changed_region=changed_region,
                        crop_png=crop_png
                    )
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
                context.settings.llm_timeout_seconds,
                context.settings.local_model
            )
        except Exception as error:
            # There is no local OCR fallback, so the slides will be captured
            # without any text.
            logger.warning("vision router unavailable, slides will have no text: %s", error)
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
