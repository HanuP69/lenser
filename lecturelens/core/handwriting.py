"""Handwriting and board detection and transcription for LectureLens."""

import re
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

BOARD_VISION_PROMPT = (
    "You are an expert academic transcriber for chalkboard and whiteboard lectures. "
    "Transcribe all legible handwriting, headings, formulas, chemical equations, "
    "and diagrams visible on this board. "
    "Write mathematical expressions and chemical notations in LaTeX ($...$ inline, $$...$$ display). "
    "If a chemical reaction or structure is drawn, describe or transcribe its reagents and pathways. "
    "Preserve bullet points and structure as clean Markdown. "
    "Output the transcription only."
)

CHROME_KEYWORDS: Tuple[str, ...] = (
    "play",
    "pause",
    "mute",
    "unmute",
    "volume",
    "youtube",
    "subscribe",
    "fullscreen",
    "theater",
    "settings",
    "speed",
    "quality",
    "playback",
    "auto",
    "skip",
    "live",
)


def detect_board_surface(image: np.ndarray) -> Tuple[bool, str]:
    """Detect whether an image represents a physical chalkboard or whiteboard.

    Returns a tuple of (is_board, board_kind).
    """
    if image is None or image.size == 0:
        return False, "none"

    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = bgr.shape[0] * bgr.shape[1]

    green_mask = (h >= 32) & (h <= 92) & (s >= 18) & (v >= 20) & (v <= 170)
    green_ratio = float(np.count_nonzero(green_mask)) / total_pixels
    if green_ratio >= 0.22:
        return True, "chalkboard_green"

    dark_mask = (s <= 50) & (v >= 10) & (v <= 115)
    dark_ratio = float(np.count_nonzero(dark_mask)) / total_pixels
    if dark_ratio >= 0.45:
        return True, "chalkboard_slate"

    white_mask = (v >= 170) & (s <= 40)
    white_ratio = float(np.count_nonzero(white_mask)) / total_pixels
    if white_ratio >= 0.40:
        white_pixels = v[white_mask]
        if len(white_pixels) > 50 and float(np.std(white_pixels)) > 3.0:
            edges = cv2.Canny(v, 50, 150)
            edge_ratio = float(np.count_nonzero(edges)) / total_pixels
            if edge_ratio >= 0.003:
                return True, "whiteboard"

    return False, "none"


def filter_chrome_lines(
    lines: List[dict[str, Any]],
    image_height: int,
    image_width: int
) -> List[dict[str, Any]]:
    """Filter out player controls and window chrome from local OCR lines."""
    if not lines:
        return []

    clean_lines: List[dict[str, Any]] = []
    top_margin = image_height * 0.08
    bottom_margin = image_height * 0.90

    for line in lines:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        lowered = text.lower()
        box = line.get("box")
        if box is not None and len(box) >= 4:
            centre_y = (box[1] + box[3]) / 2.0
            if (centre_y < top_margin or centre_y > bottom_margin) and any(
                keyword in lowered for keyword in CHROME_KEYWORDS
            ):
                continue
            if centre_y > bottom_margin and re.search(r"^\d{1,2}:\d{2}", lowered):
                continue
        elif any(keyword == lowered for keyword in CHROME_KEYWORDS):
            continue

        clean_lines.append(line)

    return clean_lines


def is_board_wipe(
    previous_thumbnail: Optional[np.ndarray],
    current_thumbnail: np.ndarray,
    wipe_threshold: float = 0.25
) -> bool:
    """Detect whether a significant portion of a board has been erased."""
    if previous_thumbnail is None or current_thumbnail is None:
        return False
    if previous_thumbnail.shape != current_thumbnail.shape:
        return False
    diff = np.abs(previous_thumbnail.astype(float) - current_thumbnail.astype(float))
    changed_ratio = float(np.mean(diff > 25))
    return changed_ratio >= wipe_threshold
