"""Reading the text of the screen on the machine.

The text of a slide is read by a model that is asked over the network.
The model reads it no better than a small reader that runs on the
processor, and the request that carries a whole picture of the screen
is the most expensive part of the run.

This reader runs on the processor, costs nothing and keeps the picture
of the screen on the machine. The model is only asked for the parts it
cannot do.
"""

import threading
from typing import Any, Optional

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.local_ocr")

# The reader is made once. Making it costs a second, and it is used for
# every picture.
_ENGINE: Optional[Any] = None
_ENGINE_FAILED = False
_LOCK = threading.Lock()


def get_engine() -> Optional[Any]:
    """The reader, made the first time it is asked for.

    None is returned when the reader is not there. The capture then
    falls back to the model, which is what it did before.
    """
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    with _LOCK:
        if _ENGINE is not None:
            return _ENGINE
        if _ENGINE_FAILED:
            return None
        try:
            from rapidocr import RapidOCR

            _ENGINE = RapidOCR(
                params={
                    # The runtime uses every core of the machine by
                    # itself. The agents run side by side, so each one is
                    # given a single thread.
                    "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
            )
        except Exception as error:
            _ENGINE_FAILED = True
            logger.warning(
                "the local reader of the text is not available (%s); the model reads "
                "the text of the screen instead",
                error
            )
            return None
    return _ENGINE


def is_available() -> bool:
    """Whether the local reader can be used."""
    return get_engine() is not None


def read_text(image: Any, regions: Optional[list[Any]] = None) -> dict[str, Any]:
    """The text of a picture.

    The words are the ones that were read, the number of the lines and
    the boxes of the lines, so that the parts of the screen can be
    picked out later.

    When the parts of the screen are known, the parts of the window
    around the lecture are left out: the name of the window, the bar at
    the bottom and the name of the speaker in the corner. Those are not
    the lecture, and they are read on every single slide.

    The whole screen is read and the parts of the window are taken out
    afterwards. Reading only the parts that hold text was tried and it
    reads less: a slide whose text the model of the layout did not
    recognise comes out empty.
    """
    engine = get_engine()
    if engine is None:
        return {"text": "", "lines": [], "word_count": 0}
    result = _read_the_whole(engine, image)
    if regions:
        result = _drop_the_window(result, regions)
    return result


def _drop_the_window(result: dict[str, Any], regions: list[Any]) -> dict[str, Any]:
    """The text without the parts of the window around the lecture."""
    from lecturelens.core.layout_model import KIND_CHROME

    bars = [
        region for region in regions
        if getattr(region, "kind", "") == KIND_CHROME
    ]
    if not bars:
        return result
    kept: list[dict[str, Any]] = []
    for line in result.get("lines", []):
        box = line.get("box")
        if box is None:
            kept.append(line)
            continue
        centre_x = (box[0] + box[2]) / 2.0
        centre_y = (box[1] + box[3]) / 2.0
        inside = any(
            region.box[0] <= centre_x <= region.box[2]
            and region.box[1] <= centre_y <= region.box[3]
            for region in bars
        )
        if not inside:
            kept.append(line)
    result["lines"] = kept
    result["text"] = " ".join(str(line.get("text", "")) for line in kept).strip()
    result["word_count"] = len(kept)
    return result


def _read_the_whole(engine: Any, image: Any) -> dict[str, Any]:
    try:
        output = engine(image)
    except Exception as error:
        logger.warning("the local reader failed: %s", error)
        return {"text": "", "lines": [], "word_count": 0}
    texts = list(getattr(output, "txts", None) or [])
    scores = list(getattr(output, "scores", None) or [])
    boxes = getattr(output, "boxes", None)
    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        box = None
        if boxes is not None and index < len(boxes):
            points = [[float(x), float(y)] for x, y in boxes[index]]
            left = min(point[0] for point in points)
            top = min(point[1] for point in points)
            right = max(point[0] for point in points)
            bottom = max(point[1] for point in points)
            box = [round(left, 1), round(top, 1), round(right, 1), round(bottom, 1)]
        lines.append(
            {
                "text": text,
                "score": round(float(scores[index]), 4) if index < len(scores) else 0.0,
                "box": box
            }
        )
    return {
        "text": " ".join(texts),
        "lines": lines,
        "word_count": len(texts)
    }
