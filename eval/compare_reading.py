"""Which way of reading the screen is the best?

The screens of a lecture are read in different ways, and every way is
measured against the notes of the course: the share of the words that
were found, in both directions.

  whole    the whole screen is read
  regions  only the parts that hold text are read
  clean    the whole screen is read and the parts of the window around
           the lecture are taken out

The name of the course and the licence are on every page of the notes
and say nothing about the slide, so they are taken out of the truth as
well. Leaving them in makes a system that leaves them out of the notes
look worse than it is.

    python eval/compare_reading.py --video A_mit_6_0001_spring2018_lecture1
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from score import (  # noqa: E402
    load_page_texts,
    load_readings,
    match_pages_monotonic,
    word_scores,
)

from lecturelens.core.layout_model import KIND_CHROME, get_shared_layout_model  # noqa: E402
from lecturelens.core.local_ocr import get_engine, read_text  # noqa: E402


def frame_at(video: Path, seconds: float) -> Any:
    """One picture of a video."""
    import av

    with av.open(str(video)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        container.seek(int(seconds / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="bgr24")
    raise RuntimeError("the video has no picture at that moment")


def inside(box: Any, region_box: tuple[int, int, int, int]) -> bool:
    """Whether a line lies inside a part of the screen."""
    points = box if isinstance(box[0], (list, tuple)) else [
        [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]
    ]
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    region_left, region_top, region_right, region_bottom = region_box
    centre_x = (left + right) / 2.0
    centre_y = (top + bottom) / 2.0
    return (
        region_left <= centre_x <= region_right
        and region_top <= centre_y <= region_bottom
    )


def repeated_words(pages: list[dict[str, Any]]) -> set[str]:
    """The words that are on nearly every page of the notes."""
    counts: Counter = Counter()
    for page in pages:
        words = set(re.sub(r"[^a-z0-9]+", " ", page["text"].lower()).split())
        counts.update(words)
    limit = 0.6 * len(pages)
    return {word for word, count in counts.items() if count >= limit}


def clean_truth(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pages without the words that are on every page."""
    common = repeated_words(pages)
    for page in pages:
        words = [
            word for word in re.sub(r"[^a-z0-9]+", " ", page["text"].lower()).split()
            if word not in common
        ]
        page["clean_text"] = " ".join(words)
    return pages


def slide_moments(readings: list[dict[str, Any]], matches: list[Optional[int]]) -> list[int]:
    """The pictures where the slide really changed."""
    chosen: list[int] = []
    previous: Optional[int] = None
    for index, match in enumerate(matches):
        if match is None:
            continue
        if previous is not None and match != previous:
            chosen.append(index)
        previous = match
    return chosen


def read_only_the_parts(engine: Any, frame: Any, regions: list[Any]) -> str:
    """The text of the parts of the screen that hold text.

    The capture does not read the screen this way, because a slide whose
    text the model of the layout did not recognise comes out empty. It
    is measured here all the same, to show why.
    """
    from PIL import Image

    from lecturelens.core.layout_model import KIND_FIGURE_TITLE, KIND_TEXT, KIND_TITLE

    wanted = [
        region for region in regions
        if region.kind in (KIND_TEXT, KIND_TITLE, KIND_FIGURE_TITLE)
    ]
    if not wanted:
        return ""
    picture = Image.fromarray(frame)
    texts: list[str] = []
    for region in wanted:
        left, top, right, bottom = region.box
        if right - left < 8 or bottom - top < 8:
            continue
        try:
            output = engine(np.array(picture.crop((left, top, right, bottom))))
        except Exception:
            continue
        texts.extend(list(getattr(output, "txts", None) or []))
    return " ".join(texts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="A_mit_6_0001_spring2018_lecture1")
    parser.add_argument("--limit", type=int, default=0, help="only the first N slides")
    args = parser.parse_args()

    from score import video_entry

    manifest = json_manifest()
    entry = video_entry(manifest, args.video)
    data_root = Path(manifest["data_root"])
    video = data_root / entry["file"]

    readings = load_readings(manifest, args.video)["readings"]
    pages = load_page_texts(entry["ground_truth"].get("sources", []), data_root)
    if not pages:
        print("there are no notes for this video, so the reading cannot be measured")
        return 1
    matches = match_pages_monotonic(readings, pages)
    pages = clean_truth(pages)
    chosen = slide_moments(readings, matches)
    if args.limit > 0:
        chosen = chosen[: args.limit]
    print(f"{len(pages)} pages of the notes, {len(chosen)} slide changes")

    model = get_shared_layout_model()
    if not model.is_available():
        print("the model of the layout is not available, so the window cannot be taken out")
        return 1

    totals = {
        "whole": [0.0, 0.0, 0.0],
        "regions": [0.0, 0.0, 0.0],
        "clean": [0.0, 0.0, 0.0],
    }
    counted = 0
    started = time.monotonic()

    for number, index in enumerate(chosen, start=1):
        reading = readings[index]
        truth = pages[matches[index]]["clean_text"]
        frame = frame_at(video, float(reading["t"]))
        regions = model.detect(frame)

        whole = read_text(frame)
        engine = get_engine()
        region_reading = (
            read_only_the_parts(engine, frame, regions) if engine is not None else ""
        )

        kept: list[str] = []
        for line, box in zip(whole.get("lines", []), reading.get("boxes", [])):
            if any(
                inside(box, region.box)
                for region in regions
                if region.kind == KIND_CHROME
            ):
                continue
            kept.append(line["text"])
        clean = " ".join(kept)

        for name, text in (
            ("whole", whole.get("text", "")),
            ("regions", region_reading),
            ("clean", clean),
        ):
            scores = word_scores(text, truth)
            totals[name] = [
                totals[name][0] + scores[0],
                totals[name][1] + scores[1],
                totals[name][2] + scores[2],
            ]
        counted += 1
        print(
            f"  {number:2}/{len(chosen)}  t={float(reading['t']):7.0f}s  "
            f"parts={len(regions):2}  "
            f"whole={len(whole.get('text', '').split()):3}w  "
            f"regions={len(region_reading.split()):3}w  "
            f"clean={len(clean.split()):3}w",
            flush=True
        )

    print()
    print(f"{counted} slides, {time.monotonic() - started:.0f}s")
    for name in ("whole", "regions", "clean"):
        precision, recall, f1 = totals[name]
        print(
            f"  {name:8}  precision {precision / counted:.4f}  "
            f"recall {recall / counted:.4f}  f1 {f1 / counted:.4f}"
        )
    return 0


def json_manifest() -> dict[str, Any]:
    import json

    with open(REPO / "eval" / "manifest.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    sys.exit(main())
