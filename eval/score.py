"""Measure the capture.

Two things are measured.

The trigger: which moments of the video the capture decided to look at,
against the moments where the screen really changed. The ground truth of
a change is the place where the page of the notes that fits the picture
changes.

The text: what the capture read, against the text of the notes of that
page. The number is the share of the words that were found, in both
directions, because a slide that shows only half of its text is not a
mistake of the reading.

    python eval/score.py --list
    python eval/score.py --video B_statquest_gradient_descent --trigger
    python eval/score.py --video X --captures captures.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval" / "manifest.json"

# The words that are on every page of a set of notes, such as the name of
# the course. They say nothing about the slide and they make every page
# look the same.
STOP_WORDS = {
    "the", "a", "an", "of", "and", "to", "in", "is", "are", "for", "on",
    "with", "that", "this", "it", "as", "by", "be", "or", "from", "at",
    "we", "you", "can", "not", "have", "has", "was", "were", "will"
}


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        return json.load(handle)


def video_entry(manifest: dict[str, Any], video_id: str) -> dict[str, Any]:
    for entry in manifest["videos"]:
        if entry["id"] == video_id:
            return entry
    raise SystemExit(f"the video {video_id} is not in the manifest")


def load_readings(manifest: dict[str, Any], video_id: str) -> dict[str, Any]:
    path = REPO / "eval" / "readings" / f"{video_id}.json"
    if not path.exists():
        raise SystemExit(
            f"there are no readings for {video_id}. Run:\n"
            f"  python eval/extract_readings.py --video {video_id}"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def words_of(text: str) -> list[str]:
    """The words of a text, with the noise taken out."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return [word for word in cleaned.split() if word and word not in STOP_WORDS]


def word_scores(captured: str, truth: str) -> tuple[float, float, float]:
    """How much of the text was found, in both directions."""
    captured_words = Counter(words_of(captured))
    truth_words = Counter(words_of(truth))
    if not captured_words and not truth_words:
        return 1.0, 1.0, 1.0
    overlap = sum((captured_words & truth_words).values())
    precision = overlap / max(1, sum(captured_words.values()))
    recall = overlap / max(1, sum(truth_words.values()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return precision, recall, f1


def novelty_signal(readings: list[dict[str, Any]], window: int = 6) -> list[float]:
    """How much of the text on the screen is new.

    The text of the last few pictures is the memory of what has already
    been said on the screen. The share of the words that are not in that
    memory is how new the screen is.
    """
    values: list[float] = []
    for index, reading in enumerate(readings):
        current = set(words_of(reading.get("text", "")))
        if not current:
            values.append(0.0)
            continue
        seen: set[str] = set()
        for previous in readings[max(0, index - window):index]:
            seen.update(words_of(previous.get("text", "")))
        if not seen:
            values.append(1.0)
            continue
        values.append(1.0 - len(current & seen) / len(current))
    return values


def fire_times(
    readings: list[dict[str, Any]],
    threshold: float,
    min_gap_seconds: float
) -> list[float]:
    """The moments the capture would have looked at."""
    signal = novelty_signal(readings)
    fired: list[float] = []
    last = -1e9
    for reading, novelty in zip(readings, signal):
        moment = float(reading["t"])
        if novelty >= threshold and (moment - last) >= min_gap_seconds:
            fired.append(moment)
            last = moment
    return fired


def load_page_texts(sources: list[str], data_root: Path) -> list[dict[str, Any]]:
    """The text of every page of the notes."""
    import pymupdf

    pages: list[dict[str, Any]] = []
    for source in sources:
        path = data_root / source
        if not path.exists():
            continue
        document = pymupdf.open(str(path))
        for number, page in enumerate(document, start=1):
            text = " ".join(page.get_text().split())
            pages.append({"source": path.name, "page": number, "text": text})
        document.close()
    return pages


def match_pages(
    readings: list[dict[str, Any]],
    pages: list[dict[str, Any]]
) -> list[Optional[int]]:
    """Which page of the notes fits every picture.

    The best page is the ground truth of that picture. When no page fits
    at all, the screen was not showing a slide, and that is worth
    knowing.
    """
    from rapidfuzz import fuzz

    page_texts = [page["text"].lower() for page in pages]
    matches: list[Optional[int]] = []
    for reading in readings:
        text = reading.get("text", "").lower()
        if not text.strip():
            matches.append(None)
            continue
        scores = [fuzz.token_set_ratio(text, page_text) for page_text in page_texts]
        best = max(range(len(scores)), key=lambda index: scores[index])
        matches.append(best if scores[best] >= 55 else None)
    return matches


def match_pages_monotonic(
    readings: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    threshold: float = 55.0,
    advance_cost: float = 40.0
) -> list[Optional[int]]:
    """Which page of the notes fits every picture, when the pages only go forward.

    The slides of a lecture go forward and never back. Looking at every
    picture on its own gives a wrong answer, because the slides of one
    lecture share much of their text and the wrong one often has more
    words in common.

    The pages are lined up with the pictures the way a row of slides is
    lined up with a video of that lecture. Moving to the next page costs
    something, so the page only moves on when the pictures clearly
    belong to the next one.
    """
    from rapidfuzz import fuzz

    page_texts = [page["text"].lower() for page in pages]
    if not page_texts:
        return [None] * len(readings)

    matches: list[Optional[int]] = [None] * len(readings)
    usable: list[int] = []
    similarities: list[list[float]] = []
    for index, reading in enumerate(readings):
        text = reading.get("text", "").lower()
        if not text.strip():
            continue
        usable.append(index)
        similarities.append(
            [fuzz.token_set_ratio(text, page_text) for page_text in page_texts]
        )

    if not usable:
        return matches

    page_count = len(page_texts)
    # The best total for the pictures so far, when the last one is on the
    # page j.
    previous: list[float] = [float("-inf")] * page_count
    previous[0] = 0.0
    came_from: list[list[int]] = []

    for row in similarities:
        current = [float("-inf")] * page_count
        row_from = [-1] * page_count
        # The best total that ends on a page before this one.
        best_before = float("-inf")
        best_before_page = -1
        for page in range(page_count):
            stay = previous[page]
            advance = best_before - advance_cost
            if advance >= stay:
                current[page] = advance + row[page]
                row_from[page] = best_before_page
            else:
                current[page] = stay + row[page]
                row_from[page] = page
            if previous[page] > best_before:
                best_before = previous[page]
                best_before_page = page
        previous = current
        came_from.append(row_from)

    # The path is put together again from the end.
    end_page = max(range(page_count), key=lambda page: previous[page])
    path: list[int] = []
    page = end_page
    for row in range(len(similarities) - 1, -1, -1):
        path.append(page)
        page = came_from[row][page]
        if page < 0:
            break
    path.reverse()

    for position, frame_index in enumerate(usable):
        if position >= len(path):
            break
        page = path[position]
        if similarities[position][page] >= threshold:
            matches[frame_index] = page
    return matches


def slide_change_times(
    readings: list[dict[str, Any]],
    pages: list[dict[str, Any]]
) -> list[float]:
    """The moments where the slide on the screen changed.

    The pages of the notes are lined up with the pictures, and the page
    only goes forward. The moment of the change is not the moment the
    lining up noticed it, it is the moment the new page started to fit
    better than the old one.
    """
    page_texts = [page["text"].lower() for page in pages]
    if not page_texts:
        return []

    matches = match_pages_monotonic(readings, pages)
    times: list[float] = []
    previous: Optional[int] = None
    for index, match in enumerate(matches):
        if match is None or match == previous:
            continue
        if previous is not None:
            times.append(_moment_the_page_took_over(
                readings, page_texts, previous, match, index
            ))
        previous = match
    return times


def _moment_the_page_took_over(
    readings: list[dict[str, Any]],
    page_texts: list[str],
    old_page: int,
    new_page: int,
    from_index: int
) -> float:
    """The first picture where the new page fits better than the old one."""
    from rapidfuzz import fuzz

    for index in range(from_index, len(readings)):
        text = readings[index].get("text", "").lower()
        if not text.strip():
            continue
        new_score = fuzz.token_set_ratio(text, page_texts[new_page])
        old_score = fuzz.token_set_ratio(text, page_texts[old_page])
        if new_score >= old_score:
            return float(readings[index]["t"])
    return float(readings[from_index]["t"])


def change_times(
    readings: list[dict[str, Any]],
    matches: list[Optional[int]]
) -> list[float]:
    """The moments where the screen really changed.

    Only a change from one slide to another counts. A moment where the
    screen showed something that is not a slide, such as a terminal, is
    not a change of the slide, and the screen was not a lecture there
    either.
    """
    changes: list[float] = []
    previous: Optional[int] = None
    for reading, match in zip(readings, matches):
        if match is None:
            continue
        if previous is not None and match != previous:
            changes.append(float(reading["t"]))
        previous = match
    return changes


def score_trigger(
    fired: list[float],
    changes: list[float],
    tolerance: float = 4.0
) -> dict[str, float]:
    """How well the fired moments line up with the real changes."""
    if not changes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fires": len(fired), "changes": 0}
    matched_changes = set()
    true_positives = 0
    for moment in fired:
        for index, change in enumerate(changes):
            if index in matched_changes:
                continue
            if abs(moment - change) <= tolerance:
                matched_changes.add(index)
                true_positives += 1
                break
    precision = true_positives / len(fired) if fired else 0.0
    recall = true_positives / len(changes) if changes else 0.0
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fires": len(fired),
        "changes": len(changes)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="the videos of the manifest")
    parser.add_argument("--video", help="the id of a video")
    parser.add_argument("--trigger", action="store_true", help="run the trigger of the text")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=5.0)
    parser.add_argument("--captures", help="a json file with the captures of a system")
    parser.add_argument("--tolerance", type=float, default=4.0)
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list or not args.video:
        print(f"{'id':36} {'archetype':16} {'height':>7}  ground truth")
        for entry in manifest["videos"]:
            truth = entry["ground_truth"]["kind"]
            print(f"{entry['id']:36} {entry['archetype']:16} {entry['height']:>7}  {truth}")
        return 0

    entry = video_entry(manifest, args.video)
    readings = load_readings(manifest, args.video)["readings"]
    data_root = Path(manifest["data_root"])

    if args.trigger:
        pages = load_page_texts(entry["ground_truth"].get("sources", []), data_root)
        if not pages:
            print("there are no notes for this video, so the changes cannot be found")
            return 1
        matches = match_pages(readings, pages)
        changes = change_times(readings, matches)
        fired = fire_times(readings, args.threshold, args.min_gap)
        result = score_trigger(fired, changes, args.tolerance)
        print(f"{args.video}: threshold {args.threshold}, gap {args.min_gap}s")
        print(f"  fires   : {result['fires']}")
        print(f"  changes : {result['changes']}")
        print(f"  precision {result['precision']}  recall {result['recall']}  f1 {result['f1']}")
        return 0

    if args.captures:
        with open(args.captures, "r", encoding="utf-8") as handle:
            captures = json.load(handle)
        pages = load_page_texts(entry["ground_truth"].get("sources", []), data_root)
        matches = match_pages(readings, pages)
        by_time = {round(float(r["t"]), 1): index for index, r in enumerate(readings)}
        total = [0.0, 0.0, 0.0]
        counted = 0
        for capture in captures:
            moment = round(float(capture["t"]), 1)
            index = by_time.get(moment)
            if index is None or matches[index] is None:
                continue
            truth = pages[matches[index]]["text"]
            scores = word_scores(capture.get("text", ""), truth)
            total = [total[0] + scores[0], total[1] + scores[1], total[2] + scores[2]]
            counted += 1
        if counted == 0:
            print("no capture lined up with a page of the notes")
            return 1
        print(f"{args.video}: {counted} captures that line up with a page")
        print(f"  precision {total[0] / counted:.4f}  recall {total[1] / counted:.4f}  f1 {total[2] / counted:.4f}")
        return 0

    parser.error("give --trigger or --captures")
    return 2


if __name__ == "__main__":
    sys.exit(main())
