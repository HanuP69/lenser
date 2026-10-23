"""The board crops that the reading of a blackboard is measured on.

A clip of a teacher writing on a board is cut into a handful of frames,
spread evenly over the video. Beside every frame there is an empty row
in `truth.jsonl`, and the row is filled by hand with what is on the
board at that moment. The filled file is the ground truth of the
board-reading (Tier 0 to Tier 2) and of the capture of board states
(Phase 3).

    python eval/board_pack.py input_video/chem.mp4 --count 20 --out eval/board_pack/chem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="the clip of the board")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", default="eval/board_pack/board")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    parser.add_argument("--seconds", type=float, default=0.0, help="cut only this window (0 = whole video)")
    args = parser.parse_args()

    source = Path(args.video)
    if not source.exists():
        print(f"not found: {source}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"the video could not be opened: {source}", file=sys.stderr)
        return 1
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / frames_per_second
    if args.seconds > 0:
        duration = min(duration, args.seconds)

    # The frames are spread over the window, but the first and the last
    # percent are left out, so the player chrome and the fade to the
    # next clip are not in the pack.
    margin = duration * 0.02
    window = max(0.0, duration - 2 * margin)
    slots = [args.start + margin + window * (index + 0.5) / args.count for index in range(args.count)]

    rows: list[dict[str, Any]] = []
    for index, moment in enumerate(slots, start=1):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, moment) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            print(f"the frame at {moment:.1f}s could not be read", file=sys.stderr)
            continue
        height, width = frame.shape[:2]
        if width > 1920:
            # The size of the pack has to stay small enough to look at
            # quickly; a board is read no better at 4k than at 1080p.
            frame = cv2.resize(frame, (1920, int(height * 1920 / width)))
        name = f"board_{index:03d}_t{moment:07.1f}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append({"t": round(moment, 1), "image": name, "text": "", "notes": ""})
    capture.release()

    truth_path = out_dir / "truth.jsonl"
    if truth_path.exists():
        # A pack that was partly labelled keeps its labels. The frames
        # keep their names, so only the rows of the new frames are added.
        existing = truth_path.read_text(encoding="utf-8").splitlines()
        names = {json.loads(line)["image"] for line in existing if line.strip()}
        rows = [row for row in rows if row["image"] not in names] + [
            json.loads(line) for line in existing if line.strip()
        ]
        rows.sort(key=lambda row: float(row["t"]))
    with open(truth_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unfilled = sum(1 for row in rows if not row["text"].strip())
    print(f"frames: {len(rows)}   empty rows to fill by hand: {unfilled}")
    print(f"pack: {out_dir}   truth: {truth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
