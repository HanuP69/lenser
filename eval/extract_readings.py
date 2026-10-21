"""Read the text of the screen at every step of a video.

This is the honest reader. It is the fastest way to find out what was on
the screen, without a model in the way, and the text it produces is what
the capture is measured against.

The readings also carry the signal that says whether there is something
new on the screen, which is what the trigger of the capture should use
instead of the difference of the pixels.

    python eval/extract_readings.py --video B_statquest_gradient_descent
    python eval/extract_readings.py --all --step 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval" / "manifest.json"

_ENGINE = None


def get_engine() -> Any:
    """The text reader. One for the whole process.

    The runtime uses every core of the machine by itself. When many of
    them run side by side they fight over the cores and it gets slower,
    not faster, so each one is given a single thread.
    """
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR

        _ENGINE = RapidOCR(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    return _ENGINE


def read_frame(image: Any) -> dict[str, Any]:
    """The text of one picture."""
    engine = get_engine()
    output = engine(image)
    texts = list(getattr(output, "txts", None) or [])
    scores = list(getattr(output, "scores", None) or [])
    boxes = getattr(output, "boxes", None)
    return {
        "text": " ".join(texts),
        "lines": len(texts),
        "score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "boxes": [[[round(float(x), 1), round(float(y), 1)] for x, y in box] for box in boxes]
        if boxes is not None
        else [],
    }


def frame_at(video: Path, seconds: float) -> Optional[Any]:
    """One picture, without decoding everything in front of it.

    Decoding the whole video and throwing most of the pictures away took
    a quarter of an hour for a handful of pictures. Seeking takes a
    hundredth of a second.
    """
    import av

    with av.open(str(video)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        duration = float(container.duration) / 1e6 if container.duration else 0.0
        if seconds >= duration - 0.5:
            return None
        container.seek(int(seconds / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            return frame.to_ndarray(format="bgr24")
    return None


def read_one(job: tuple[str, float]) -> tuple[float, dict[str, Any]]:
    video_path, seconds = job
    image = frame_at(Path(video_path), seconds)
    if image is None:
        return seconds, {}
    return seconds, read_frame(image)


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        return json.load(handle)


def video_entry(manifest: dict[str, Any], video_id: str) -> dict[str, Any]:
    for entry in manifest["videos"]:
        if entry["id"] == video_id:
            return entry
    raise SystemExit(f"the video {video_id} is not in the manifest")


def extract(
    manifest: dict[str, Any],
    video_id: str,
    step_seconds: float,
    workers: int,
    out_dir: Path
) -> Path:
    entry = video_entry(manifest, video_id)
    root = Path(manifest["data_root"])
    video_path = root / entry["file"]
    if not video_path.exists():
        raise SystemExit(f"the video is not there: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{video_id}.json"
    if target.exists():
        print(f"  {video_id}: the readings are already there")
        return target

    import av

    with av.open(str(video_path)) as container:
        duration = float(container.duration) / 1e6 if container.duration else 0.0
    limit = entry.get("use_seconds")
    if limit:
        duration = min(duration, float(limit))

    seconds = [index * step_seconds for index in range(int(duration / step_seconds) + 1)]
    print(f"  {video_id}: {duration:.0f}s, {len(seconds)} pictures, {workers} workers")

    started = time.monotonic()
    readings: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, (moment, result) in enumerate(pool.map(read_one, [(str(video_path), s) for s in seconds], chunksize=4), start=1):
            if not result:
                continue
            readings.append({"t": round(moment, 3), **result})
            if number % 50 == 0:
                rate = number / max(0.001, time.monotonic() - started)
                left = (len(seconds) - number) / max(0.001, rate)
                print(f"    {number}/{len(seconds)}  {rate:.1f} pictures a second  {left / 60:.1f} min left")

    payload = {
        "video_id": video_id,
        "archetype": entry["archetype"],
        "step_seconds": step_seconds,
        "duration_seconds": round(duration, 3),
        "readings": readings,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"  {video_id}: {len(readings)} readings in {(time.monotonic() - started) / 60:.1f} min -> {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="the id of one video of the manifest")
    parser.add_argument("--all", action="store_true", help="every video of the manifest")
    parser.add_argument("--step", type=float, default=2.0, help="seconds between the pictures")
    parser.add_argument("--workers", type=int, default=8, help="how many pictures at the same time")
    parser.add_argument("--out", default=str(REPO / "eval" / "readings"), help="where the readings go")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("give one --video or --all")

    manifest = load_manifest()
    out_dir = Path(args.out)
    videos = [entry["id"] for entry in manifest["videos"]] if args.all else [args.video]
    for video_id in videos:
        extract(manifest, video_id, args.step, args.workers, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
