"""The five minutes of speech that the transcription is measured on.

A window of a recorded clip is pulled out as a wav file, cut into rows
of half a minute in `audio_truth.jsonl`, and every row is filled by
hand with what was said in it. The filled file is the ground truth of
the audio work (Phase 5), so the models are compared against what a
person hears and not against what another model hears.

    python eval/audio_truth_pack.py input_video/chem.mp4 --start 60 --seconds 300
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import av
import numpy as np


def resampled_mono(container: av.container.Container, index: int, start: float, seconds: float, rate: int = 16000) -> np.ndarray:
    """The audio of one stream, one channel, at the rate of the ASR."""
    stream = container.streams.audio[index]
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="s16", layout="mono", rate=rate)
    pieces: list[np.ndarray] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            moment = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if moment + float(frame.samples / stream.rate) < start:
                continue
            if moment > start + seconds:
                break
            for resampled in resampler.resample(frame):
                pieces.append(resampled.to_ndarray())
    if not pieces:
        return np.zeros(0, dtype=np.int16)
    # The resampler hands over stereo-shaped arrays sometimes; a mono
    # stream of s16 comes back as one channel either way.
    joined = np.concatenate([piece.reshape(-1)[: piece.size] for piece in pieces])
    joined = joined.astype(np.int16).reshape(-1)
    offset = 0
    return joined[offset:offset + int(seconds * rate)] if len(joined) > int(seconds * rate) else joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="the clip the speech is taken from")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--out", default="eval/audio_pack")
    parser.add_argument("--name", default="", help="the name of the pack (default: the name of the video)")
    parser.add_argument("--row", type=float, default=30.0, help="how long a row of the truth is")
    args = parser.parse_args()

    source = Path(args.video)
    if not source.exists():
        print(f"not found: {source}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or source.stem
    wav_path = out_dir / f"{name}.wav"

    container = av.open(str(source))
    if not container.streams.audio:
        print(f"the video holds no audio: {source}", file=sys.stderr)
        return 1
    audio = resampled_mono(container, 0, args.start, args.seconds)
    container.close()
    if len(audio) == 0:
        print("no audio came out of the window", file=sys.stderr)
        return 1

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(audio.tobytes())

    rows: list[dict[str, Any]] = []
    moment = args.start
    while moment < args.start + audio.shape[0] / 16000.0:
        end = min(moment + args.row, args.start + audio.shape[0] / 16000.0)
        rows.append({"start": round(moment, 1), "end": round(end, 1), "text": ""})
        moment = end
    truth_path = out_dir / f"{name}_truth.jsonl"
    with open(truth_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"audio: {wav_path}  ({len(audio) / 16000.0:.1f}s, 16kHz mono)")
    print(f"truth rows: {len(rows)}  ->  {truth_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
