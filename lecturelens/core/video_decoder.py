"""Decoding a recorded lecture file into the streams the agents consume.

The live agents are fed by the capture devices: the microphone pushes
0.1 s blocks and the screen pushes frames once per capture interval. A
recording already holds that information, so video mode plays the file
instead of the devices: the audio track becomes mono blocks of the
capture rate, and video frames become the same PNG bytes the screen
capture would produce. The decode primitives live in one module, so a
bug in them is fixed once for every caller, and no agent has to know
PyAV exists.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import av
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# The live capture runs the microphone at this rate, and every agent
# downstream is tuned to it, so the audio track is resampled to the
# same numbers instead of inventing a new configuration for video mode.
TARGET_AUDIO_SAMPLE_RATE: int = 16000


@dataclass
class MediaInfo:
    duration_seconds: Optional[float]
    audio_streams: int
    video_streams: int


def probe_media(path: Path) -> MediaInfo:
    """The shape of the file, for logging and for a friendly CLI summary.

    A missing or unreadable file raises from av.open, which is fine: the
    command cannot do anything useful without the file anyway.
    """
    with av.open(str(path)) as container:
        duration: Optional[float] = None
        if container.duration is not None:
            # The container duration counts microseconds.
            duration = float(container.duration) / 1_000_000.0
        return MediaInfo(
            duration_seconds=duration,
            audio_streams=len(container.streams.audio),
            video_streams=len(container.streams.video)
        )


def iter_audio_blocks(
    path: Path,
    target_sample_rate: int = TARGET_AUDIO_SAMPLE_RATE,
    block_seconds: float = 0.1
) -> Iterator[np.ndarray]:
    """Play the audio track as float32 mono blocks the size the mic feeds.

    A container is opened for this leg alone: demuxing walks the packets
    in file order, so a consumer that already drained the video leg of
    another container must start from a fresh open, or it sits beyond
    the end of the stream and receives nothing.
    """
    block_samples = int(round(target_sample_rate * block_seconds))
    with av.open(str(path)) as container:
        audio_streams = container.streams.audio
        if len(audio_streams) == 0:
            logger.info("no audio stream in %s, the run carries the slides alone", path)
            return
        # s16 keeps the quantization an integer wav file would have, so
        # the transcription agents downstream see the same numbers the
        # live path produces from the same recording.
        resampler = av.AudioResampler(format="s16", rate=target_sample_rate, layout="mono")
        buffer = np.empty(0, dtype=np.float32)
        for packet in container.demux(audio_streams[0]):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError as error:
                # One damaged packet costs a whisper of audio, not the
                # lecture; the run keeps going like a dropped block does.
                logger.warning("unreadable audio packet in %s: %s", path, error)
                continue
            for frame in frames:
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                    buffer = np.concatenate((buffer, samples))
                    while buffer.size >= block_samples:
                        yield buffer[:block_samples]
                        buffer = buffer[block_samples:]
        # The resampler holds a tail of filt samples between conversions;
        # the padded leftovers at the end of the track leave the buffer.
        for converted in resampler.resample(None):
            samples = converted.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
            buffer = np.concatenate((buffer, samples))
            while buffer.size >= block_samples:
                yield buffer[:block_samples]
                buffer = buffer[block_samples:]
        if buffer.size > 0:
            # A short tail is the same shape a device that stopped
            # mid-block would deliver, and the segmenter closes on it.
            yield buffer


def iter_sampled_frames(
    path: Path,
    fps: float = 1.0
) -> Iterator[Tuple[float, np.ndarray]]:
    """Play the video track at fps as (seconds inside the file, rgb array).

    The seconds are video time: a lecture that spans an hour of file
    time spans an hour of elapsed lecture time, whatever wall-clock
    hours passed around the recording. Frames between the sampling
    points are skipped, because the change gate downstream only needs
    the slide moments, not every motion of the file.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    minimum_gap = 1.0 / fps
    last_yield: Optional[float] = None
    with av.open(str(path)) as container:
        video_streams = container.streams.video
        if len(video_streams) == 0:
            logger.info("no video stream in %s, the run carries the speech alone", path)
            return
        warned_missing_time = False
        for frame in container.decode(video=0):
            seconds = frame.time
            if seconds is None:
                # A record without timestamps cannot be placed on the
                # lecture clock; it is skipped rather than guessed.
                if not warned_missing_time:
                    logger.warning("frame without a presentation timestamp in %s", path)
                    warned_missing_time = True
                continue
            if last_yield is not None and (seconds - last_yield) < minimum_gap:
                continue
            yield seconds, frame.to_ndarray(format="rgb24")
            last_yield = seconds


def encode_png_bytes(image: np.ndarray) -> bytes:
    """The agents exchange frames as PNG bytes, so a decoded frame does too."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
