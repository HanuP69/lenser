"""The run of a lecture that is already a file.

Video mode plays a recording instead of the capture devices, and the
rest of the machine runs exactly as it does live: the audio blocks go
through the speech segmenter, the segments go through the transcription
agent, the frames go through the change gate and the screen agent, and
the finished timeline is structured into notes. No Redis or Postgres is
needed, because one process is the only consumer: an in-memory bus
receives the events the agents publish, and the run walks them once the
decode is over. The agents keep their publish code unchanged, which is
the point of the stub.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lecturelens.agents.audio_agent import (
    AudioAgent,
    AudioChunk,
    SpeechSegmenter,
    convert_numpy_to_wav_bytes
)
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.agents.sync_agent import SyncAgent
from lecturelens.agents.structuring_agent import StructuringAgent
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.video_decoder import (
    encode_png_bytes,
    iter_audio_blocks,
    iter_sampled_frames,
    probe_media
)
from lecturelens.schemas.events import SlideCaptured, TranscriptChunk

logger = logging.getLogger(__name__)


class InMemoryEventBus:
    """The bus stub that publishes into the run itself.

    A live session publishes to Redis so agents in other processes read
    the streams. The offline run is one process, so a list is enough and
    the events stay reachable after the agents are done with them. Only
    publish_resilient is needed by the reused agent code.
    """

    def __init__(self) -> None:
        self.published: List[Tuple[str, Any]] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> Optional[str]:
        self.published.append((stream_name, event))
        return f"memory-{len(self.published)}"

    def events_of_type(self, event_type: type) -> List[Any]:
        """The published events of one class, in publish order."""
        return [event for _, event in self.published if isinstance(event, event_type)]

    async def close(self) -> None:
        pass


@dataclass
class OfflineRunResult:
    session_dir: Path
    duration_seconds: Optional[float]
    transcript_events: int
    slide_events: int
    timeline_entries: int
    note_segments: int


async def run_offline_video(
    video_path: Path,
    settings: Settings,
    session_id: str,
    fps: Optional[float] = None
) -> OfflineRunResult:
    """Turn one lecture file into a session with notes.

    The decode runs as fast as the machine can, so a lecture shorter
    than an hour of file time usually finishes writing the notes well
    before an hour of wall-clock time.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"the lecture file does not exist: {video_path}")
    effective_fps = fps if fps is not None else settings.video_frame_fps

    session_dir = settings.create_session_directories(session_id)
    media = probe_media(video_path)

    # The clock is anchored once, at the start of the run. Video time is
    # the elapsed lecture time: zero at the first frame of the file. The
    # structuring agent loads the same anchor from clock.json, so the
    # headings in the notes carry the video minutes.
    base_epoch = time.time()
    clock = SessionClock.from_dict({
        "reference_wall_time": base_epoch,
        "reference_monotonic_time": time.monotonic(),
        "elapsed_offset_seconds": 0.0
    })
    clock.save_to_file(session_dir / "clock.json")

    # The usage account writes into the session's logs, so the usage
    # report of an offline session reads the same way a live one does.
    from lecturelens.router import usage as router_usage
    router_usage.set_sink(session_dir / "logs" / "llm_usage.jsonl")

    bus = InMemoryEventBus()
    transcript_events = await _run_audio_leg(
        video_path, settings, session_id, bus, clock, base_epoch
    )
    slide_events = await _run_screen_leg(
        video_path, settings, session_id, bus, clock, base_epoch, effective_fps
    )

    timeline_entries = await _write_timeline(
        settings, session_id, bus, clock, transcript_events, slide_events
    )

    note_segments = 0
    if timeline_entries > 0:
        note_segments = await _run_structuring(settings, session_id)

    await bus.close()
    return OfflineRunResult(
        session_dir=session_dir,
        duration_seconds=media.duration_seconds,
        transcript_events=len(transcript_events),
        slide_events=len(slide_events),
        timeline_entries=timeline_entries,
        note_segments=note_segments
    )


async def _run_audio_leg(
    video_path: Path,
    settings: Settings,
    session_id: str,
    bus: InMemoryEventBus,
    clock: SessionClock,
    base_epoch: float
) -> List[TranscriptChunk]:
    """Transcribe the audio track through the same segments the mic feeds.

    The transcription agent is reused whole: the wav bytes of every
    segment go through its ingest, which keeps the chunk saving, the
    hallucination filter and the language lock identical to the live
    path.
    """
    audio_agent = AudioAgent(
        settings=settings,
        redis_bus=bus,
        session_clock=clock,
        session_id=session_id
    )
    # The whisper model is loaded once before the file is decoded, so
    # the first segment does not pay for the load in its latency. A
    # failure here is not fatal: the model is then loaded lazily in the
    # first transcription instead.
    await audio_agent.preload_model()

    segmenter = SpeechSegmenter(
        sample_rate=settings.mic_sample_rate,
        energy_threshold=settings.vad_energy_threshold,
        min_chunk_seconds=settings.mic_chunk_duration_seconds,
        max_chunk_seconds=settings.audio_max_chunk_seconds,
        silence_gap_seconds=settings.audio_silence_gap_seconds
    )

    transcript_events: List[TranscriptChunk] = []

    async def transcribe_chunk(chunk: AudioChunk) -> None:
        wav_bytes = convert_numpy_to_wav_bytes(chunk.samples, settings.mic_sample_rate)
        event = await audio_agent.ingest_wav_bytes(
            wav_bytes,
            # The capture epoch is the end of the chunk inside the file.
            # The clock maps it back onto the video time, so the event
            # starts and ends where the speech sits in the recording.
            capture_epoch=base_epoch + chunk.end_seconds
        )
        if event is not None:
            transcript_events.append(event)

    stream_elapsed = 0.0
    for block in iter_audio_blocks(
        video_path,
        target_sample_rate=settings.mic_sample_rate,
        block_seconds=settings.audio_block_seconds
    ):
        for chunk in segmenter.push(block, stream_elapsed):
            await transcribe_chunk(chunk)
        stream_elapsed += settings.audio_block_seconds
    for chunk in segmenter.flush():
        await transcribe_chunk(chunk)
    return transcript_events


async def _run_screen_leg(
    video_path: Path,
    settings: Settings,
    session_id: str,
    bus: InMemoryEventBus,
    clock: SessionClock,
    base_epoch: float,
    fps: float
) -> List[SlideCaptured]:
    """Sample the video track and let the change gate judge the frames.

    The screen agent is reused whole: the frames go through the ingest
    with the gate on, so a recording plays its slides exactly like a
    screen plays them, including the revisits and the background text
    extraction that finishes after the slides are captured.
    """
    screen_router = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        # The same task the screen agent's process builds its router for.
        # Without Redis the rate limiter of the vision provider keeps the
        # turns inside this process.
        screen_router = build_router_for_task(
            "screen_ocr",
            None,
            load_router_config(settings.router_config_path),
            settings.ollama_url,
            settings.llm_mode,
            settings.llm_timeout_seconds,
            settings.local_model
        )
    except Exception as error:
        logger.warning("vision router unavailable, slides may be captured without text: %s", error)

    screen_agent = ScreenAgent(
        settings=settings,
        redis_bus=bus,
        session_clock=clock,
        session_id=session_id,
        llm_router=screen_router
    )

    for frame_elapsed, rgb_frame in iter_sampled_frames(video_path, fps=fps):
        png_bytes = await run_in_thread_pool(encode_png_bytes, rgb_frame)
        await screen_agent.ingest_png_bytes(
            png_bytes,
            capture_epoch=base_epoch + frame_elapsed,
            enforce_gate=True,
            # The gate gets the video time like every other timestamp of
            # the run. The capture epoch stays a wall-clock stamp for the
            # events, so the gate is never fed the wall clock.
            gate_now=frame_elapsed
        )

    if screen_agent.retry_tasks:
        # A slide that waits for its text must finish before the run
        # closes, or the notes would be written from a timeline that is
        # still missing the text a second later.
        await asyncio.gather(*list(screen_agent.retry_tasks), return_exceptions=True)

    return bus.events_of_type(SlideCaptured)


async def _write_timeline(
    settings: Settings,
    session_id: str,
    bus: InMemoryEventBus,
    clock: SessionClock,
    transcript_events: List[TranscriptChunk],
    slide_events: List[SlideCaptured]
) -> int:
    """Buffer every event and flush one final timeline.

    The flush sorts by event time and assigns the correlations, so the
    timeline.jsonl of an offline run holds the same shape a live run
    would write, ordered by where things sit in the lecture instead of
    the order the machine finished them in.
    """
    sync_agent = SyncAgent(
        settings=settings,
        redis_bus=bus,
        session_clock=clock,
        session_id=session_id
    )
    timeline_entries = 0
    for event in transcript_events:
        await sync_agent.add_entry(sync_agent.transcript_to_entry(event))
        timeline_entries += 1
    for event in slide_events:
        await sync_agent.add_entry(sync_agent.slide_to_entry(event))
        timeline_entries += 1
    # One final forced flush: there is no realtime merge window to wait
    # for when the file is already over.
    await sync_agent.flush_ready_entries(force=True)
    return timeline_entries


async def _run_structuring(settings: Settings, session_id: str) -> int:
    """Run the structuring agent over the finished timeline, in-process.

    The same pattern the reprocess command uses: the agent reads the
    timeline from the session folder, gets the notes router, and writes
    the notes. No bus is passed, because no other process is listening.
    """
    from lecturelens.router.llm_router import build_router_for_task, load_router_config

    structuring_router = None
    try:
        structuring_router = build_router_for_task(
            "structuring",
            None,
            load_router_config(settings.router_config_path),
            settings.ollama_url,
            settings.llm_mode,
            settings.llm_timeout_seconds,
            settings.local_model
        )
    except Exception as error:
        logger.warning("structuring router unavailable, the notes stay empty: %s", error)
    structuring_agent = StructuringAgent(
        settings=settings,
        session_id=session_id,
        llm_router=structuring_router,
        redis_bus=None
    )
    return await structuring_agent.process_all_pending()
