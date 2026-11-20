"""The tests of the offline video mode.

The synthetic clip is written with PyAV itself, so the tests run
against the same shape of file the user brings: an mp4 with an mpeg4
video stream and an aac audio stream. The whisper model is swapped for
a canned answer, so the transcription path is exercised without the
model load, and the routers are stubs, so no model call of the machine
is needed.
"""

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import av
import numpy as np
from PIL import Image

from lecturelens.agents import audio_agent as audio_agent_module
from lecturelens.agents.structuring_agent import STRUCTURED_DELIMITER
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.offline_pipeline import InMemoryEventBus, run_offline_video
from lecturelens.core.video_decoder import (
    encode_png_bytes,
    iter_audio_blocks,
    iter_sampled_frames,
    probe_media
)
from lecturelens.router import llm_router as llm_router_module

SAMPLE_RATE = 16000
CLIP_SECONDS = 10.0
CLIP_FPS = 2.0


def write_synthetic_clip(path: Path) -> None:
    """A ten second clip: three solid plates on the video, two speech runs.

    The plates change at 0 s, 4 s and 8 s, and the speech runs sit at
    0-4 s and 4.6-8.6 s, so the segmenter closes one chunk per run with
    the default silence gap of half a second.
    """
    with av.open(str(path), "w") as container:
        video_stream = container.add_stream("mpeg4", rate=int(CLIP_FPS))
        video_stream.width = 64
        video_stream.height = 48
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")

        plate_colors = [(240, 240, 240), (0, 0, 200), (0, 150, 0)]
        total_frames = int(CLIP_SECONDS * CLIP_FPS)
        for index in range(total_frames):
            elapsed = index / CLIP_FPS
            color = plate_colors[min(int(elapsed // 4.0), 2)]
            image = np.empty((48, 64, 3), dtype=np.uint8)
            image[:, :] = color
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)

        time_axis = np.arange(int(SAMPLE_RATE * CLIP_SECONDS), dtype=np.float32) / SAMPLE_RATE
        speaking = ((time_axis < 4.0) | (time_axis >= 4.6) & (time_axis < 8.6))
        samples = np.where(speaking, 0.3 * np.sin(2.0 * np.pi * 440.0 * time_axis), 0.0)
        pcm = (samples.astype(np.float32) * 32767.0).astype(np.int16)
        for start in range(0, len(pcm), 1600):
            piece = pcm[start:start + 1600]
            frame = av.AudioFrame.from_ndarray(piece.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = start
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


class StubRouter:
    """The contract of the structuring router: one call per batch."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        text = str(messages[0].content)
        self.prompts.append(text)
        if "CURRENT LECTURE EVENTS" in text:
            notes = "## Binary Search\nBinary search splits the range in half every step."
            structured = {
                "definitions": [],
                "action_items": [],
                "key_points": ["Binary search runs in O(log n)."]
            }
            return f"{notes}\n\n{STRUCTURED_DELIMITER}\n\n{json.dumps(structured)}"
        raise AssertionError(f"an unexpected call reached the router: {text[:90]!r}")


async def fake_transcription_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
    """The whisper call returns canned speech; the preload funds nothing."""
    if getattr(function, "__name__", "") == "preload_whisper_model":
        return None
    return {
        "text": "binary search splits the range in half",
        "confidence": 0.9,
        "language": "en",
        "language_probability": 0.95
    }


def make_offline_settings(temp_dir: str) -> Settings:
    settings = Settings()
    settings.output_dir = Path(temp_dir)
    settings.video_frame_fps = CLIP_FPS
    # The slide leg must not schedule the vision retries in the tests:
    # no router and no local readers, so the slides flow straight onto
    # the timeline.
    settings.screen_local_ocr = False
    settings.screen_layout_model = False
    # The drift filter would ask the embeddings server of the machine.
    settings.enable_drift_filter = False
    # Slides change four seconds apart in the clip; the default interval
    # of five seconds would swallow the second and third plate.
    settings.slide_min_interval_seconds = 1.0
    return settings


class OfflineTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clip = Path(self.temp_dir.name) / "lecture.mp4"
        write_synthetic_clip(self.clip)
        self.settings = make_offline_settings(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class TestDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.clip = Path(cls.temp_dir.name) / "lecture.mp4"
        write_synthetic_clip(cls.clip)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_audio_blocks_are_mono_float_at_target_rate(self) -> None:
        blocks = list(iter_audio_blocks(self.clip))
        self.assertGreaterEqual(len(blocks), int(CLIP_SECONDS / 0.1) - 2)
        for block in blocks:
            self.assertEqual(block.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(block)))
            self.assertLessEqual(float(np.max(np.abs(block))), 1.0)
        total = sum(float(block.size) for block in blocks)
        # The aac record carries a little encoder padding, so the total
        # is allowed a fifth of a second of drift either way.
        self.assertAlmostEqual(total / SAMPLE_RATE, CLIP_SECONDS, delta=0.3)

    def test_speech_blocks_and_silence_blocks_alternate(self) -> None:
        blocks = list(iter_audio_blocks(self.clip))
        energies = [float(np.sqrt(np.mean(np.square(block)))) for block in blocks]
        # The tone runs to 4.0 s, so the block that starts at 4.0 s sits
        # in the silence gap between the two speech runs.
        self.assertGreater(energies[0], 0.1)
        first_silence_index = int(4.0 / 0.1)
        self.assertLess(energies[first_silence_index], 0.01)
        self.assertGreater(energies[first_silence_index - 1], 0.1)

    def test_sampled_frames_walk_video_time(self) -> None:
        points = list(iter_sampled_frames(self.clip, fps=1.0))
        times = [seconds for seconds, _ in points]
        self.assertEqual(times[0], 0.0)
        self.assertTrue(all(later >= earlier for earlier, later in zip(times, times[1:])))
        self.assertGreaterEqual(times[-1], CLIP_SECONDS - 1.5)
        for earlier, later in zip(times, times[1:]):
            self.assertGreaterEqual(later - earlier, 1.0 - 0.01)
        # Every frame is a decoded rgb image of the plate.
        for _, image in points:
            self.assertEqual(image.shape[2], 3)

    def test_sampled_frames_honour_a_higher_fps(self) -> None:
        points = list(iter_sampled_frames(self.clip, fps=2.0))
        self.assertGreaterEqual(len(points), int(CLIP_SECONDS * 2.0) - 2)

    def test_sampled_frames_reject_a_bad_fps(self) -> None:
        with self.assertRaises(ValueError):
            list(iter_sampled_frames(self.clip, fps=0.0))

    def test_png_encode_round_trip(self) -> None:
        image = np.full((48, 64, 3), 120, dtype=np.uint8)
        png_bytes = encode_png_bytes(image)
        decoded = Image.open(__import__("io").BytesIO(png_bytes))
        self.assertEqual(decoded.size, (64, 48))

    def test_probe_reports_the_streams(self) -> None:
        info = probe_media(self.clip)
        self.assertEqual(info.audio_streams, 1)
        self.assertEqual(info.video_streams, 1)
        self.assertIsNotNone(info.duration_seconds)
        self.assertAlmostEqual(info.duration_seconds, CLIP_SECONDS, delta=0.7)


class TestOfflineRun(OfflineTestBase):
    async def _run(self, session_id: str) -> Any:
        with mock.patch.object(
            audio_agent_module, "run_in_process_pool", side_effect=fake_transcription_pool
        ), mock.patch.object(
            llm_router_module,
            "build_router_for_task",
            side_effect=lambda task, *args, **kwargs: None if task == "screen_ocr" else StubRouter()
        ), mock.patch.object(
            llm_router_module, "load_router_config", return_value={}
        ):
            return await asyncio.wait_for(
                run_offline_video(self.clip, self.settings, session_id), timeout=180
            )

    async def test_offline_run_writes_a_full_session(self) -> None:
        session_id = "offline_full"
        result = await self._run(session_id)

        # Two speech runs close two chunks, and the three plates become
        # three slides at 0 s, 4 s and 8 s of video time.
        self.assertEqual(result.transcript_events, 2)
        self.assertEqual(result.slide_events, 3)
        self.assertEqual(result.timeline_entries, 5)
        self.assertGreaterEqual(result.note_segments, 1)
        self.assertAlmostEqual(result.duration_seconds, CLIP_SECONDS, delta=0.7)

        # The run anchors the video time on clock.json, and zero offset,
        # so the structuring agent reads the same lecture seconds.
        session_dir = result.session_dir
        clock = SessionClock.load_from_file(session_dir / "clock.json")
        self.assertEqual(clock.elapsed_offset_seconds, 0.0)

        # The timeline is ordered by event time, not by the order the
        # legs finished in: every transcript entry carries the seconds
        # of its chunk inside the file.
        timeline_lines = (session_dir / "timeline.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        entries = [json.loads(line) for line in timeline_lines if line.strip()]
        self.assertEqual(len(entries), 5)
        elapsed_values = [float(entry["elapsed_seconds"]) for entry in entries]
        self.assertEqual(
            elapsed_values, sorted(elapsed_values), "the timeline is not in event time"
        )
        transcripts = [entry for entry in entries if entry["source_type"] == "transcript"]
        self.assertEqual(len(transcripts), 2)
        for entry in transcripts:
            self.assertLessEqual(float(entry["elapsed_seconds"]), CLIP_SECONDS)
        slides = [entry for entry in entries if entry["source_type"] == "slide"]
        self.assertEqual(len(slides), 3)
        # A slide sits where the plate changed: ~0 s, ~4 s and ~8 s.
        self.assertLess(float(slides[0]["elapsed_seconds"]), 1.5)
        self.assertAlmostEqual(float(slides[1]["elapsed_seconds"]), 4.0, delta=1.5)
        self.assertAlmostEqual(float(slides[2]["elapsed_seconds"]), 8.0, delta=2.0)

        # The notes carry the video time in the headings and the words
        # of the transcript in the body.
        notes_text = (session_dir / "notes.md").read_text(encoding="utf-8")
        self.assertIn("Binary search", notes_text)
        self.assertIsNotNone(re.search(r"### 00:0\d:\d\d", notes_text))
        self.assertTrue((session_dir / "notes.html").exists())

    async def test_offline_run_is_repeatable_on_a_fresh_session(self) -> None:
        """A second session from the same file starts its counters at zero."""
        first = await self._run("offline_first")
        second = await self._run("offline_second")
        self.assertEqual(second.transcript_events, first.transcript_events)
        self.assertEqual(second.slide_events, first.slide_events)


class TestScreenGateOnVideoTime(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_png_bytes_gate_runs_on_video_time(self) -> None:
        """The gate takes the video time, not the wall clock of the run.

        Two frames a fraction of the gap apart inside the file stay
        rejected, even though the wall clock ran much longer between the
        two ingests once the decode glides faster than realtime.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = make_offline_settings(temp_dir)
            bus = InMemoryEventBus()
            agent = ScreenAgent(
                settings=settings,
                redis_bus=bus,
                session_clock=SessionClock(),
                session_id="offline_gate",
                llm_router=None
            )

            def png_of(color: tuple[int, int, int]) -> bytes:
                image = np.empty((48, 64, 3), dtype=np.uint8)
                image[:, :] = color
                return encode_png_bytes(image)

            # The gate accepts the first frame, then the second plate
            # settles the way the live polls do: the poll after the one
            # that first saw the change is stable against the previous.
            # Everything is measured on the video time, so an accept
            # never happens inside the minimum interval of the file.
            first = await agent.ingest_png_bytes(
                png_of((240, 240, 240)), capture_epoch=1000.0, enforce_gate=True, gate_now=0.0
            )
            same_again = await agent.ingest_png_bytes(
                png_of((240, 240, 240)), capture_epoch=1001.0, enforce_gate=True, gate_now=1.0
            )
            first_blue_poll = await agent.ingest_png_bytes(
                png_of((0, 0, 200)), capture_epoch=1002.0, enforce_gate=True, gate_now=2.0
            )
            blue_accepted = await agent.ingest_png_bytes(
                png_of((0, 0, 200)), capture_epoch=1003.0, enforce_gate=True, gate_now=3.0
            )
            # The green plate polls too soon, as the file's clock says, so
            # only the interval elapsed in video time accepts it.
            early_green = await agent.ingest_png_bytes(
                png_of((0, 150, 0)), capture_epoch=1003.9, enforce_gate=True, gate_now=3.9
            )
            green_accepted = await agent.ingest_png_bytes(
                png_of((0, 150, 0)), capture_epoch=1004.3, enforce_gate=True, gate_now=4.3
            )
            self.assertIsNotNone(first)
            self.assertIsNone(same_again)
            self.assertIsNone(first_blue_poll)
            self.assertIsNotNone(blue_accepted)
            self.assertIsNone(early_green)
            self.assertIsNotNone(green_accepted)
            self.assertNotEqual(first.slide_number, blue_accepted.slide_number)
            self.assertNotEqual(blue_accepted.slide_number, green_accepted.slide_number)
