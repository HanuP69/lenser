import asyncio
import io
import tempfile
import time
import unittest
import wave
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np

from lecturelens.agents import audio_agent
from lecturelens.agents.audio_agent import (
    AudioAgent,
    PendingTranscription,
    SpeechSegmenter,
    convert_numpy_to_wav_bytes,
    is_hallucinated_text
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import set_flag
from lecturelens.core.clock import SessionClock

SAMPLE_RATE = 16000
BLOCK = SAMPLE_RATE // 10


def speech_block(level: float = 0.2) -> np.ndarray:
    return np.full(BLOCK, level, dtype=np.float32)


def silence_block() -> np.ndarray:
    return np.zeros(BLOCK, dtype=np.float32)


def make_segmenter(**overrides: Any) -> SpeechSegmenter:
    parameters = dict(
        sample_rate=SAMPLE_RATE,
        energy_threshold=0.01,
        min_chunk_seconds=1.0,
        max_chunk_seconds=3.0,
        silence_gap_seconds=0.3
    )
    parameters.update(overrides)
    return SpeechSegmenter(**parameters)


class TestSpeechSegmenter(unittest.TestCase):
    def test_pure_silence_emits_nothing(self) -> None:
        segmenter = make_segmenter()
        emitted = []
        for index in range(100):
            emitted.extend(segmenter.push(silence_block(), index * 0.1 + 0.1))
        emitted.extend(segmenter.flush())
        self.assertEqual(emitted, [])

    def test_chunk_is_cut_at_pause_not_mid_speech(self) -> None:
        segmenter = make_segmenter()
        emitted = []
        elapsed = 0.0
        for _ in range(15):
            elapsed += 0.1
            emitted.extend(segmenter.push(speech_block(), elapsed))
        self.assertEqual(emitted, [])
        for _ in range(4):
            elapsed += 0.1
            emitted.extend(segmenter.push(silence_block(), elapsed))
        self.assertEqual(len(emitted), 1)
        chunk = emitted[0]
        self.assertGreaterEqual(chunk.end_seconds - chunk.start_seconds, 1.5)

    def test_max_length_forces_cut(self) -> None:
        segmenter = make_segmenter()
        emitted = []
        elapsed = 0.0
        for _ in range(35):
            elapsed += 0.1
            emitted.extend(segmenter.push(speech_block(), elapsed))
        self.assertGreaterEqual(len(emitted), 1)
        duration = emitted[0].end_seconds - emitted[0].start_seconds
        self.assertLessEqual(duration, 3.2)

    def test_no_samples_are_lost_between_chunks(self) -> None:
        segmenter = make_segmenter()
        total_pushed = 0
        emitted_samples = 0
        elapsed = 0.0
        for index in range(120):
            elapsed += 0.1
            block = speech_block() if (index % 20) < 14 else silence_block()
            total_pushed += block.size
            for chunk in segmenter.push(block, elapsed):
                emitted_samples += chunk.samples.size
        for chunk in segmenter.flush():
            emitted_samples += chunk.samples.size
        self.assertGreater(emitted_samples, 0)
        self.assertLessEqual(emitted_samples, total_pushed)
        speech_samples = 6 * 14 * BLOCK
        self.assertGreaterEqual(emitted_samples, speech_samples)

    def test_timestamps_follow_stream_origin(self) -> None:
        segmenter = make_segmenter()
        emitted = []
        elapsed = 50.0
        for _ in range(15):
            elapsed += 0.1
            emitted.extend(segmenter.push(speech_block(), elapsed))
        for _ in range(4):
            elapsed += 0.1
            emitted.extend(segmenter.push(silence_block(), elapsed))
        self.assertAlmostEqual(emitted[0].start_seconds, 50.0, delta=0.3)

    def test_flush_emits_trailing_speech(self) -> None:
        segmenter = make_segmenter()
        for index in range(5):
            segmenter.push(speech_block(), (index + 1) * 0.1)
        self.assertEqual(len(segmenter.flush()), 1)

    def test_discontinuity_resets_origin(self) -> None:
        segmenter = make_segmenter()
        for index in range(5):
            segmenter.push(speech_block(), (index + 1) * 0.1)
        flushed = segmenter.mark_discontinuity()
        self.assertEqual(len(flushed), 1)
        self.assertIsNone(segmenter.origin_seconds)


class TestAudioHelpers(unittest.TestCase):
    def test_hallucination_filter(self) -> None:
        self.assertTrue(is_hallucinated_text("Thank you."))
        self.assertTrue(is_hallucinated_text("  you  "))
        self.assertTrue(is_hallucinated_text(""))
        self.assertTrue(is_hallucinated_text("go go go go go go go go go"))
        self.assertFalse(is_hallucinated_text("Thank you all for coming to the lecture today."))
        self.assertFalse(is_hallucinated_text("Binary search runs in log n time."))

    def test_hallucination_filter_uses_the_language_of_the_audio(self) -> None:
        """The phrase list was English only, so a Hindi or Hinglish lecture
        was never filtered."""
        self.assertTrue(is_hallucinated_text("धन्यवाद", language="hi"))
        self.assertTrue(is_hallucinated_text("अगले वीडियो", language="hi"))
        self.assertTrue(is_hallucinated_text("dhanyavaad", language="hinglish"))
        self.assertTrue(is_hallucinated_text("agle video", language="hinglish"))
        # The sign off of another language is not a hallucination.
        self.assertFalse(is_hallucinated_text("धन्यवाद", language="en"))
        # Real content is never filtered.
        self.assertFalse(is_hallucinated_text("बाइनरी सर्च लॉग एन टाइम में चलता है", language="hi"))

    def test_hallucination_filter_accepts_extra_phrases(self) -> None:
        self.assertFalse(is_hallucinated_text("pause and resume"))
        self.assertTrue(
            is_hallucinated_text("pause and resume", extra_phrases=["pause and resume"])
        )

    def test_wav_conversion_round_trip(self) -> None:
        samples = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        wav_bytes = convert_numpy_to_wav_bytes(samples, SAMPLE_RATE)
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            self.assertEqual(reader.getframerate(), SAMPLE_RATE)
            self.assertEqual(reader.getnframes(), 1600)


class FakeBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append((stream_name, event))
        return "1-1"


class ScriptedSource:
    def __init__(self, blocks: list[Optional[np.ndarray]]) -> None:
        self.blocks = list(blocks)
        self.started = False
        self.stopped = False
        self.restarts = 0

    def start(self) -> None:
        self.started = True

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        if len(self.blocks) == 0:
            time.sleep(0.01)
            return None
        return self.blocks.pop(0)

    def stop(self) -> None:
        self.stopped = True

    def restart(self) -> None:
        self.restarts += 1


def make_agent(temp_dir: str, source: Any, bus: Any) -> AudioAgent:
    settings = Settings()
    settings.output_dir = Path(temp_dir)
    settings.mic_sample_rate = SAMPLE_RATE
    settings.mic_chunk_duration_seconds = 1.0
    settings.audio_max_chunk_seconds = 3.0
    settings.audio_silence_gap_seconds = 0.3
    settings.audio_queue_capacity = 2
    settings.audio_stall_restart_seconds = 0.5
    return AudioAgent(settings, bus, SessionClock(), "audio_test", block_source=source)


async def fake_process_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
    if function is audio_agent.preload_whisper_model:
        return True
    await asyncio.sleep(0.3)
    return {"text": "binary search is logarithmic", "confidence": 0.9, "language": "en", "language_probability": 0.95}


class TestAudioAgent(unittest.IsolatedAsyncioTestCase):
    async def test_capture_is_not_blocked_by_slow_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks: list[Optional[np.ndarray]] = []
            for _ in range(6):
                for _ in range(15):
                    blocks.append(speech_block())
                for _ in range(4):
                    blocks.append(silence_block())
            bus = FakeBus()
            source = ScriptedSource(blocks)
            agent = make_agent(temp_dir, source, bus)

            async def stopper() -> None:
                while len(source.blocks) > 0:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.2)
                set_flag(agent.control.session_dir, "stop.flag")

            with mock.patch.object(audio_agent, "run_in_process_pool", side_effect=fake_process_pool):
                await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=30)

            wav_files = sorted((Path(temp_dir) / "audio_test" / "raw" / "audio").glob("chunk_*.wav"))
            self.assertEqual(len(wav_files), 6)
            self.assertGreaterEqual(len(bus.published), 1)
            self.assertTrue(source.started)
            self.assertTrue(source.stopped)
            first_event = bus.published[0][1]
            self.assertEqual(first_event.text, "binary search is logarithmic")
            self.assertIsNotNone(first_event.raw_audio_path)

    async def test_chunk_numbering_continues_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = make_agent(temp_dir, ScriptedSource([]), FakeBus())
            agent.chunk_counter = 5
            audio_dir = agent.raw_audio_dir
            (audio_dir / "chunk_00005.wav").write_bytes(b"x")
            restarted = make_agent(temp_dir, ScriptedSource([]), FakeBus())
            self.assertEqual(restarted.chunk_counter, 5)

    async def test_language_locks_after_confident_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bus = FakeBus()
            agent = make_agent(temp_dir, ScriptedSource([]), bus)
            agent.settings.whisper_lock_language = True
            pending = PendingTranscription(b"", "p.wav", 0.0, 1.0)
            captured_languages: list[Optional[str]] = []

            async def fake_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
                captured_languages.append(args[3])
                return {"text": "hola mundo esto es", "confidence": 0.9, "language": "es", "language_probability": 0.95}

            with mock.patch.object(audio_agent, "run_in_process_pool", side_effect=fake_pool):
                await agent.transcribe_and_publish(pending)
                await agent.transcribe_and_publish(pending)
            self.assertEqual(captured_languages, [None, "es"])

    async def test_hallucinated_chunks_are_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bus = FakeBus()
            agent = make_agent(temp_dir, ScriptedSource([]), bus)

            async def fake_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
                return {"text": "Thank you.", "confidence": 0.9, "language": "en", "language_probability": 0.9}

            with mock.patch.object(audio_agent, "run_in_process_pool", side_effect=fake_pool):
                result = await agent.transcribe_and_publish(PendingTranscription(b"", "p.wav", 0.0, 1.0))
            self.assertIsNone(result)
            self.assertEqual(bus.published, [])

    async def test_transcription_failure_is_logged_and_loop_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bus = FakeBus()
            agent = make_agent(temp_dir, ScriptedSource([]), bus)
            agent.transcription_queue = asyncio.Queue()
            outcomes = [RuntimeError("model missing"), {"text": "real words here", "confidence": 0.8, "language": "en", "language_probability": 0.9}]

            async def fake_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            with mock.patch.object(audio_agent, "run_in_process_pool", side_effect=fake_pool):
                await agent.transcription_queue.put(PendingTranscription(b"", "a.wav", 0.0, 1.0))
                await agent.transcription_queue.put(PendingTranscription(b"", "b.wav", 1.0, 2.0))
                await agent.transcription_queue.put(None)
                await asyncio.wait_for(agent.transcription_loop(), timeout=5)
            self.assertEqual(len(bus.published), 1)
            self.assertEqual(agent.control.stats["transcription_errors"], 1)

    async def test_backlog_overflow_keeps_raw_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = make_agent(temp_dir, ScriptedSource([]), FakeBus())
            from lecturelens.agents.audio_agent import AudioChunk
            for _ in range(5):
                await agent.handle_chunk(AudioChunk(speech_block(), 0.0, 0.1))
            wav_files = list(agent.raw_audio_dir.glob("chunk_*.wav"))
            self.assertEqual(len(wav_files), 5)
            self.assertEqual(agent.transcription_queue.qsize(), 2)
            self.assertEqual(len(agent.spillover_queue), 3)
            self.assertEqual(agent.dropped_transcriptions, 0)

    async def test_stalled_source_is_restarted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = ScriptedSource([])
            agent = make_agent(temp_dir, source, FakeBus())

            async def stopper() -> None:
                await asyncio.sleep(1.5)
                set_flag(agent.control.session_dir, "stop.flag")

            with mock.patch.object(audio_agent, "run_in_process_pool", side_effect=fake_process_pool):
                await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=15)
            self.assertGreaterEqual(source.restarts, 1)


class TestMixedAndSystemBlockSource(unittest.TestCase):
    def test_mixed_block_source_both_present(self) -> None:
        from lecturelens.agents.audio_agent import MixedBlockSource
        mixed_source = MixedBlockSource(16000, 0.1)
        b1 = np.full(1600, 0.3, dtype=np.float32)
        b2 = np.full(1600, 0.2, dtype=np.float32)
        mixed_source.mic_source.block_queue.put(b1)
        mixed_source.system_source.block_queue.put(b2)
        out = mixed_source.read_block(0.1)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(float(out[0]), 0.5, places=4)

    def test_mixed_block_source_fallback_when_mic_none(self) -> None:
        from lecturelens.agents.audio_agent import MixedBlockSource
        mixed_source = MixedBlockSource(16000, 0.1)
        b2 = np.full(1600, 0.4, dtype=np.float32)
        mixed_source.system_source.block_queue.put(b2)
        out = mixed_source.read_block(0.01)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(float(out[0]), 0.4, places=4)

    def test_mixed_block_source_fallback_when_sys_none(self) -> None:
        from lecturelens.agents.audio_agent import MixedBlockSource
        mixed_source = MixedBlockSource(16000, 0.1)
        b1 = np.full(1600, 0.35, dtype=np.float32)
        mixed_source.mic_source.block_queue.put(b1)
        out = mixed_source.read_block(0.1)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(float(out[0]), 0.35, places=4)

    def test_system_block_source_strict_mode_raises_when_no_device(self) -> None:
        from lecturelens.agents.audio_agent import SystemBlockSource
        with mock.patch("lecturelens.agents.audio_agent.find_loopback_microphone", return_value=(None, "No loopback")):
            sys_source = SystemBlockSource(16000, 0.1, strict=True)
            with self.assertRaises(RuntimeError):
                sys_source.start()


if __name__ == "__main__":
    unittest.main()
