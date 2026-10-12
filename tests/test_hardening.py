import asyncio
import io
import json
import re
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from lecturelens.agents import screen_agent
from lecturelens.agents.audio_agent import SpeechSegmenter
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.agents.structuring_agent import StructuringAgent
from lecturelens.agents.sync_agent import SyncAgent, entry_fingerprint
from lecturelens.cli.main import session_has_live_agents
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.markdown_formatter import format_clean_markdown, normalize_headings
from lecturelens.core.redis_bus import RedisBus, handler_accepts_message_id
from lecturelens.exports.anki_exporter import format_card_field_html
from lecturelens.exports.pdf_exporter import create_pdf_from_markdown, latex_to_markup
from lecturelens.knowledge.concept_graph import ConceptGraph, normalize_concept_name
from lecturelens.schemas.events import TimelineEntry, TranscriptChunk
from tests.test_structuring_agent import MockRouter, timeline_line


def redis_is_available() -> bool:
    import socket
    try:
        with socket.create_connection(("localhost", 6379), timeout=0.3):
            return True
    except OSError:
        return False


REDIS_AVAILABLE = redis_is_available()


class TestClockInverse(unittest.TestCase):
    def test_elapsed_for_epoch_round_trips(self) -> None:
        clock = SessionClock(elapsed_offset_seconds=100.0)
        epoch = clock.epoch_for_elapsed(130.0)
        self.assertAlmostEqual(clock.elapsed_for_epoch(epoch), 130.0, places=6)

    def test_elapsed_for_epoch_never_goes_below_offset(self) -> None:
        clock = SessionClock(elapsed_offset_seconds=50.0)
        self.assertEqual(clock.elapsed_for_epoch(clock.reference_wall_time - 1000.0), 50.0)


@unittest.skipUnless(REDIS_AVAILABLE, "redis server not available")
class TestRedisBusBacklog(unittest.IsolatedAsyncioTestCase):
    async def test_unacked_backlog_is_replayed_once_without_spinning(self) -> None:
        session = "spin_" + uuid.uuid4().hex[:6]
        stream = f"lecturelens:{session}:stream:transcript"
        bus = RedisBus("redis://localhost:6379")
        try:
            for index in range(3):
                await bus.publish(stream, TranscriptChunk(
                    session_id=session, timestamp_epoch=time.time(), start_seconds=0.0,
                    end_seconds=1.0, text=f"m{index}", confidence=0.9
                ))
            seen: list[str] = []

            async def handler(data: dict, message_id: str) -> None:
                seen.append(message_id)

            first = asyncio.create_task(bus.consume(
                stream, "g", "c1", handler, block_ms=100, should_stop=lambda: True, auto_ack=False
            ))
            await asyncio.sleep(0.8)
            first.cancel()
            self.assertEqual(len(seen), 3)
            seen.clear()
            second = asyncio.create_task(bus.consume(
                stream, "g", "c1", handler, block_ms=100, should_stop=lambda: True, auto_ack=False
            ))
            await asyncio.sleep(1.2)
            second.cancel()
            self.assertEqual(len(seen), 3, "backlog must be delivered exactly once per restart")
        finally:
            await bus.client.delete(stream)
            await bus.close()

    def test_handler_arity_detection(self) -> None:
        async def one(data: dict) -> None:
            return None

        async def two(data: dict, message_id: str) -> None:
            return None

        self.assertFalse(handler_accepts_message_id(one))
        self.assertTrue(handler_accepts_message_id(two))


class SyncHardeningBase(unittest.IsolatedAsyncioTestCase):
    session = "hardening_sync"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.settings.sync_merge_window_seconds = 0.2
        self.settings.correlation_window_seconds = 5.0
        self.bus = RedisBus("redis://localhost:6379")
        self.agent = SyncAgent(self.settings, self.bus, SessionClock(), self.session)

    async def asyncTearDown(self) -> None:
        await self.bus.close()
        self.temp_dir.cleanup()

    def make_entry(self, timestamp: float, source_type: str, content: str = "text") -> TimelineEntry:
        return TimelineEntry(
            session_id=self.session, timestamp_epoch=timestamp, source_type=source_type,
            content=content, correlation_id=None, importance_flags=[], metadata={}
        )


class TestSyncCorrelationActiveSlide(SyncHardeningBase):
    def test_speech_long_after_slide_change_still_belongs_to_visible_slide(self) -> None:
        slide = self.make_entry(100.0, "slide")
        speech = self.make_entry(160.0, "transcript")
        self.agent.assign_correlation_ids([slide, speech])
        self.assertEqual(speech.correlation_id, slide.correlation_id)

    def test_speech_just_before_slide_change_belongs_to_new_slide(self) -> None:
        old_slide = self.make_entry(0.0, "slide")
        speech = self.make_entry(98.0, "transcript")
        new_slide = self.make_entry(100.0, "slide")
        self.agent.assign_correlation_ids([old_slide, speech, new_slide])
        self.assertEqual(speech.correlation_id, new_slide.correlation_id)
        self.assertNotEqual(old_slide.correlation_id, new_slide.correlation_id)

    def test_speech_beyond_active_age_is_not_linked(self) -> None:
        slide = self.make_entry(0.0, "slide")
        speech = self.make_entry(5000.0, "transcript")
        self.agent.assign_correlation_ids([slide, speech])
        self.assertIsNone(speech.correlation_id)

    def test_active_correlation_can_be_disabled(self) -> None:
        self.settings.correlation_active_seconds = 0.0
        slide = self.make_entry(100.0, "slide")
        speech = self.make_entry(160.0, "transcript")
        self.agent.assign_correlation_ids([slide, speech])
        self.assertIsNone(speech.correlation_id)

    def test_anchor_history_is_kept_for_the_active_window(self) -> None:
        slide = self.make_entry(0.0, "slide")
        self.agent.assign_correlation_ids([slide])
        later_speech = self.make_entry(400.0, "transcript")
        self.agent.assign_correlation_ids([later_speech])
        self.assertEqual(later_speech.correlation_id, slide.correlation_id)


@unittest.skipUnless(REDIS_AVAILABLE, "redis server not available")
class TestSyncDuplicateProtection(SyncHardeningBase):
    async def test_entry_already_in_timeline_is_not_written_twice_after_restart(self) -> None:
        first = self.make_entry(10.0, "transcript", "unique line")
        await self.agent.add_entry(first)
        await self.agent.flush_ready_entries(force=True)
        restarted = SyncAgent(self.settings, self.bus, SessionClock(), self.session)
        await restarted.add_entry(self.make_entry(10.0, "transcript", "unique line"))
        await restarted.flush_ready_entries(force=True)
        lines = [line for line in restarted.timeline_path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(restarted.duplicates_skipped, 1)

    async def test_same_message_id_is_not_buffered_twice(self) -> None:
        await self.agent.add_entry(self.make_entry(1.0, "transcript", "a"), "s", "g", "1-0")
        await self.agent.add_entry(self.make_entry(1.0, "transcript", "a"), "s", "g", "1-0")
        self.assertEqual(len(self.agent.pending_buffer), 1)

    async def test_identical_message_ids_on_different_streams_are_both_kept(self) -> None:
        await self.agent.add_entry(self.make_entry(1.0, "transcript", "a"), "stream:transcript", "g", "1-0")
        await self.agent.add_entry(self.make_entry(2.0, "slide", "b"), "stream:slide", "g", "1-0")
        self.assertEqual(len(self.agent.pending_buffer), 2)

    def test_reclaim_idle_exceeds_the_hold_window(self) -> None:
        self.settings.sync_merge_window_seconds = 30.0
        self.assertGreater(self.agent.reclaim_idle_milliseconds(), 30000)

    def test_fingerprint_changes_with_content_and_time(self) -> None:
        base = self.make_entry(1.0, "slide", "x")
        self.assertNotEqual(entry_fingerprint(base), entry_fingerprint(self.make_entry(1.0, "slide", "y")))
        self.assertNotEqual(entry_fingerprint(base), entry_fingerprint(self.make_entry(2.0, "slide", "x")))
        self.assertEqual(entry_fingerprint(base), entry_fingerprint(self.make_entry(1.0, "slide", "x")))


class StructuringHardeningBase(unittest.IsolatedAsyncioTestCase):
    session_id = "hardening_struct"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.session_dir = self.settings.create_session_directories(self.session_id)
        self.timeline_path = self.session_dir / "timeline.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_agent(self) -> StructuringAgent:
        return StructuringAgent(self.settings, self.session_id, MockRouter())

    def write_entries(self, count: int, same_second: bool = False) -> None:
        lines = []
        for index in range(count):
            stamp = 1_700_000_000.0 + (index * 0.1 if same_second else index * 100.0)
            lines.append(timeline_line(stamp, "transcript", f"entry number {index} with some words"))
        self.timeline_path.write_text("".join(lines), encoding="utf-8")

    def section_count(self) -> int:
        notes = (self.session_dir / "notes.md").read_text(encoding="utf-8")
        return len(re.findall(r"^### \d\d:\d\d:\d\d", notes, re.MULTILINE))

    async def crash_on_save(self, crash_index: int) -> None:
        agent = self.make_agent()
        original_save = agent.save_state
        counter = {"calls": 0}

        def crashing_save(*args: Any, **kwargs: Any) -> Any:
            counter["calls"] += 1
            if counter["calls"] == crash_index:
                raise RuntimeError("simulated crash")
            return original_save(*args, **kwargs)

        agent.save_state = crashing_save
        with self.assertRaises(RuntimeError):
            await agent.process_all_pending()


class TestNotesIdempotency(StructuringHardeningBase):
    async def test_crash_on_first_batch_does_not_duplicate_notes(self) -> None:
        self.settings.max_slice_characters = 12000
        self.write_entries(2)
        await self.crash_on_save(1)
        await self.make_agent().process_all_pending()
        self.assertEqual(self.section_count(), 1)

    async def test_crash_on_later_batches_does_not_duplicate_notes(self) -> None:
        self.settings.max_slice_characters = 80
        self.write_entries(4)
        await self.crash_on_save(2)
        await self.make_agent().process_all_pending()
        self.assertEqual(self.section_count(), 4)

    async def test_batches_sharing_the_same_wall_second_are_all_written(self) -> None:
        self.settings.max_slice_characters = 80
        self.write_entries(3, same_second=True)
        await self.make_agent().process_all_pending()
        self.assertEqual(self.section_count(), 3)

    async def test_user_edits_survive_restart(self) -> None:
        self.settings.max_slice_characters = 12000
        self.write_entries(1)
        await self.make_agent().process_all_pending()
        with open(self.session_dir / "notes.md", "a", encoding="utf-8") as handle:
            handle.write("\nMY OWN ADDITION\n")
        self.make_agent()
        self.assertIn("MY OWN ADDITION", (self.session_dir / "notes.md").read_text(encoding="utf-8"))

    async def test_marker_is_ignored_when_notes_were_deleted(self) -> None:
        self.settings.max_slice_characters = 12000
        self.write_entries(1)
        await self.make_agent().process_all_pending()
        (self.session_dir / "notes.md").unlink()
        (self.session_dir / "context.json").unlink()
        await self.make_agent().process_all_pending()
        self.assertTrue((self.session_dir / "notes.md").exists())

    def test_reset_outputs_removes_marker(self) -> None:
        agent = self.make_agent()
        agent.save_notes_marker(123)
        agent.reset_outputs()
        self.assertEqual(agent.load_notes_marker(), -1)


class TestJsonRecovery(unittest.TestCase):
    def test_list_is_recovered_from_messy_model_output(self) -> None:
        from lecturelens.agents.structuring_agent import extract_json_payload
        card = '[{"question":"q","answer":"a"}]'
        cases = [
            "Sure! " + card + " bye",
            "See [1] below.\n" + card,
            card + "\n(note: see [2])",
            "```json\n" + card + "\n```",
            '{"flashcards":' + card + "}",
        ]
        for text in cases:
            result = extract_json_payload(text)
            self.assertIsInstance(result, list, text)
            self.assertEqual(result[0]["question"], "q", text)

    def test_truncated_array_salvages_complete_items(self) -> None:
        from lecturelens.agents.structuring_agent import extract_json_payload
        result = extract_json_payload('[{"question":"q","answer":"a"},{"question":"q2","ans')
        self.assertEqual(len(result), 1)

    def test_garbage_returns_none(self) -> None:
        from lecturelens.agents.structuring_agent import extract_json_payload
        self.assertIsNone(extract_json_payload("no json here"))


class TestAdaptiveVoiceDetection(unittest.TestCase):
    sample_rate = 16000
    block = 1600

    def noise(self, rng: np.random.Generator, rms: float) -> np.ndarray:
        return (rng.standard_normal(self.block) * rms).astype(np.float32)

    def speech(self, rms: float) -> np.ndarray:
        t = np.arange(self.block) / self.sample_rate
        return (np.sin(2 * np.pi * 220 * t) * rms * np.sqrt(2)).astype(np.float32)

    def run_scenario(self, background: float, speech_rms: float, talk_seconds: int = 60) -> tuple[int, SpeechSegmenter]:
        rng = np.random.default_rng(1)
        segmenter = SpeechSegmenter(self.sample_rate, 0.01, 0.5, 15.0, 0.8)
        clock = 0.0
        for _ in range(300):
            segmenter.push(self.noise(rng, background), clock)
            clock += 0.1
        chunks = 0
        for index in range(talk_seconds * 10):
            talking = (index % 40) < 25
            block = self.speech(speech_rms) + self.noise(rng, background) if talking else self.noise(rng, background)
            chunks += len(segmenter.push(block, clock))
            clock += 0.1
        chunks += len(segmenter.flush())
        return chunks, segmenter

    def test_normal_room_detects_speech(self) -> None:
        chunks, _ = self.run_scenario(0.003, 0.03)
        self.assertGreater(chunks, 5)

    def test_soft_speaker_over_slightly_noisy_room_is_not_swallowed(self) -> None:
        chunks, segmenter = self.run_scenario(0.009, 0.015)
        self.assertGreater(chunks, 5)
        self.assertEqual(segmenter.effective_threshold(), 0.01)

    def test_threshold_never_ratchets_up_in_quiet_rooms(self) -> None:
        _, segmenter = self.run_scenario(0.005, 0.012)
        self.assertEqual(segmenter.effective_threshold(), 0.01)

    def test_loud_room_noise_alone_produces_no_chunks(self) -> None:
        chunks, _ = self.run_scenario(0.02, 0.0)
        self.assertEqual(chunks, 0)

    def test_speech_over_loud_room_is_still_detected(self) -> None:
        chunks, segmenter = self.run_scenario(0.02, 0.08)
        self.assertGreater(chunks, 5)
        self.assertGreater(segmenter.effective_threshold(), 0.01)

    def test_threshold_is_capped(self) -> None:
        _, segmenter = self.run_scenario(0.2, 0.0, talk_seconds=30)
        self.assertLessEqual(segmenter.effective_threshold(), 0.04 + 1e-9)


def make_png(size: tuple[int, int] = (640, 360)) -> bytes:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).rectangle([50, 50, 300, 120], fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class RecordingBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append(event)
        return "1-1"


class FlakyVisionRouter:
    def __init__(self, fail_first: int) -> None:
        self.calls = 0
        self.fail_first = fail_first

    async def ainvoke(self, messages: Any) -> Any:
        from langchain_core.messages import AIMessage
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("rate limited")
        return AIMessage(content="Recovered slide text")


class TestScreenAgentHardening(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, temp_dir: str, router: Any) -> tuple[ScreenAgent, RecordingBus]:
        settings = Settings()
        settings.output_dir = Path(temp_dir)
        settings.vision_retry_delays_seconds = [0.05, 0.05]
        bus = RecordingBus()
        return ScreenAgent(settings, bus, SessionClock(), "screen_hardening", router), bus

    async def empty_ocr(self, function: Any, png_bytes: bytes, tesseract_path: str) -> str:
        return ""

    async def test_slide_timestamp_is_capture_time_not_after_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = self.make_agent(temp_dir, FlakyVisionRouter(0))
            capture = agent.session_clock.get_timestamp() - 42.0
            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=self.empty_ocr):
                event = await agent.ingest_png_bytes(make_png(), capture_epoch=capture)
            self.assertEqual(event.timestamp_epoch, capture)

    async def test_vision_failure_is_retried_and_text_recovered_with_original_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            router = FlakyVisionRouter(fail_first=2)
            agent, bus = self.make_agent(temp_dir, router)
            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=self.empty_ocr):
                first = await agent.ingest_png_bytes(make_png())
            self.assertEqual(first.extracted_text, "")
            await asyncio.wait_for(asyncio.gather(*list(agent.retry_tasks)), timeout=5)
            self.assertEqual(len(bus.published), 2)
            recovered = bus.published[1]
            self.assertEqual(recovered.extracted_text, "Recovered slide text")
            self.assertEqual(recovered.slide_number, first.slide_number)
            self.assertEqual(recovered.timestamp_epoch, first.timestamp_epoch)

    async def test_no_retry_scheduled_without_a_vision_router(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = self.make_agent(temp_dir, None)
            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=self.empty_ocr):
                await agent.ingest_png_bytes(make_png())
            self.assertEqual(len(agent.retry_tasks), 0)

    async def test_stop_cancels_pending_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = self.make_agent(temp_dir, FlakyVisionRouter(fail_first=99))
            agent.settings.vision_retry_delays_seconds = [30.0]
            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=self.empty_ocr):
                await agent.ingest_png_bytes(make_png())
            tasks = list(agent.retry_tasks)
            self.assertEqual(len(tasks), 1)
            agent.stop()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.assertTrue(tasks[0].cancelled())


class TestBoardOcclusion(unittest.TestCase):
    def make_agent(self, temp_dir: str) -> Any:
        from tests.test_camera_agent import make_agent, FakeRouter
        agent, _ = make_agent(temp_dir, FakeRouter())
        return agent

    def test_large_sudden_change_is_held_until_it_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = self.make_agent(temp_dir)
            agent.settings.board_occlusion_patience = 3
            base = np.full((90, 160), 220, dtype=np.uint8)
            agent.last_published_thumbnail = base
            agent.last_publish_time = 0.0
            blocked = base.copy()
            blocked[:, 20:140] = 30
            self.assertFalse(agent.has_board_changed(blocked, 100.0))
            self.assertFalse(agent.has_board_changed(blocked, 101.0))
            self.assertTrue(agent.has_board_changed(blocked, 102.0))

    def test_small_writing_change_is_accepted_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = self.make_agent(temp_dir)
            base = np.full((90, 160), 220, dtype=np.uint8)
            agent.last_published_thumbnail = base
            agent.last_publish_time = 0.0
            written = base.copy()
            written[10:16, 10:60] = 20
            self.assertTrue(agent.has_board_changed(written, 100.0))

    def test_occlusion_counter_resets_when_the_board_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = self.make_agent(temp_dir)
            base = np.full((90, 160), 220, dtype=np.uint8)
            agent.last_published_thumbnail = base
            agent.last_publish_time = 0.0
            blocked = base.copy()
            blocked[:, 20:140] = 30
            agent.has_board_changed(blocked, 100.0)
            agent.has_board_changed(base.copy(), 101.0)
            self.assertEqual(agent.occlusion_hold_count, 0)


class TestFormatterHardening(unittest.TestCase):
    def test_code_blocks_keep_indentation_and_are_not_turned_into_headings(self) -> None:
        raw = "## A\n\n```python\ndef solve(n):\n    return Fibonacci\n    Done Early\n```\n"
        self.assertIn("    return Fibonacci\n    Done Early", format_clean_markdown(raw))

    def test_tables_stay_tables(self) -> None:
        cleaned = format_clean_markdown("| Name | Value |\n|---|---|\n| Alpha Beta | One |")
        self.assertNotIn("#", cleaned)

    def test_numbered_items_are_not_promoted_to_headings(self) -> None:
        cleaned = format_clean_markdown("1. First step here.\n4. Binary Search\n10. Merge Sort")
        self.assertNotIn("#", cleaned)

    def test_nested_list_indentation_survives(self) -> None:
        cleaned = format_clean_markdown("- parent item here\n    - child item here")
        self.assertIn("    - child item here", cleaned)

    def test_display_math_inside_code_fences_is_untouched(self) -> None:
        cleaned = format_clean_markdown("```bash\necho $$ and $$\n```")
        self.assertIn("echo $$ and $$", cleaned)

    def test_unicode_math_is_preserved(self) -> None:
        cleaned = format_clean_markdown("\U0001d465 \u2208 \u2308y\u2309 \u2300")
        for symbol in ("\U0001d465", "\u2208", "\u2308", "\u2300"):
            self.assertIn(symbol, cleaned)

    def test_title_case_line_inside_a_paragraph_is_not_a_heading(self) -> None:
        cleaned = normalize_headings("Some intro sentence here.\nGravity Constant\nThe value is 9.8.")
        self.assertNotIn("#", cleaned)

    def test_standalone_title_case_line_becomes_a_heading(self) -> None:
        cleaned = normalize_headings("Intro text here.\n\nGravity Constant\nThe value is 9.8.")
        self.assertIn("#### Gravity Constant", cleaned)

    def test_continuation_marker_is_removed_from_plain_lines_too(self) -> None:
        cleaned = normalize_headings("A sentence about Topic (Cont.) that keeps going on and on here.")
        self.assertNotIn("(Cont.)", cleaned)


class TestAnkiMathHardening(unittest.TestCase):
    def test_prices_are_left_alone(self) -> None:
        for text in ("costs $5 and later $10 more", "between $5-$10 range", "It cost $5 total.", "$10$"):
            self.assertNotIn("\\(", format_card_field_html(text), text)

    def test_short_symbolic_math_is_converted(self) -> None:
        for expression in ("$NP$", "$f(x, y)$", "$a, b$", "$P(A|B)$", "$2n$", "$x_i$", "$n^2$"):
            self.assertIn("\\(", format_card_field_html(expression), expression)

    def test_prose_between_dollars_is_not_math(self) -> None:
        self.assertNotIn("\\(", format_card_field_html("pay $x for the very long service and $y"))


class TestPdfHardening(unittest.TestCase):
    def test_latex_is_converted_to_readable_markup(self) -> None:
        markup = latex_to_markup(r"\frac{a}{b} + \sum_{i=0}^{n} x_i \leq \alpha^2")
        self.assertIn("(a)/(b)", markup)
        self.assertIn("<sub>i=0</sub>", markup)
        self.assertIn("<super>n</super>", markup)
        self.assertIn("&lt;=", markup)
        self.assertIn("alpha", markup)
        self.assertNotIn("\\", markup)

    def test_markup_escapes_angle_brackets(self) -> None:
        self.assertNotIn("<b>", latex_to_markup("a<b>c"))

    def test_pdf_builds_for_awkward_content(self) -> None:
        documents = {
            "ragged table": "| A | B | C |\n|---|---|---|\n| 1 | 2 |\n| 1 |",
            "long code": "```\nx = " + "1234567890" * 40 + "\n```",
            "display math": "$$\n\\frac{a}{b} + \\sqrt{x}\n$$",
            "wide table": "|" + "|".join(f"H{i}" for i in range(8)) + "|\n|" + "|".join("---" for _ in range(8)) + "|\n|" + "|".join("some words here" for _ in range(8)) + "|",
        }
        for label, markdown in documents.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "out.pdf"
                create_pdf_from_markdown(markdown, target, "T")
                self.assertGreater(target.stat().st_size, 500, label)


class TestConceptNormalization(unittest.TestCase):
    def test_variants_normalize_to_the_same_name(self) -> None:
        names = ["Depth-First Search (DFS)", "depth first searches", "The depth first search"]
        self.assertEqual(len({normalize_concept_name(name) for name in names}), 1)

    def test_cross_session_lookup_is_tolerant_to_rephrasing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = ConceptGraph(Path(temp_dir) / "c.db")
            graph.add_or_update_concept("Depth-First Search (DFS)", "s1", 1.0, "d", [])
            found = graph.find_concept_in_other_sessions("depth first searches", "s2")
            self.assertEqual([record["session_id"] for record in found], ["s1"])

    def test_different_concepts_are_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = ConceptGraph(Path(temp_dir) / "c.db")
            graph.add_or_update_concept("Depth-First Search", "s1", 1.0, "d", [])
            self.assertEqual(graph.find_concept_in_other_sessions("Breadth-First Search", "s2"), [])


class TestLiveSessionGuard(unittest.TestCase):
    def write_heartbeat(self, session_dir: Path, name: str, state: str, age: float) -> None:
        heartbeat_dir = session_dir / "heartbeats"
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        payload = {"agent": name, "pid": 1, "time": time.time() - age, "state": state, "stats": {}}
        (heartbeat_dir / f"{name}.json").write_text(json.dumps(payload))

    def test_fresh_live_heartbeat_blocks_reprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_heartbeat(Path(temp_dir), "audio_agent", "live", 1.0)
            self.assertTrue(session_has_live_agents(Path(temp_dir)))

    def test_stopped_or_stale_heartbeats_do_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_heartbeat(Path(temp_dir), "audio_agent", "stopped", 1.0)
            self.write_heartbeat(Path(temp_dir), "sync_agent", "live", 600.0)
            self.assertFalse(session_has_live_agents(Path(temp_dir)))

    def test_missing_heartbeats_do_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(session_has_live_agents(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
