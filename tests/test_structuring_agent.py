import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from lecturelens.agents.structuring_agent import (
    SliceBatch,
    StructuringAgent,
    extract_json_payload,
    split_entries_into_batches
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import FINALIZE_FLAG_NAME, set_flag


class MockRouter:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.fail = False
        self.definition_term = "Binary Search"

    async def ainvoke(self, messages: list) -> str:
        if self.fail:
            raise RuntimeError("llm unavailable")
        text = str(messages[0].content)
        self.prompts.append(text)
        if "Extract all technical terms" in text:
            return json.dumps([{"term": self.definition_term, "definition": "A search algorithm that runs in logarithmic time."}])
        if "action items" in text:
            return json.dumps(["Complete homework 3 by Friday."])
        if "core key points" in text:
            return json.dumps(["Binary search runs in O(log n).", "Requires sorted input."])
        if "Compress the following" in text:
            return "Binary search divides the search space in half."
        return "## Binary Search\nBinary search is an efficient algorithm on sorted arrays."

    def summary_prompts(self) -> list[str]:
        matching: list[str] = []
        for prompt in self.prompts:
            if "CURRENT LECTURE EVENTS" in prompt:
                matching.append(prompt)
        return matching


class MockRedisBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, stream_name: str, event: Any) -> str:
        self.published.append((stream_name, event))
        return "1-0"

    async def publish_resilient(self, stream_name: str, event: Any) -> Optional[str]:
        return await self.publish(stream_name, event)


def timeline_line(timestamp: float, source: str, content: str, flags: list[str] = None, correlation: str = None, metadata: dict = None) -> str:
    return json.dumps({
        "session_id": "s",
        "timestamp_epoch": timestamp,
        "source_type": source,
        "content": content,
        "correlation_id": correlation,
        "importance_flags": flags or [],
        "metadata": metadata or {}
    }) + "\n"


class StructuringTestBase(unittest.IsolatedAsyncioTestCase):
    session_id = "test_struct_session"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.session_dir = self.settings.create_session_directories(self.session_id)
        self.timeline_path = self.session_dir / "timeline.jsonl"
        self.router = MockRouter()
        self.bus = MockRedisBus()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_agent(self, router: Any = None, session_id: str = None) -> StructuringAgent:
        return StructuringAgent(
            settings=self.settings,
            session_id=session_id or self.session_id,
            llm_router=router or self.router,
            redis_bus=self.bus
        )

    def write_lines(self, lines: list[str]) -> None:
        with open(self.timeline_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(lines))

    def notes_text(self) -> str:
        path = self.session_dir / "notes.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""


class TestStructuringBasics(StructuringTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.write_lines([
            timeline_line(100.0, "transcript", "Today we cover binary search.", ["exam_relevant"], "c1"),
            timeline_line(101.0, "slide", "Binary Search: O(log n)", [], "c1", metadata={"slide_number": 1})
        ])

    async def test_process_batch_writes_notes_and_publishes_segment(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        content = self.notes_text()
        self.assertIn("Binary Search", content)
        self.assertIn("Important Exam Topic", content)
        self.assertIn("Complete homework 3", content)
        self.assertEqual(len(self.bus.published), 1)
        stream_name, segment = self.bus.published[0]
        self.assertIn("stream:notes", stream_name)
        self.assertEqual(segment.sequence_number, 1)
        self.assertIn("binary search", segment.concept_tags)
        self.assertEqual(segment.slide_references, [1])

    async def test_correlated_events_reach_the_llm_as_one_unit(self) -> None:
        await self.make_agent().process_batch()
        prompt = self.router.summary_prompts()[0]
        self.assertIn("[TRANSCRIPT][exam_relevant] Today we cover binary search. | [SLIDE] Binary Search: O(log n)", prompt)

    async def test_running_context_is_saved_and_fed_to_next_batch(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        state = agent.load_state()
        self.assertEqual(state["running_context"], "Binary search divides the search space in half.")
        self.write_lines([timeline_line(200.0, "transcript", "Next we discuss hashing.")])
        await agent.process_batch()
        self.assertIn("Binary search divides the search space in half.", self.router.summary_prompts()[1])

    async def test_offset_advances_and_second_pass_is_a_no_op(self) -> None:
        agent = self.make_agent()
        self.assertEqual(await agent.process_all_pending(), 1)
        self.assertEqual(agent.load_state()["timeline_byte_offset"], self.timeline_path.stat().st_size)
        self.assertEqual(await agent.process_all_pending(), 0)

    async def test_sequence_number_increments_across_batches(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        self.assertEqual(agent.sequence_number, 1)
        self.write_lines([timeline_line(200.0, "transcript", "Second part of lecture.")])
        await agent.process_batch()
        self.assertEqual(agent.sequence_number, 2)
        restarted = self.make_agent()
        self.assertEqual(restarted.sequence_number, 2)

    async def test_returned_running_context_respects_max_characters(self) -> None:
        self.settings.context_max_characters = 10
        agent = self.make_agent()
        batches = agent.read_pending_batches()
        returned_context = await agent.process_single_batch(batches[0], "initial context string")
        self.assertLessEqual(len(returned_context), 10)
        self.assertEqual(len(agent.load_state()["running_context"]), len(returned_context))


class TestStructuringRestartAndIdempotency(StructuringTestBase):
    async def test_restart_does_not_duplicate_notes(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Segment one.")])
        first = self.make_agent()
        await first.process_all_pending()
        notes_after_first = self.notes_text()

        restarted = self.make_agent()
        self.assertEqual(await restarted.process_all_pending(), 0)
        self.assertEqual(self.notes_text(), notes_after_first)

        self.write_lines([timeline_line(2.0, "transcript", "Segment two.")])
        await restarted.process_all_pending()
        self.assertEqual(self.notes_text().count("### "), 2)

    async def test_blank_and_corrupt_lines_do_not_break_offsets(self) -> None:
        good_one = timeline_line(1.0, "transcript", "Good one.")
        good_two = timeline_line(2.0, "transcript", "Good two.")
        self.write_lines([good_one, "\n", "{broken json\n", good_two])
        agent = self.make_agent()
        await agent.process_all_pending()
        self.assertEqual(agent.load_state()["timeline_byte_offset"], self.timeline_path.stat().st_size)
        prompt = self.router.summary_prompts()[0]
        self.assertIn("Good one.", prompt)
        self.assertIn("Good two.", prompt)


class TestStructuringLifecycle(StructuringTestBase):
    async def test_reset_outputs_then_reprocess_rebuilds_cleanly(self) -> None:
        self.write_lines([
            timeline_line(1.0, "transcript", "Segment one."),
            timeline_line(2.0, "transcript", "Segment two.")
        ])
        agent = self.make_agent()
        await agent.process_all_pending()
        first_notes = self.notes_text()

        agent.reset_outputs()
        self.assertEqual(self.notes_text(), "")
        self.assertEqual(agent.load_state()["timeline_byte_offset"], 0)

        await agent.process_all_pending()
        self.assertEqual(self.notes_text().count("### "), first_notes.count("### "))

    async def test_finalize_flag_runs_final_pass_and_exits(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Last words of the lecture.")])
        self.settings.structuring_interval_seconds = 3600
        agent = self.make_agent()
        set_flag(self.session_dir, FINALIZE_FLAG_NAME)
        await asyncio.wait_for(agent.run(), timeout=20)
        self.assertIn("Binary Search", self.notes_text())


class TestBatchHelpers(unittest.TestCase):
    def test_extract_json_payload_variants(self) -> None:
        self.assertEqual(extract_json_payload('["a"]'), ["a"])
        self.assertEqual(extract_json_payload('```json\n["a"]\n```'), ["a"])
        self.assertEqual(extract_json_payload('```\n["a"]\n```'), ["a"])
        self.assertEqual(extract_json_payload('{"key_points": ["a", "b"]}'), ["a", "b"])

    def test_split_entries_respects_max_characters(self) -> None:
        entries = [
            {"content": "a" * 100, "_line_end_offset": 100},
            {"content": "b" * 50, "_line_end_offset": 200},
            {"content": "c" * 50, "_line_end_offset": 300}
        ]
        batches = split_entries_into_batches(entries, max_characters=150, final_offset=300)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0].entries), 2)
        self.assertEqual(batches[0].end_offset, 200)
        self.assertEqual(len(batches[1].entries), 1)
        self.assertEqual(batches[1].end_offset, 300)


if __name__ == "__main__":
    unittest.main()
