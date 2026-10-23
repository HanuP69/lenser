import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from lecturelens.agents.structuring_agent import (
    STRUCTURED_DELIMITER,
    StructuringAgent,
    StructuringBatchError,
    build_running_context,
    extract_json_payload,
    scrub_reasoning_chatter,
    split_entries_into_batches,
    split_notes_and_structured,
    trim_prompt_groups
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import FINALIZE_FLAG_NAME, set_flag


class MockRouter:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.fail = False
        self.definition_term = "Binary Search"
        # When the structured block comes back broken, the repair call is
        # expected to fix it, and the number of prompts says how many
        # requests the batch took.
        self.break_json = False
        self.bare_math = False
        # When the notes come back with the model's reasoning inside,
        # the scrub has to take the self-talk out without a second ask.
        self.leaky_notes = False

    async def ainvoke(self, messages: list) -> str:
        if self.fail:
            raise RuntimeError("llm unavailable")
        text = str(messages[0].content)
        self.prompts.append(text)
        if "CURRENT LECTURE EVENTS" in text:
            # The one call per batch: the notes, then the delimiter, then
            # the three lists of the structured block.
            if self.leaky_notes and not self.bare_math:
                # The shapes the live model actually leaked into the
                # notes (the reasoning of the second batch), kept as a
                # standing regression against the scrub.
                notes = (
                    "## Binary Search\n"
                    "    Binary search is an efficient algorithm on sorted arrays.\n"
                    "\n"
                    "    *   `definitions`: [] (none new)\n"
                    "    *   `action_items`: [\"Complete homework 3 by Friday.\"]\n"
                    "\n"
                    "*Self-Correction on LaTeX:* The prompt gives `$64.8\\%$` as an example.\n"
                    "\n"
                    "    Wait, looking at the transcript again: \"NETURE\" This looks like noise.\n"
                    "\n"
                    "*One more check on the Figure marker:*\n"
                    "    The prompt says: \"The marker must appear once.\"\n"
                    "\n"
                    "*JSON check:*\n"
                    "    `{\"definitions\": [], \"action_items\": [\"Do the thing\"]}`\n"
                    "\n"
                    "*Wait, is there math in the text?* No.\n"
                    "\n"
                    "    Let's double check the slide 2 transcript.\n"
                    "    \"BEGIN TAPE Back to the spoken part.\"\n"
                    "    Translation/Interpretation: \"Back to the spoken part.\"\n"
                    "\n"
                    "    Final check on formatting:\n"
                    "    1. Markdown (##, bullets, paragraphs) - Yes.\n"
                    "    2. ===NOTES_END=== - Yes.\n"
                    "\n"
                    "*Double check figure ID:* `fig_7e7af948fb072de6`. Correct.\n"
                    "\n"
                    "*Finalizing content:*\n"
                    "## Binary Search\n"
                    "    Binary search divides the search space in half."
                )
            else:
                notes = (
                    "## Hashing\nThe load factor is O(1) per lookup."
                    if self.bare_math
                    else "## Binary Search\nBinary search is an efficient algorithm on sorted arrays."
                )
            structured = (
                {"definitions": [{"term": self.definition_term, "definition": "A search algorithm that runs in logarithmic time."}],
                 "action_items": [],
                 "key_points": []}
                if self.bare_math else {
                    "definitions": [{"term": self.definition_term, "definition": "A search algorithm that runs in logarithmic time."}],
                    "action_items": ["Complete homework 3 by Friday."],
                    "key_points": ["Binary search runs in O(log n).", "Requires sorted input."],
                }
            )
            if self.break_json:
                return f"{notes}\n{STRUCTURED_DELIMITER}\nnot json at all [<<<broken"
            return f"{notes}\n\n{STRUCTURED_DELIMITER}\n\n{json.dumps(structured)}"
        if "supposed to follow them came out broken" in text:
            # The repair call: only the structured block is asked for.
            return json.dumps({
                "definitions": [{"term": self.definition_term, "definition": "A search algorithm that runs in logarithmic time."}],
                "action_items": ["Complete homework 3 by Friday."],
                "key_points": ["Binary search runs in O(log n)."]
            })
        if "Reply with the rewritten markdown only" in text:
            # The latex pass: only the mathematics changes.
            return "## Hashing\nThe load factor is $O(1)$ per lookup."
        raise AssertionError(f"an unexpected call reached the router: {text[:90]!r}")

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
        self.assertIn("Binary Search", segment.concept_tags)
        self.assertEqual(segment.slide_references, [1])

    async def test_empty_summary_is_retried_then_falls_back_to_raw_content(self) -> None:
        """An empty summary used to publish an empty note segment, burn a
        sequence number and advance the offset, losing the batch."""
        agent = self.make_agent()
        agent.max_batch_retries = 1

        original_synthesis = agent.synthesis

        async def empty_synthesis(entries: Any, overlap_entries: Any, context: str) -> dict:
            result = await original_synthesis(entries, overlap_entries, context)
            return {**result, "summary": ""}

        agent.synthesis = empty_synthesis  # type: ignore[method-assign]

        # First attempt: the batch is retried, nothing is published and the
        # byte offset is not advanced.
        with self.assertRaises(StructuringBatchError):
            await agent.process_batch()
        self.assertEqual(self.bus.published, [])
        self.assertEqual(agent.sequence_number, 0)
        self.assertEqual(agent.load_state()["timeline_byte_offset"], 0)
        self.assertEqual(self.notes_text(), "")

        # Second attempt: the retry budget is exhausted, so the raw timeline
        # content is used instead of dropping the batch.
        await agent.process_batch()

        self.assertEqual(len(self.bus.published), 1)
        stream_name, segment = self.bus.published[0]
        self.assertIn("stream:notes", stream_name)
        self.assertEqual(segment.sequence_number, 1)
        self.assertNotEqual(segment.content.strip(), "")
        self.assertIn("Today we cover binary search.", segment.content)
        self.assertIn("Today we cover binary search.", self.notes_text())
        self.assertGreater(agent.load_state()["timeline_byte_offset"], 0)

    async def test_retry_counter_survives_a_restart(self) -> None:
        """The counter lives in its own file, so the retry cap also applies
        across restarts. The reprocess command restores context.json from a
        backup when a run fails, which would otherwise wipe the counter."""
        agent = self.make_agent()
        agent.max_batch_retries = 1

        original_synthesis = agent.synthesis

        async def empty_synthesis(entries: Any, overlap_entries: Any, context: str) -> dict:
            result = await original_synthesis(entries, overlap_entries, context)
            return {**result, "summary": ""}

        agent.synthesis = empty_synthesis  # type: ignore[method-assign]

        with self.assertRaises(StructuringBatchError):
            await agent.process_batch()

        restarted = self.make_agent()
        self.assertEqual(restarted._batch_attempts, agent._batch_attempts)
        self.assertEqual(sum(restarted._batch_attempts.values()), 1)

    async def test_quarantined_entries_are_written_once_per_batch(self) -> None:
        """The quarantine was written before the synthesis, so a retried batch
        appended the same entries again."""
        agent = self.make_agent()
        agent.max_batch_retries = 1
        self.write_lines([
            timeline_line(300.0, "transcript", "This video is sponsored by our friends", ["ad"]),
            timeline_line(301.0, "transcript", "The definition of a binary search tree", ["definition"]),
        ])

        async def failing_synthesis(entries: Any, overlap_entries: Any, context: str) -> dict:
            raise RuntimeError("llm is down")

        agent.synthesis = failing_synthesis  # type: ignore[method-assign]

        class FakeDrift:
            def __init__(self) -> None:
                self.quarantined_entries: list[dict[str, Any]] = []

            def filter_batch_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
                quarantined = [e for e in entries if "ad" in (e.get("importance_flags") or [])]
                self.quarantined_entries = list(quarantined)
                return [e for e in entries if "ad" not in (e.get("importance_flags") or [])]

            def to_dict(self) -> dict[str, Any]:
                return {"quarantined_entries": list(self.quarantined_entries)}

            def load_dict(self, data: dict[str, Any]) -> None:
                return None

            def reset(self) -> None:
                self.quarantined_entries.clear()

        agent.drift_filter = FakeDrift()  # type: ignore[assignment]

        # The first attempt fails, so nothing is written yet.
        with self.assertRaises(StructuringBatchError):
            await agent.process_batch()
        self.assertFalse(agent.quarantine_path.exists())

        # The second attempt falls back to the raw content and the batch
        # finishes, so the quarantine is written exactly once.
        await agent.process_batch()
        self.assertTrue(agent.quarantine_path.exists())
        lines = [
            json.loads(line)
            for line in agent.quarantine_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 1)
        self.assertIn("sponsored by", lines[0]["content"])

    async def test_raw_fallback_is_counted_in_the_stats(self) -> None:
        """The raw text fallback has to be visible, otherwise the unedited
        parts of the notes are invisible to the operator."""
        agent = self.make_agent()
        agent.max_batch_retries = 1

        async def failing_synthesis(entries: Any, overlap_entries: Any, context: str) -> dict:
            raise RuntimeError("llm is down")

        agent.synthesis = failing_synthesis  # type: ignore[method-assign]

        with self.assertRaises(StructuringBatchError):
            await agent.process_batch()
        self.assertNotIn("raw_fallback_batches", agent.control.stats)

        await agent.process_batch()
        self.assertEqual(agent.control.stats.get("raw_fallback_batches"), 1)
        self.assertIsNotNone(agent.control.stats.get("raw_fallback_offset"))

    async def test_reset_outputs_clears_the_quarantine_file(self) -> None:
        """The quarantine belongs to the run, so a reprocess must not append
        to the file of the previous run."""
        agent = self.make_agent()
        agent.quarantine_path.write_text('{"event_id": "old"}\n', encoding="utf-8")
        self.assertTrue(agent.quarantine_path.exists())

        agent.reset_outputs()
        self.assertFalse(agent.quarantine_path.exists())

    async def test_correlated_events_reach_the_llm_as_one_unit(self) -> None:
        await self.make_agent().process_batch()
        prompt = self.router.summary_prompts()[0]
        self.assertIn("[TRANSCRIPT][exam_relevant] Today we cover binary search. | [SLIDE] Binary Search: O(log n)", prompt)

    async def test_running_context_is_saved_and_fed_to_next_batch(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        state = agent.load_state()
        # The context is built by counting lines of the notes; the model
        # is not asked for it any more.
        self.assertTrue(state["running_context"].startswith("Topics so far:"))
        self.assertIn("Binary Search", state["running_context"])
        self.write_lines([timeline_line(200.0, "transcript", "Next we discuss hashing.")])
        await agent.process_batch()
        self.assertIn("Binary Search", self.router.summary_prompts()[1])

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

    async def test_returned_running_context_respects_max_lines(self) -> None:
        self.settings.context_max_lines = 2
        agent = self.make_agent()
        batches = agent.read_pending_batches()
        multi_line_context = "Line 1\nLine 2\nLine 3\nLine 4"
        returned_context = await agent.process_single_batch(batches[0], multi_line_context)
        self.assertLessEqual(len(returned_context.splitlines()), 2)
        self.assertEqual(agent.load_state()["running_context"], returned_context)


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


class TestOneCallPerBatch(StructuringTestBase):
    """The shape of Phase 1: one request per batch, two at the very most."""

    def setUp(self) -> None:
        super().setUp()
        self.write_lines([
            timeline_line(100.0, "transcript", "Today we cover binary search.", [], "c1"),
            timeline_line(101.0, "slide", "Binary Search: O(log n)", [], "c1", metadata={"slide_number": 1})
        ])

    async def test_one_call_carries_the_notes_and_the_lists(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        # One request for the whole batch: no definitions call, no action
        # items call, no key points call, no compression call.
        self.assertEqual(len(self.router.prompts), 1)
        notes = self.notes_text()
        self.assertIn("Binary Search", notes)
        self.assertIn("**Action Items**", notes)
        stream_name, segment = self.bus.published[0]
        self.assertEqual(segment.definitions[0]["term"], "Binary Search")
        self.assertEqual(segment.key_points, ["Binary search runs in O(log n).", "Requires sorted input."])

    async def test_a_broken_structured_block_costs_one_repair(self) -> None:
        self.router.break_json = True
        agent = self.make_agent()
        await agent.process_batch()
        # The notes call, then exactly one repair for the lost block.
        self.assertEqual(len(self.router.prompts), 2)
        self.assertIn("Binary Search", self.notes_text())
        _, segment = self.bus.published[0]
        self.assertEqual(segment.definitions[0]["term"], "Binary Search")
        # The action items come out of the repaired block and land in the
        # section of the notes.
        self.assertIn("Complete homework 3", segment.content)

    async def test_an_empty_batch_costs_no_call_at_all(self) -> None:
        other = self.temp_dir.name + "/empty_test"
        settings = Settings()
        settings.output_dir = Path(other)
        session_dir = settings.create_session_directories("empty_struct")
        with open(session_dir / "timeline.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(timeline_line(1.0, "transcript", ""))
            handle.write(timeline_line(2.0, "slide", "", [], metadata={"slide_number": 1}))
        router = MockRouter()
        agent = StructuringAgent(
            settings=settings,
            session_id="empty_struct",
            llm_router=router,
            redis_bus=MockRedisBus()
        )
        await agent.process_batch()
        self.assertEqual(len(router.prompts), 0)
        self.assertEqual(agent.load_state()["timeline_byte_offset"], (session_dir / "timeline.jsonl").stat().st_size)
        self.assertEqual(agent.control.stats.get("empty_batches_skipped"), 1)
        self.assertFalse((session_dir / "notes.md").exists())

    async def test_the_latex_pass_runs_only_when_the_checks_fail(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        # The mathematics is already inside the dollar signs is not the case
        # here: there is no mathematics at all, so no pass was staged.
        self.assertEqual(len(self.router.prompts), 1)

        self.router.bare_math = True
        self.router.prompts.clear()
        other = self.make_agent()
        self.write_lines([timeline_line(150.0, "transcript", "Now we cover hashing.", [], "c2")])
        await other.process_batch()
        self.assertEqual(len(self.router.prompts), 2)
        self.assertIn("$O(1)$", self.notes_text())

    async def test_the_model_s_self_talk_never_reaches_the_notes(self) -> None:
        self.router.leaky_notes = True
        agent = self.make_agent()
        await agent.process_batch()
        self.assertEqual(len(self.router.prompts), 1)
        notes = self.notes_text()
        # The clean substance stays.
        self.assertIn("## Binary Search", notes)
        self.assertIn("divides the search space in half.", notes)
        self.assertIn("**Action Items**", notes)
        # Every shape of the reasoning that the live model leaked is gone.
        self.assertNotIn("Self-Correction", notes)
        self.assertNotIn("`definitions`:", notes)
        self.assertNotIn("`action_items`:[", notes.replace(" ", ""))
        self.assertNotIn("Wait,", notes)
        self.assertNotIn("One more check", notes)
        self.assertNotIn("The prompt says", notes)
        self.assertNotIn("JSON check", notes)
        self.assertNotIn("double check", notes)
        self.assertNotIn("Translation/Interpretation", notes)
        self.assertNotIn("Final check on formatting", notes)
        self.assertNotIn("- Yes.", notes)
        self.assertNotIn("Double check figure ID", notes)
        self.assertNotIn("Finalizing content", notes)
        self.assertNotIn("NOTES_END", notes)
        # The lists behind the delimiter were read as usual.
        _, segment = self.bus.published[0]
        self.assertEqual(segment.definitions[0]["term"], "Binary Search")

    async def test_the_prompt_text_is_trimmed_by_counting(self) -> None:
        self.router.prompts.clear()
        self.router.break_json = False
        self.write_lines([
            # The same sentence handed over twice under two groups.
            timeline_line(300.0, "transcript", "We begin with caching today.", [], "c1"),
            timeline_line(301.0, "transcript", "We begin with caching today.", [], "c2"),
            # Filler plus overlapping chunks of speech.
            timeline_line(303.0, "transcript", "umm umm let me explain the idea again.", [], "c3"),
            timeline_line(304.0, "transcript", "the idea again in a table.", [], "c4"),
            # A revisited slide repeats the slide it belongs to.
            timeline_line(305.0, "slide", "Caching in memory", [], "c5",
                          metadata={"slide_number": 3, "revisited_slide_number": 3}),
            timeline_line(306.0, "slide", "Caching in memory", [], "c6", metadata={"slide_number": 3})
        ])
        agent = StructuringAgent(
            settings=self.settings,
            session_id=self.session_id,
            llm_router=MockRouter(),
            redis_bus=self.bus
        )
        await agent.process_batch()
        # All of the lines share one batch, so exactly one ask carried
        # the trimmed text.
        self.assertEqual(len(agent.llm_router.prompts), 1)
        prompt = agent.llm_router.prompts[0]
        self.assertEqual(prompt.count("We begin with caching today."), 1)
        self.assertNotIn("umm", prompt)
        self.assertEqual(prompt.count("Caching in memory"), 1)
        self.assertIn("the idea again in a table.", prompt)


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

    def test_the_start_of_a_batch_does_not_move_when_more_events_arrive(self) -> None:
        """The retry counter is keyed by the start of the batch.

        Keyed by the end of the batch it always looked like the first
        attempt, because the end moves when a new event arrives. The raw
        content fallback then never happened and the lecture material was
        lost.
        """
        entries = [
            {"content": "a" * 100, "_line_start_offset": 10, "_line_end_offset": 100},
            {"content": "b" * 50, "_line_start_offset": 100, "_line_end_offset": 200}
        ]
        first = split_entries_into_batches(entries, max_characters=1000, final_offset=200)
        self.assertEqual(len(first), 1)

        # A new event arrives, so the batch now ends further along.
        entries.append(
            {"content": "c" * 50, "_line_start_offset": 200, "_line_end_offset": 300}
        )
        second = split_entries_into_batches(entries, max_characters=1000, final_offset=300)

        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].start_offset, second[0].start_offset)
        self.assertNotEqual(first[0].end_offset, second[0].end_offset)

    def test_the_time_mark_of_a_section_points_at_the_speech(self) -> None:
        """A slide that was published late must not set the time mark.

        The text of a slide is read by a model that takes a while, so the
        slide reaches the timeline long after it was captured. It then
        sits in front of a section that is about something else, and the
        section got a time mark of the first twenty seconds of the
        lecture while it was about the fifth minute.
        """
        from lecturelens.agents.structuring_agent import StructuringAgent

        agent = StructuringAgent.__new__(StructuringAgent)
        entries = [
            {"source_type": "slide", "timestamp_epoch": 22.0, "content": "a slide"},
            {"source_type": "transcript", "timestamp_epoch": 300.0, "content": "speech"},
            {"source_type": "transcript", "timestamp_epoch": 320.0, "content": "more speech"},
        ]
        self.assertEqual(agent.section_start_timestamp(entries), 300.0)

    def test_the_time_mark_falls_back_to_the_first_entry_without_speech(self) -> None:
        from lecturelens.agents.structuring_agent import StructuringAgent

        agent = StructuringAgent.__new__(StructuringAgent)
        entries = [
            {"source_type": "slide", "timestamp_epoch": 22.0, "content": "a slide"},
            {"source_type": "slide", "timestamp_epoch": 40.0, "content": "another slide"},
        ]
        self.assertEqual(agent.section_start_timestamp(entries), 22.0)

    def test_a_message_that_is_too_long_is_recognised(self) -> None:
        from lecturelens.agents.structuring_agent import is_message_too_long_error

        self.assertTrue(is_message_too_long_error(Exception(
            "Error code: 400 - {'error': {'message': 'This provider limits the length "
            "of a single message. Shorten your last message, or retry and another "
            "provider will take it.'}}"
        )))
        self.assertTrue(is_message_too_long_error(Exception("maximum context length exceeded")))
        self.assertTrue(is_message_too_long_error(Exception("context length exceeded")))
        self.assertFalse(is_message_too_long_error(Exception("rate limit reached")))
        self.assertFalse(is_message_too_long_error(Exception("connection refused")))

    def test_a_long_batch_is_split_in_half(self) -> None:
        """A provider that refuses a long message must not cost the notes.

        A whole lecture of eleven minutes is one batch when nothing was
        processed while it was recorded, and the provider refused it with
        a plain error, so the lecture was left without notes.
        """
        from lecturelens.agents import structuring_agent as sa

        settings = Settings()
        with tempfile.TemporaryDirectory() as folder:
            settings.output_dir = Path(folder)
            settings.create_session_directories("split_test")
            agent = sa.StructuringAgent(
                settings=settings,
                session_id="split_test",
                llm_router=None,
                redis_bus=None
            )

            seen_sizes: list[int] = []
            accepted_sizes: list[int] = []

            async def fake_synthesis(slice_entries: Any, overlap_entries: Any, context: str) -> dict:
                seen_sizes.append(len(slice_entries))
                if len(slice_entries) > 2:
                    raise RuntimeError(
                        "Error code: 400 - This provider limits the length of a single message."
                    )
                accepted_sizes.append(len(slice_entries))
                return {"summary": "a summary of the batch"}

            agent.synthesis = fake_synthesis

            entries = [
                {
                    "source_type": "transcript",
                    "timestamp_epoch": float(index),
                    "content": f"entry {index}",
                    "_line_end_offset": (index + 1) * 10,
                }
                for index in range(8)
            ]
            batch = sa.SliceBatch(entries=entries, end_offset=80, start_offset=0)

            asyncio.run(agent.process_single_batch(batch, "context"))

            # The batch was split until the provider took it.
            self.assertGreater(len(seen_sizes), 1)
            self.assertLessEqual(max(accepted_sizes), 2)
            self.assertEqual(len(accepted_sizes), 4)
            # Every entry of the batch was done, none was dropped.
            self.assertEqual(sum(accepted_sizes), 8)


class TestTheOneCallContract(unittest.TestCase):
    """The parsing and the deterministic helpers of the one call."""

    def test_the_notes_and_the_block_are_split_at_the_delimiter(self) -> None:
        raw = (
            "## Topic\nSome notes about the topic.\n\n"
            + STRUCTURED_DELIMITER
            + '\n\n{"definitions": [{"term": "Cache", "definition": "a fast store"}],'
            + ' "action_items": [], "key_points": ["one"]}'
        )
        notes, structured = split_notes_and_structured(raw)
        self.assertEqual(notes, "## Topic\nSome notes about the topic.")
        self.assertEqual(structured["definitions"][0]["term"], "Cache")
        self.assertEqual(structured["key_points"], ["one"])
        # The tags of the concepts come from the terms without asking.
        self.assertEqual(structured["concept_tags"], ["Cache"])

    def test_a_lost_delimiter_still_leaves_the_notes_whole(self) -> None:
        raw = (
            "## Topic\nThe example {\"definitions\": [1, 2]} sits inside the notes.\n\n"
            '{"definitions": [{"term": "Real", "definition": "the real one"}],'
            + ' "action_items": ["do"], "key_points": []}'
        )
        notes, structured = split_notes_and_structured(raw)
        self.assertIn("The example", notes)
        self.assertNotIn('"Real"', notes)
        self.assertEqual(structured["definitions"][0]["term"], "Real")

    def test_a_block_that_is_not_the_lists_is_not_guesswork(self) -> None:
        notes, structured = split_notes_and_structured('{"a": 1, "b": 2}')
        self.assertEqual(notes, '{"a": 1, "b": 2}')
        self.assertIsNone(structured)

    def test_the_running_context_counts_lines_not_requests(self) -> None:
        first = build_running_context(
            "", "## Torque\nbody of the section",
            [{"term": "Torque", "definition": "a turning force"}],
            ["a point"]
        )
        self.assertTrue(first.startswith("Topics so far: Torque"))
        second = build_running_context(
            first, "## Angular Momentum\nmore",
            [{"term": "L", "definition": "angular momentum"}],
            ["another point"]
        )
        self.assertIn("Topics so far: Torque; Angular Momentum", second)
        self.assertIn("L = angular momentum", second)
        self.assertNotIn("Torque = ", second)

    def test_the_running_context_stays_short(self) -> None:
        summary = "".join(f"## Heading number {index}\ntext\n" for index in range(60))
        context = build_running_context("", summary, [], [])
        self.assertLessEqual(len(context), 1203)

    def test_the_prompt_loses_only_what_repeats(self) -> None:
        groups = [
            {"correlation_id": "a", "items": [
                {"source_type": "transcript", "content": "Okay umm so the hash is open.", "importance_flags": [], "metadata": {}},
                {"source_type": "transcript", "content": "Okay umm so the hash is open.", "importance_flags": [], "metadata": {}}
            ]},
            {"correlation_id": "b", "items": [
                {"source_type": "slide", "content": "Open Hashing", "importance_flags": [], "metadata": {"slide_number": 2}},
                {"source_type": "slide", "content": "Open Hashing", "importance_flags": [], "metadata": {"slide_number": 2, "revisited_slide_number": 2}}
            ]}
        ]
        trimmed = trim_prompt_groups(groups)
        texts = [
            str(item.get("content", ""))
            for group in trimmed for item in group["items"]
        ]
        self.assertEqual(texts.count("Okay so the hash is open."), 1)
        self.assertEqual(texts.count("Open Hashing"), 1)

    def test_the_model_s_reasoning_is_scrubbed_out(self) -> None:
        # The exact shapes the live model leaked into the notes of the
        # second batch of the p1_slides session.
        leaky = (
            "## The Popularity and Versatility of Redis\n"
            "\n"
            "    Redis is characterized as much more than a simple store.\n"
            "\n"
            "    *   `definitions`: [] (none new)\n"
            "    *   `action_items`: [\"Remember that Redis works with Python, Go, .NET\"]\n"
            "    *   `key_points`: [\"most loved\"]\n"
            "\n"
            "*Self-Correction on LaTeX:* The prompt gives `$64.8\\%$` as an example.\n"
            "\n"
            "    Wait, looking at the transcript again: \"NETURE DESCRIPTION\" This looks like noise.\n"
            "\n"
            "*One more check on the Figure marker:*\n"
            "    The prompt says: \"The marker must appear once.\"\n"
            "    Slide 2 is the \"Most Loved Database\" slide.\n"
            "\n"
            "*JSON check:*\n"
            "    `{\"definitions\": [], \"action_items\": [\"Note the language compatibility\"]}`\n"
            "\n"
            "*Wait, is there math in the text?* No.\n"
            "\n"
            "    Let's double check the slide 2 transcript.\n"
            "    \"BEGIN TAPE Back to the spoken part.\"\n"
            "    Translation/Interpretation: \"Back to the spoken part.\"\n"
            "\n"
            "    Final check on formatting:\n"
            "    1. Markdown (##, bullets, paragraphs) - Yes.\n"
            "    2. Figure marker on own line - Yes.\n"
            "    3. ===NOTES_END=== - Yes.\n"
            "    4. JSON - Yes.\n"
            "\n"
            "*Double check figure ID:* `fig_7e7af948fb072de6`. Correct.\n"
            "\n"
            "*Finalizing content:*\n"
            "## Clean Section\n"
            "    Real notes stay here."
        )
        cleaned = scrub_reasoning_chatter(leaky)
        for piece in (
            "## The Popularity and Versatility of Redis",
            "Redis is characterized as much more than a simple store.",
            "## Clean Section",
            "Real notes stay here.",
        ):
            self.assertIn(piece, cleaned)
        for gone in (
            "`definitions`:",
            "`action_items`:",
            "`key_points`:",
            "Self-Correction",
            "Wait, looking at the transcript",
            "One more check",
            "The prompt says",
            "Slide 2 is the",
            "JSON check",
            '"definitions"',
            "is there math in the text",
            "slide 2 transcript",
            "BEGIN TAPE",
            "Translation/Interpretation",
            "Final check on formatting",
            "Markdown (##, bullets, paragraphs) - Yes.",
            "NOTES_END",
            "Double check figure ID",
            "Finalizing content",
        ):
            self.assertNotIn(gone, cleaned)

    def test_bare_mathematics_is_found_outside_the_dollars(self) -> None:
        from lecturelens.agents.structuring_agent import math_left_unformatted

        self.assertTrue(math_left_unformatted("The cost is O(n log n) per query."))
        self.assertTrue(math_left_unformatted("Raise it to 2^n."))
        self.assertFalse(math_left_unformatted("The cost is $O(\\log n)$ per query."))
        self.assertFalse(math_left_unformatted("We compare the sizes of 10 and 10."))
        # A number with a caret inside a dollar span is not bare math.
        self.assertFalse(math_left_unformatted("$n \\cdot 2^k$ items"))
        self.assertFalse(math_left_unformatted("a ```python\ndef f(x):\n    pass\n``` block"))


if __name__ == "__main__":
    unittest.main()
