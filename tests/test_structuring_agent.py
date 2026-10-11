import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lecturelens.agents.structuring_agent import (
    SliceBatch,
    StructuringAgent,
    extract_json_payload,
    split_entries_into_batches
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import FINALIZE_FLAG_NAME, set_flag
from tests.support import pluck


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
        if "key technical terms" in text:
            return json.dumps([{"term": self.definition_term, "definition": "A search algorithm that runs in logarithmic time."}])
        if "action items" in text:
            return json.dumps(["Complete homework 3 by Friday."])
        if "study flashcards" in text:
            return json.dumps([{"question": "What is binary search?", "answer": "Logarithmic search."}])
        if "Compress the following" in text:
            return "Binary search divides the search space in half."
        return "## Binary Search\nBinary search is an efficient algorithm on sorted arrays."

    def summary_prompts(self) -> list[str]:
        matching: list[str] = []
        for prompt in self.prompts:
            if "NEW LECTURE TIMELINE SLICE" in prompt:
                matching.append(prompt)
        return matching


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

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_agent(self, router: Any = None, session_id: str = None) -> StructuringAgent:
        return StructuringAgent(
            settings=self.settings,
            session_id=session_id or self.session_id,
            llm_router=router or self.router
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
            timeline_line(101.0, "slide", "Binary Search: O(log n)", [], "c1")
        ])

    async def test_process_batch_writes_notes_cards_and_concepts(self) -> None:
        agent = self.make_agent()
        await agent.process_batch()
        content = self.notes_text()
        self.assertIn("Binary Search", content)
        self.assertIn("Important Exam Topic", content)
        self.assertIn("Today we cover binary search.", content)
        self.assertIn("Complete homework 3", content)
        concepts = agent.concept_graph.get_all_concepts_for_session(self.session_id)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["concept_name"], "binary search")
        self.assertTrue((self.session_dir / "flashcards.apkg").exists())

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
        self.assertEqual(agent.load_state()["processed_byte_offset"], self.timeline_path.stat().st_size)
        before = self.notes_text()
        self.assertEqual(await agent.process_all_pending(), 0)
        self.assertEqual(self.notes_text(), before)


class TestStructuringReliability(StructuringTestBase):
    async def test_llm_failure_keeps_data_and_retries_later(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Heaps are trees.")])
        agent = self.make_agent()
        self.router.fail = True
        with self.assertRaises(RuntimeError):
            await agent.process_all_pending()
        self.assertEqual(agent.load_state()["processed_byte_offset"], 0)
        self.assertEqual(self.notes_text(), "")

        self.router.fail = False
        self.assertEqual(await agent.process_all_pending(), 1)
        self.assertIn("Binary Search", self.notes_text())

    async def test_guarded_pass_swallows_error_and_counts_failure(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Heaps are trees.")])
        agent = self.make_agent()
        self.router.fail = True
        await agent.run_guarded_pass()
        self.assertEqual(agent.consecutive_failures, 1)
        self.router.fail = False
        await agent.run_guarded_pass()
        self.assertEqual(agent.consecutive_failures, 0)
        self.assertIn("Binary Search", self.notes_text())

    async def test_partial_last_line_is_left_for_next_pass(self) -> None:
        complete = timeline_line(1.0, "transcript", "First complete line.")
        partial = timeline_line(2.0, "transcript", "Second line being written")
        self.write_lines([complete, partial[:25]])
        agent = self.make_agent()
        await agent.process_all_pending()
        self.assertEqual(agent.load_state()["processed_byte_offset"], len(complete.encode("utf-8")))
        self.assertIn("First complete line.", self.router.summary_prompts()[0])

        self.write_lines([partial[25:]])
        await agent.process_all_pending()
        self.assertIn("Second line being written", self.router.summary_prompts()[1])
        self.assertNotIn("First complete line.", self.router.summary_prompts()[1].split("NEW LECTURE TIMELINE SLICE")[1])

    async def test_restart_resumes_without_duplicating_notes(self) -> None:
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

    async def test_flashcards_survive_restart_and_are_deduplicated(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Segment one.", ["exam_relevant"])])
        first = self.make_agent()
        await first.process_all_pending()
        first_cards = first.load_flashcards()
        self.assertGreaterEqual(len(first_cards), 1)

        self.write_lines([timeline_line(2.0, "transcript", "Segment two.", ["exam_relevant"])])
        restarted = self.make_agent()
        restarted.llm_router = self.router
        await restarted.process_all_pending()
        questions = pluck(restarted.load_flashcards(), "question")
        self.assertEqual(len(questions), len(set(questions)))
        self.assertTrue(set(pluck(first_cards, "question")).issubset(set(questions)))

    async def test_blank_and_corrupt_lines_do_not_break_offsets(self) -> None:
        good_one = timeline_line(1.0, "transcript", "Good one.")
        good_two = timeline_line(2.0, "transcript", "Good two.")
        self.write_lines([good_one, "\n", "{broken json\n", good_two])
        agent = self.make_agent()
        await agent.process_all_pending()
        self.assertEqual(agent.load_state()["processed_byte_offset"], self.timeline_path.stat().st_size)
        prompt = self.router.summary_prompts()[0]
        self.assertIn("Good one.", prompt)
        self.assertIn("Good two.", prompt)

    async def test_unsorted_late_entries_are_ordered_by_timestamp(self) -> None:
        self.write_lines([
            timeline_line(50.0, "board", "late board text"),
            timeline_line(10.0, "transcript", "early speech")
        ])
        await self.make_agent().process_all_pending()
        prompt = self.router.summary_prompts()[0]
        self.assertLess(prompt.index("early speech"), prompt.index("late board text"))

    async def test_large_backlog_is_split_into_bounded_batches(self) -> None:
        self.settings.max_slice_characters = 400
        lines = []
        for index in range(30):
            lines.append(timeline_line(float(index), "transcript", f"sentence number {index} " + "x" * 60))
        self.write_lines(lines)
        agent = self.make_agent()
        processed = await agent.process_all_pending()
        self.assertGreater(processed, 3)
        self.assertEqual(agent.load_state()["processed_byte_offset"], self.timeline_path.stat().st_size)
        for prompt in self.router.summary_prompts():
            self.assertLess(len(prompt), 2500)

    async def test_context_is_bounded_when_saved(self) -> None:
        self.settings.context_max_characters = 50
        agent = self.make_agent()
        agent.save_state("y" * 500, 0)
        self.assertEqual(len(agent.load_state()["running_context"]), 50)

    async def test_corrupt_state_file_is_tolerated(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Hello.")])
        (self.session_dir / "context.json").write_text("{oops", encoding="utf-8")
        agent = self.make_agent()
        self.assertEqual(await agent.process_all_pending(), 1)


class TestStructuringContent(StructuringTestBase):
    async def test_example_sentence_is_not_marked_exam_relevant(self) -> None:
        from lecturelens.core.importance import detect_importance_flags
        text = "For example, consider sorting."
        self.write_lines([timeline_line(1.0, "transcript", text, detect_importance_flags(text))])
        await self.make_agent().process_all_pending()
        self.assertNotIn("Important Exam Topic", self.notes_text())

    async def test_confusion_marker_quotes_the_confused_line(self) -> None:
        self.write_lines([
            timeline_line(1.0, "transcript", "Wait what, I don't understand the pivot.", ["confusion"]),
            timeline_line(2.0, "transcript", "The pivot splits the array.")
        ])
        await self.make_agent().process_all_pending()
        notes = self.notes_text()
        self.assertIn("⚠️ **Review**: Wait what, I don't understand the pivot.", notes)
        self.assertNotIn("Review**: The pivot splits", notes)

    async def test_flag_markers_do_not_apply_to_whole_batch(self) -> None:
        self.write_lines([
            timeline_line(1.0, "transcript", "Unflagged filler.", []),
            timeline_line(2.0, "transcript", "This will be on the exam.", ["exam_relevant"])
        ])
        await self.make_agent().process_all_pending()
        notes = self.notes_text()
        self.assertIn("> - This will be on the exam.", notes)
        self.assertNotIn("> - Unflagged filler.", notes)

    async def test_revisited_slide_is_marked_and_truncated_in_prompt(self) -> None:
        long_text = "slide text " * 100
        self.write_lines([timeline_line(1.0, "slide", long_text, [], None, {"revisited_slide_number": 3})])
        await self.make_agent().process_all_pending()
        prompt = self.router.summary_prompts()[0]
        self.assertIn("(returned to earlier slide 3)", prompt)
        self.assertNotIn(long_text, prompt)

    async def test_empty_content_produces_no_notes_section(self) -> None:
        self.write_lines([timeline_line(1.0, "slide", "   ")])
        agent = self.make_agent()
        await agent.process_all_pending()
        self.assertEqual(self.notes_text(), "")
        self.assertGreater(agent.load_state()["processed_byte_offset"], 0)

    async def test_previously_covered_note_across_sessions_of_one_course(self) -> None:
        self.settings.course_name = "algorithms"
        self.write_lines([timeline_line(1.0, "transcript", "Binary search intro.")])
        await self.make_agent().process_all_pending()

        second_id = "lecture_two"
        second_dir = self.settings.create_session_directories(second_id)
        with open(second_dir / "timeline.jsonl", "a", encoding="utf-8") as handle:
            handle.write(timeline_line(5.0, "transcript", "Binary search again."))
        second_agent = self.make_agent(session_id=second_id)
        await second_agent.process_all_pending()
        notes = (second_dir / "notes.md").read_text(encoding="utf-8")
        self.assertIn(f"Previously covered in {self.session_id}", notes)
        course_rows = second_agent.course_graph.find_concept("binary search")
        self.assertEqual(len(course_rows), 2)

    async def test_concepts_in_a_single_session_do_not_flag_themselves(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Binary search intro.")])
        await self.make_agent().process_all_pending()
        self.assertNotIn("Previously covered", self.notes_text())


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
        self.assertEqual(agent.concept_graph.get_all_concepts_for_session(self.session_id), [])
        self.assertEqual(agent.load_state()["processed_byte_offset"], 0)

        await agent.process_all_pending()
        self.assertEqual(self.notes_text().count("### "), first_notes.count("### "))
        self.assertEqual(len(agent.concept_graph.get_all_concepts_for_session(self.session_id)), 1)

    async def test_finalize_flag_runs_final_pass_and_exits(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Last words of the lecture.")])
        self.settings.structuring_interval_seconds = 3600
        agent = self.make_agent()
        set_flag(self.session_dir, FINALIZE_FLAG_NAME)
        await asyncio.wait_for(agent.run(), timeout=20)
        self.assertIn("Binary Search", self.notes_text())

    async def test_periodic_pass_runs_on_interval(self) -> None:
        self.write_lines([timeline_line(1.0, "transcript", "Interval content.")])
        self.settings.structuring_interval_seconds = 1
        agent = self.make_agent()

        async def stopper() -> None:
            for _ in range(60):
                await asyncio.sleep(0.2)
                if self.notes_text():
                    break
            set_flag(self.session_dir, FINALIZE_FLAG_NAME)

        await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=30)
        self.assertIn("Binary Search", self.notes_text())


class TestBatchHelpers(unittest.TestCase):
    def test_extract_json_payload_variants(self) -> None:
        self.assertEqual(extract_json_payload('["a"]'), ["a"])
        self.assertEqual(extract_json_payload('```json\n["a"]\n```'), ["a"])
        self.assertEqual(extract_json_payload('```\n["a"]\n```'), ["a"])
        self.assertIsNone(extract_json_payload("not json"))

    def test_split_entries_into_batches_offsets(self) -> None:
        records = []
        for index in range(6):
            records.append(({"content": "x" * 100, "timestamp_epoch": index}, (index + 1) * 10))
        batches = split_entries_into_batches(records, 300, 60)
        self.assertGreater(len(batches), 1)
        self.assertEqual(batches[-1].end_offset, 60)
        total_entries = 0
        for batch in batches:
            total_entries += len(batch.entries)
        self.assertEqual(total_entries, 6)
        offsets = []
        for batch in batches:
            offsets.append(batch.end_offset)
        self.assertEqual(offsets, sorted(offsets))

    def test_only_corrupt_records_yield_no_batches(self) -> None:
        self.assertEqual(split_entries_into_batches([(None, 5), (None, 9)], 300, 9), [])


class TestSlidingOverlap(StructuringTestBase):
    async def test_overlap_extracted_and_passed_to_subsequent_batch(self) -> None:
        self.write_lines([
            timeline_line(10.0, "transcript", "First part of theorem: A is true."),
            timeline_line(110.0, "transcript", "Trailing sentence: and therefore B holds.")
        ])
        agent = self.make_agent()
        await agent.process_batch()

        self.assertGreater(len(agent.current_overlap), 0)
        self.assertEqual(agent.current_overlap[-1]["content"], "Trailing sentence: and therefore B holds.")

        self.write_lines([
            timeline_line(125.0, "transcript", "Because B holds, theorem C is proven.")
        ])
        await agent.process_batch()

        prompts = self.router.summary_prompts()
        self.assertEqual(len(prompts), 2)
        second_prompt = prompts[1]
        self.assertIn("PREVIOUS OVERLAP BUFFER", second_prompt)
        self.assertIn("Trailing sentence: and therefore B holds.", second_prompt)
        self.assertIn("Because B holds, theorem C is proven.", second_prompt)

    async def test_overlap_persists_in_context_json_across_reloads(self) -> None:
        self.write_lines([
            timeline_line(50.0, "transcript", "Alpha concept"),
            timeline_line(100.0, "transcript", "Beta concept ending at 100s")
        ])
        agent1 = self.make_agent()
        await agent1.process_batch()

        state_path = self.session_dir / "context.json"
        self.assertTrue(state_path.exists())
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIn("overlap_entries", state_data)
        self.assertGreater(len(state_data["overlap_entries"]), 0)

        agent2 = self.make_agent()
        self.assertEqual(len(agent2.current_overlap), len(state_data["overlap_entries"]))
        self.assertEqual(agent2.current_overlap[-1]["content"], "Beta concept ending at 100s")


class TestLLMFlashcardGeneration(StructuringTestBase):
    async def test_llm_generated_flashcards_are_preserved_directly_without_hardcoding(self) -> None:
        class RichCardRouter(MockRouter):
            async def ainvoke(self, messages: list) -> str:
                text = str(messages[0].content)
                self.prompts.append(text)
                if "key technical terms" in text:
                    return json.dumps([{"term": "Ms Fairchild", "definition": "An elegant passenger."}])
                if "action items" in text:
                    return "[]"
                if "study flashcards" in text:
                    return json.dumps([
                        {
                            "question": "In 'Hearts and Hands', what role does Ms. Fairchild play in enabling Mr. Easton's deception?",
                            "answer": "Ms. Fairchild's high social standing and prior acquaintance create pressure for Easton, prompting him to feign being the Marshal to conceal his convict status."
                        }
                    ])
                if "Compress the following" in text:
                    return "Summary context."
                return "## Literature Analysis\nDiscussion of Hearts and Hands."

        agent = self.make_agent(router=RichCardRouter())
        self.write_lines([timeline_line(1.0, "transcript", "Today we discuss Hearts and Hands.")])
        await agent.process_all_pending()
        cards = agent.load_flashcards()
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["question"], "In 'Hearts and Hands', what role does Ms. Fairchild play in enabling Mr. Easton's deception?")
        self.assertIn("Ms. Fairchild's high social standing", cards[0]["answer"])

    async def test_flashcard_prompt_provides_full_lecture_context_to_llm(self) -> None:
        agent = self.make_agent()
        self.write_lines([
            timeline_line(1.0, "transcript", "Binary search runs in O(log n).", ["exam_relevant"])
        ])
        await agent.process_all_pending()
        flashcard_prompts = [p for p in self.router.prompts if "study flashcards" in p]
        self.assertEqual(len(flashcard_prompts), 1)
        self.assertIn("LECTURE SUMMARY:", flashcard_prompts[0])
        self.assertIn("KEY CONCEPTS & DEFINITIONS DISCUSSED:", flashcard_prompts[0])

    async def test_flashcards_empty_when_no_content_in_slice(self) -> None:
        agent = self.make_agent()
        self.write_lines([timeline_line(1.0, "transcript", "")])
        await agent.process_all_pending()
        cards = agent.load_flashcards()
        self.assertEqual(len(cards), 0)


if __name__ == "__main__":
    unittest.main()
