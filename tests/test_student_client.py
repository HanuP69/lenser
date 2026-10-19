import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.client.student_client import (
    PersonalizationEngine,
    StudentClient,
    StudentProfile
)
from lecturelens.schemas.events import GeneralNoteSegment


class MockLLMRouter:
    def __init__(self, output: str) -> None:
        self.output = output
        self.invoked_prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        self.invoked_prompts.append(messages[0].content)
        return self.output


class TestPersonalizationEngine(unittest.IsolatedAsyncioTestCase):
    def make_sample_segment(self) -> GeneralNoteSegment:
        return GeneralNoteSegment(
            session_id="cs101_lec",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Binary Search Analysis",
            content="Binary search runs in O(log n) time by halving search space.",
            key_points=["Search space halves every step", "Requires sorted array"],
            concept_tags=["binary_search", "complexity"]
        )

    async def test_english_profile_returns_original_without_extra_llm_call(self) -> None:
        profile = StudentProfile(preferred_language="English")
        router = MockLLMRouter("rewritten")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, segment.content)
        self.assertEqual(len(router.invoked_prompts), 0)

    async def test_hinglish_profile_invokes_llm_and_preserves_concepts(self) -> None:
        profile = StudentProfile(preferred_language="Hinglish")
        hinglish_output = "Binary search har step pe array ko half kar deta hai, isliye iska time complexity O(log n) hota hai."
        router = MockLLMRouter(hinglish_output)
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, hinglish_output)
        self.assertEqual(len(router.invoked_prompts), 1)
        self.assertIn("Hinglish", router.invoked_prompts[0])

    async def test_rewrite_validation_fallback_on_empty_or_hallucinated(self) -> None:
        profile = StudentProfile(preferred_language="Hindi")
        router = MockLLMRouter("Too short")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, segment.content)

    async def test_english_profile_with_concise_detail_rewrites(self) -> None:
        profile = StudentProfile(preferred_language="English", detail_level="concise")
        router = MockLLMRouter("Binary search halves search space in O(log n) time. [00:01:00] binary search complexity")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertIn("halves search space", result)
        self.assertEqual(len(router.invoked_prompts), 1)


class TestStudentClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_on_note_segment_received_appends_and_updates_sequence(self) -> None:
        profile = StudentProfile(student_id="student_42", preferred_language="English")
        client = StudentClient(
            server_url="ws://localhost:8765",
            session_id="lec_1",
            profile=profile,
            output_dir=self.output_dir
        )
        segment = GeneralNoteSegment(
            session_id="lec_1",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Recursion",
            content="A function calling itself.",
            key_points=["Base case needed"],
            concept_tags=["recursion"]
        )
        content = await client.on_note_segment_received(segment)
        self.assertEqual(content, "A function calling itself.")
        self.assertEqual(client.last_sequence_number, 1)
        self.assertTrue(client.notes_file.exists())
        written_text = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("A function calling itself.", written_text)

    async def test_older_or_duplicate_sequence_is_ignored(self) -> None:
        profile = StudentProfile(preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_1", profile, self.output_dir)
        client.last_sequence_number = 5
        segment = GeneralNoteSegment(
            session_id="lec_1",
            sequence_number=3,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Old Section",
            content="Old content",
            key_points=[],
            concept_tags=[]
        )
        result = await client.on_note_segment_received(segment)
        self.assertEqual(result, "")
        self.assertEqual(client.last_sequence_number, 5)

    async def test_client_buffers_gap_and_drains_when_missing_segment_arrives(self) -> None:
        profile = StudentProfile(student_id="student_gap", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_gap", profile, self.output_dir)
        client.last_sequence_number = 1

        seg3 = GeneralNoteSegment(
            session_id="lec_gap",
            sequence_number=3,
            timestamp_epoch=1700000020.0,
            start_time="00:00:20",
            end_time="00:00:30",
            heading="Section 3",
            content="Third section content.",
            key_points=[],
            concept_tags=[]
        )
        seg2 = GeneralNoteSegment(
            session_id="lec_gap",
            sequence_number=2,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Section 2",
            content="Second section content.",
            key_points=[],
            concept_tags=[]
        )

        res3 = await client.on_note_segment_received(seg3)
        self.assertEqual(res3, "")
        self.assertEqual(client.last_sequence_number, 1)
        self.assertIn(3, client.pending_segments)

        res2 = await client.on_note_segment_received(seg2)
        self.assertEqual(client.last_sequence_number, 3)
        self.assertEqual(len(client.pending_segments), 0)
        written_text = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("Second section content.", written_text)
        self.assertIn("Third section content.", written_text)

    async def test_client_persists_sequence_number_across_instances(self) -> None:
        profile = StudentProfile(student_id="student_persist", preferred_language="English")
        client1 = StudentClient("ws://localhost:8765", "lec_p", profile, self.output_dir)
        seg1 = GeneralNoteSegment(
            session_id="lec_p",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Section 1",
            content="Content 1",
            key_points=[],
            concept_tags=[]
        )
        await client1.on_note_segment_received(seg1)
        self.assertEqual(client1.last_sequence_number, 1)

        client2 = StudentClient("ws://localhost:8765", "lec_p", profile, self.output_dir)
        self.assertEqual(client2.last_sequence_number, 1)


    async def test_client_updates_student_graph_concepts(self) -> None:
        from lecturelens.knowledge.student_graph import StudentKnowledgeGraph
        profile = StudentProfile(student_id="student_graph_test", preferred_language="English")
        graph_db = self.output_dir / "test_graph.db"
        graph = StudentKnowledgeGraph(graph_db)
        client = StudentClient(
            server_url="ws://localhost:8765",
            session_id="lec_g",
            profile=profile,
            output_dir=self.output_dir,
            student_graph=graph
        )
        segment = GeneralNoteSegment(
            session_id="lec_g",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Graph Theory",
            content="Nodes and edges form a graph.",
            key_points=[],
            concept_tags=["graph_theory", "nodes"]
        )
        await client.on_note_segment_received(segment)
        summary = graph.get_student_summary()
        self.assertEqual(summary["concepts_tracked"], 2)
        names = [c["concept_name"] for c in summary["concepts"]]
        self.assertIn("graph_theory", names)
        self.assertIn("nodes", names)


if __name__ == "__main__":
    unittest.main()
