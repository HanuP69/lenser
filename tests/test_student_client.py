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


if __name__ == "__main__":
    unittest.main()
