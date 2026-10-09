import json
import tempfile
import unittest
from pathlib import Path
from lecturelens.agents.structuring_agent import (
    StructuringAgent,
    create_structuring_graph,
    StructuringState
)
from lecturelens.config.settings import Settings


class MockRouter:
    async def ainvoke(self, messages: list) -> str:
        text = str(messages[0].content) if hasattr(messages[0], "content") else str(messages[0])
        if "key technical terms" in text:
            return json.dumps([{"term": "Binary Search", "definition": "A search algorithm that runs in logarithmic time."}])
        if "action items" in text:
            return json.dumps(["Complete homework 3 by Friday."])
        if "study flashcards" in text:
            return json.dumps([{"question": "What is binary search?", "answer": "Logarithmic search."}])
        if "Compress the following" in text:
            return "Binary search divides the search space in half."
        return "## Binary Search\nBinary search is an efficient algorithm on sorted arrays."


class TestStructuringAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)

        self.session_id = "test_struct_session"
        session_dir = self.settings.create_session_directories(self.session_id)
        self.timeline_path = session_dir / "timeline.jsonl"

        with open(self.timeline_path, "w", encoding="utf-8") as file_handle:
            entry_1 = {
                "session_id": self.session_id,
                "timestamp_epoch": 100.0,
                "source_type": "transcript",
                "content": "Today we cover binary search.",
                "correlation_id": "c1",
                "importance_flags": ["exam_relevant"]
            }
            entry_2 = {
                "session_id": self.session_id,
                "timestamp_epoch": 101.0,
                "source_type": "slide",
                "content": "Binary Search: O(log n)",
                "correlation_id": "c1",
                "importance_flags": []
            }
            file_handle.write(json.dumps(entry_1) + "\n")
            file_handle.write(json.dumps(entry_2) + "\n")

        self.router = MockRouter()
        self.agent = StructuringAgent(
            settings=self.settings,
            session_id=self.session_id,
            llm_router=self.router
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_process_batch(self) -> None:
        await self.agent.process_batch()

        notes_file = self.settings.get_session_output_dir(self.session_id) / "notes.md"
        self.assertTrue(notes_file.exists())

        with open(notes_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Binary Search", content)
        self.assertIn("Important Exam Topic", content)
        self.assertIn("Complete homework 3", content)

        concepts = self.agent.concept_graph.get_all_concepts_for_session(self.session_id)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["concept_name"], "binary search")

        flashcards_file = self.settings.get_session_output_dir(self.session_id) / "flashcards.apkg"
        self.assertTrue(flashcards_file.exists())


if __name__ == "__main__":
    unittest.main()
