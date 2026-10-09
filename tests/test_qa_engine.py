import json
import tempfile
import unittest
from pathlib import Path
from lecturelens.knowledge.qa_engine import QAEngine


class DummyLLMRouter:
    async def ainvoke(self, messages: list) -> str:
        return "QuickSort has an average time complexity of O(n log n)."


class TestQAEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)

        timeline_path = self.session_dir / "timeline.jsonl"
        with open(timeline_path, "w", encoding="utf-8") as file_handle:
            entry_1 = {
                "session_id": "test_qa",
                "timestamp_epoch": 100.0,
                "source_type": "transcript",
                "content": "QuickSort is a divide and conquer algorithm with average complexity O(n log n)."
            }
            file_handle.write(json.dumps(entry_1) + "\n")

        notes_path = self.session_dir / "notes.md"
        with open(notes_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("# Sorting Algorithms\n\nQuickSort partitions around a pivot element.\n")

        self.router = DummyLLMRouter()
        self.qa_engine = QAEngine(
            session_dir=self.session_dir,
            ollama_url="http://localhost:11434",
            llm_router=self.router
        )

        async def mock_embedding(text: str) -> list[float]:
            import hashlib
            hasher = hashlib.md5(text.encode("utf-8")).digest()
            vector = [float(b) / 255.0 for b in hasher]
            return vector

        self.qa_engine.generate_embedding = mock_embedding

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    async def test_build_index_and_ask(self) -> None:
        count = await self.qa_engine.build_index()
        self.assertGreaterEqual(count, 2)

        answer = await self.qa_engine.ask("What is the complexity of QuickSort?")
        self.assertIn("QuickSort", answer)


if __name__ == "__main__":
    unittest.main()
