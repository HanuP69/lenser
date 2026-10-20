import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage

from lecturelens.knowledge.qa_engine import QAEngine


class DummyLLMRouter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages: list) -> AIMessage:
        self.prompts.append(str(messages[0].content))
        return AIMessage(content="QuickSort has an average time complexity of O(n log n). [1]")


def fake_vector(text: str) -> list[float]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    vector: list[float] = []
    for byte in digest:
        vector.append(float(byte) / 255.0)
    return vector


class TestQAEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)
        self.embedded_texts: list[str] = []

        entry = {
            "session_id": "test_qa",
            "timestamp_epoch": 100.0,
            "source_type": "transcript",
            "content": "QuickSort is a divide and conquer algorithm with average complexity O(n log n)."
        }
        (self.session_dir / "timeline.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        (self.session_dir / "notes.md").write_text(
            "# Sorting Algorithms\n\nQuickSort partitions around a pivot element.\n", encoding="utf-8"
        )
        self.router = DummyLLMRouter()
        self.engine = QAEngine(self.session_dir, "http://localhost:11434", self.router)

        async def mock_embeddings(texts: list[str]) -> list[list[float]]:
            self.embedded_texts.extend(texts)
            vectors: list[list[float]] = []
            for text in texts:
                vectors.append(fake_vector(text))
            return vectors

        self.engine.generate_embeddings = mock_embeddings

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    async def test_build_index_and_ask_returns_sources(self) -> None:
        count = await self.engine.build_index()
        self.assertGreaterEqual(count, 2)
        answer = await self.engine.ask("What is the complexity of QuickSort?")
        self.assertIn("QuickSort", answer)
        self.assertIn("Sources:", answer)
        self.assertIn("notes.md", answer)

    async def test_ask_with_sources_lists_timeline_offsets(self) -> None:
        answer, references = await self.engine.ask_with_sources("QuickSort complexity")
        timeline_references = 0
        for reference in references:
            if reference.startswith("timeline @"):
                timeline_references += 1
        self.assertGreaterEqual(timeline_references, 1)
        self.assertIn("EXCERPTS", self.router.prompts[0])

    async def test_second_index_does_not_reembed_unchanged_content(self) -> None:
        await self.engine.build_index()
        first_count = len(self.embedded_texts)
        await self.engine.build_index()
        self.assertEqual(len(self.embedded_texts), first_count)

    async def test_new_timeline_lines_are_embedded_incrementally(self) -> None:
        await self.engine.build_index()
        first_count = len(self.embedded_texts)
        extra = {"timestamp_epoch": 130.0, "source_type": "transcript", "content": "Mergesort is stable."}
        with open(self.session_dir / "timeline.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(extra) + "\n")
        await self.engine.build_index()
        self.assertEqual(len(self.embedded_texts), first_count + 1)

    async def test_changed_notes_replace_stale_chunks(self) -> None:
        await self.engine.build_index()
        count_before = self.engine.collection.count()
        (self.session_dir / "notes.md").write_text("# Heaps\n\nA heap is a tree.\n", encoding="utf-8")
        await self.engine.build_index()
        self.assertEqual(self.engine.collection.count(), count_before)
        stored = self.engine.collection.get(include=[])["ids"]
        self.assertEqual(len(stored), len(set(stored)))

    async def test_empty_session_gives_clear_message(self) -> None:
        empty_dir = Path(self.temp_dir.name) / "empty"
        empty_dir.mkdir()
        engine = QAEngine(empty_dir, "http://localhost:11434", self.router)
        answer = await engine.ask("anything")
        self.assertIn("No indexed lecture content", answer)

    async def test_empty_session_retrieve_does_not_call_embeddings(self) -> None:
        empty_dir = Path(self.temp_dir.name) / "empty_retrieve"
        empty_dir.mkdir()
        engine = QAEngine(empty_dir, "http://localhost:11434", self.router)
        called = False

        async def fail_embeddings(texts: list[str]) -> list[list[float]]:
            nonlocal called
            called = True
            raise AssertionError("generate_embeddings should not be called when empty")

        engine.generate_embeddings = fail_embeddings
        results = await engine.retrieve("anything")
        self.assertEqual(results, [])
        self.assertFalse(called)

    async def test_embedding_model_change_recreates_collection(self) -> None:
        await self.engine.build_index()
        self.assertGreater(self.engine.collection.count(), 0)
        engine2 = QAEngine(
            self.session_dir,
            "http://localhost:11434",
            self.router,
            embedding_model="new_custom_embedder"
        )
        self.assertEqual(engine2.collection.count(), 0)

    async def test_missing_postgres_chunks_retried_from_chroma(self) -> None:
        await self.engine.build_index()
        self.assertGreater(self.engine.collection.count(), 0)
        inserted_records = []
        mock_storage = mock.MagicMock()
        mock_storage._lock = asyncio.Lock()
        fake_cursor = mock.AsyncMock()
        fake_cursor.fetchall = mock.AsyncMock(return_value=[])
        fake_cursor.__aenter__ = mock.AsyncMock(return_value=fake_cursor)
        fake_cursor.__aexit__ = mock.AsyncMock(return_value=None)
        fake_conn = mock.MagicMock()
        fake_conn.cursor = mock.MagicMock(return_value=fake_cursor)
        mock_storage.get_connection = mock.AsyncMock(return_value=fake_conn)

        async def record_insert(session_id, source_type, source_id, chunk_text, vector, **kwargs):
            inserted_records.append((source_id, chunk_text, vector))
            return True

        mock_storage.insert_embedding = mock.AsyncMock(side_effect=record_insert)
        self.engine.storage = mock_storage
        self.embedded_texts.clear()
        await self.engine.build_index()
        self.assertEqual(len(self.embedded_texts), 0)
        self.assertGreater(len(inserted_records), 0)

    async def test_stale_chunks_deleted_from_postgres_on_build_index(self) -> None:
        await self.engine.build_index()
        deleted_from_pg = []
        mock_storage = mock.MagicMock()
        mock_storage.delete_embeddings = mock.AsyncMock(side_effect=lambda sess, ids: deleted_from_pg.extend(ids))
        mock_storage.delete_session_embeddings = mock.AsyncMock(return_value=0)
        self.engine.storage = mock_storage

        (self.session_dir / "notes.md").write_text("# New Title\n\nEntirely new notes content.\n", encoding="utf-8")
        await self.engine.build_index()
        self.assertGreater(len(deleted_from_pg), 0)


if __name__ == "__main__":
    unittest.main()
