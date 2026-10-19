import asyncio
import os
import sys
import unittest
from typing import Any, Optional
from unittest import mock

from lecturelens.core.event_loop import ensure_event_loop_policy

ensure_event_loop_policy()

from lecturelens.config.settings import Settings
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import GeneralNoteSegment, TimelineEntry
from lecturelens.storage.persister import PostgresPersister
from lecturelens.storage.postgres import PostgresStorage


class FakeStorage:
    def __init__(self) -> None:
        self.timeline_events: list[dict[str, Any]] = []
        self.general_notes: list[dict[str, Any]] = []
        self.concepts: list[dict[str, Any]] = []
        self.fail_on_insert = False

    async def insert_timeline_event(self, entry: TimelineEntry | dict[str, Any]) -> bool:
        if self.fail_on_insert:
            raise RuntimeError("Database connection lost")
        if isinstance(entry, TimelineEntry):
            data = entry.model_dump()
        else:
            data = dict(entry)
        for existing in self.timeline_events:
            if existing.get("event_id") == data.get("event_id"):
                return False
        self.timeline_events.append(data)
        return True

    async def insert_general_note(self, note: GeneralNoteSegment | dict[str, Any]) -> bool:
        if self.fail_on_insert:
            raise RuntimeError("Database connection lost")
        if isinstance(note, GeneralNoteSegment):
            data = note.model_dump()
        else:
            data = dict(note)
        for idx, existing in enumerate(self.general_notes):
            if (
                existing.get("session_id") == data.get("session_id")
                and existing.get("sequence_number") == data.get("sequence_number")
            ):
                self.general_notes[idx] = data
                return True
        self.general_notes.append(data)
        return True

    async def insert_concept(
        self,
        course_id: Optional[str],
        concept_name: str,
        definition: Optional[str],
        related_concepts: list[str],
        session_id: str,
        timestamp_epoch: float
    ) -> bool:
        if self.fail_on_insert:
            raise RuntimeError("Database connection lost")
        data = {
            "course_id": course_id,
            "concept_name": concept_name,
            "definition": definition,
            "related_concepts": related_concepts,
            "session_id": session_id,
            "timestamp_epoch": timestamp_epoch
        }
        for idx, existing in enumerate(self.concepts):
            if (
                existing.get("course_id") == data.get("course_id")
                and existing.get("concept_name") == data.get("concept_name")
            ):
                self.concepts[idx] = data
                return True
        self.concepts.append(data)
        return True


def check_live_postgres() -> bool:
    test_url = os.environ.get("POSTGRES_TEST_URL")
    if not test_url:
        return False
    try:
        import psycopg
        with psycopg.connect(test_url, timeout=1):
            return True
    except Exception:
        return False


LIVE_PG_AVAILABLE = check_live_postgres()


class TestPersisterLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.storage = FakeStorage()
        self.settings = Settings()
        self.settings.course_name = "cs101"
        self.redis_bus = mock.AsyncMock(spec=RedisBus)
        self.persister = PostgresPersister(self.settings, self.redis_bus, self.storage)

    async def test_handle_timeline_message_stores_and_acknowledges(self) -> None:
        entry = TimelineEntry(
            session_id="sess_1",
            event_id="evt_101",
            timestamp_epoch=1700000000.0,
            elapsed_seconds=10.0,
            source_type="transcript",
            content="Binary trees have left and right children.",
            importance_flags=["definition"]
        )
        await self.persister.handle_timeline_message(
            entry.model_dump(),
            "1-0",
            "lecturelens:sess_1:stream:timeline"
        )
        self.assertEqual(len(self.storage.timeline_events), 1)
        self.assertEqual(self.storage.timeline_events[0]["event_id"], "evt_101")
        self.redis_bus.acknowledge.assert_awaited_once_with(
            "lecturelens:sess_1:stream:timeline",
            "group:persister",
            "1-0"
        )

    async def test_handle_timeline_message_idempotent_duplicate_is_safe(self) -> None:
        entry = TimelineEntry(
            session_id="sess_1",
            event_id="evt_102",
            timestamp_epoch=1700000005.0,
            elapsed_seconds=15.0,
            source_type="slide",
            content="Slide 1 Content"
        )
        await self.persister.handle_timeline_message(entry.model_dump(), "2-0", "stream:t")
        await self.persister.handle_timeline_message(entry.model_dump(), "2-0", "stream:t")
        self.assertEqual(len(self.storage.timeline_events), 1)
        self.assertEqual(self.redis_bus.acknowledge.await_count, 2)

    async def test_handle_note_message_stores_segment_and_extracts_concepts(self) -> None:
        segment = GeneralNoteSegment(
            session_id="sess_1",
            sequence_number=1,
            timestamp_epoch=1700000030.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Binary Search Trees",
            content="Properties of BSTs explained.",
            key_points=["Left subtree has keys less than root."],
            concept_tags=["binary_tree", "bst"]
        )
        await self.persister.handle_note_message(
            segment.model_dump(),
            "3-0",
            "lecturelens:sess_1:stream:notes"
        )
        self.assertEqual(len(self.storage.general_notes), 1)
        self.assertEqual(self.storage.general_notes[0]["sequence_number"], 1)
        self.assertEqual(len(self.storage.concepts), 2)
        concept_names = [c["concept_name"] for c in self.storage.concepts]
        self.assertIn("binary_tree", concept_names)
        self.assertIn("bst", concept_names)
        self.redis_bus.acknowledge.assert_awaited_once_with(
            "lecturelens:sess_1:stream:notes",
            "group:persister",
            "3-0"
        )

    async def test_crash_before_ack_leaves_ack_uncalled(self) -> None:
        self.storage.fail_on_insert = True
        entry = TimelineEntry(
            session_id="sess_1",
            event_id="evt_crash",
            timestamp_epoch=1700000010.0,
            elapsed_seconds=20.0,
            source_type="transcript",
            content="Crash test"
        )
        with self.assertRaises(RuntimeError):
            await self.persister.handle_timeline_message(
                entry.model_dump(),
                "4-0",
                "lecturelens:sess_1:stream:timeline"
            )
        self.redis_bus.acknowledge.assert_not_called()


class TestPostgresStorageSyntax(unittest.TestCase):
    def test_schema_ddl_contains_core_tables(self) -> None:
        from lecturelens.storage.postgres import CREATE_TABLES_SQL
        for table_name in ("courses", "lectures", "students", "timeline_events", "general_notes", "concepts", "embeddings"):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table_name}", CREATE_TABLES_SQL)

    def test_indexes_exist_in_ddl(self) -> None:
        from lecturelens.storage.postgres import CREATE_TABLES_SQL
        self.assertIn("idx_timeline_events_session", CREATE_TABLES_SQL)
        self.assertIn("idx_general_notes_session_seq", CREATE_TABLES_SQL)
        self.assertIn("idx_concepts_course", CREATE_TABLES_SQL)

    def test_psycopg_is_importable(self) -> None:
        import psycopg
        self.assertIsNotNone(psycopg)

    def test_get_general_notes_query_omits_unserializable_datetime_columns(self) -> None:
        import inspect
        from lecturelens.storage.postgres import PostgresStorage
        source = inspect.getsource(PostgresStorage.get_general_notes_since)
        self.assertNotIn("created_at", source)
        self.assertIn("sequence_number", source)
        self.assertIn("heading", source)
        self.assertIn("content", source)

    def test_insert_concept_query_merges_definitions_and_related_tags(self) -> None:
        import inspect
        from lecturelens.storage.postgres import PostgresStorage
        source = inspect.getsource(PostgresStorage.insert_concept)
        self.assertIn("COALESCE", source)
        self.assertIn("unnest", source)

    def test_persister_main_inspects_parse_runtime_arguments_and_session_dir(self) -> None:
        import inspect
        from lecturelens.storage import persister
        source = inspect.getsource(persister.main)
        self.assertIn("parse_runtime_arguments", source)
        self.assertIn("get_session_output_dir", source)
        self.assertIn("stop_requested", source)


@unittest.skipUnless(LIVE_PG_AVAILABLE, "Live Postgres not configured via POSTGRES_TEST_URL")
class TestPostgresLiveIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        test_url = os.environ.get("POSTGRES_TEST_URL", "")
        self.storage = PostgresStorage(test_url)
        await self.storage.init_schema()

    async def asyncTearDown(self) -> None:
        await self.storage.close()

    async def test_round_trip_timeline_and_notes(self) -> None:
        session_id = "test_live_session"
        entry = TimelineEntry(
            session_id=session_id,
            event_id="live_evt_1",
            timestamp_epoch=1700000000.0,
            elapsed_seconds=5.0,
            source_type="transcript",
            content="Live Postgres test content",
            importance_flags=["definition"]
        )
        inserted = await self.storage.insert_timeline_event(entry)
        self.assertTrue(inserted)
        duplicate = await self.storage.insert_timeline_event(entry)
        self.assertFalse(duplicate)

        events = await self.storage.get_timeline_events(session_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], "live_evt_1")

    async def test_round_trip_general_notes_and_json_serialization(self) -> None:
        import json
        session_id = "test_live_notes_sess"
        note = GeneralNoteSegment(
            session_id=session_id,
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Section 1",
            content="Live Postgres Note Content",
            key_points=["Point A", "Point B"],
            concept_tags=["tag1", "tag2"]
        )
        inserted = await self.storage.insert_general_note(note)
        self.assertTrue(inserted)

        notes = await self.storage.get_general_notes_since(session_id, after_sequence_number=0)
        self.assertEqual(len(notes), 1)
        serialized = json.dumps({"type": "note_segment", "payload": notes[0]})
        self.assertIn("Live Postgres Note Content", serialized)

    async def test_concept_merge_in_live_postgres(self) -> None:
        session_id = "test_live_concept_sess"
        course_id = "test_course_live"
        await self.storage.insert_concept(
            course_id=course_id,
            concept_name="binary_search",
            definition="Search algorithm on sorted array",
            related_concepts=["array", "sorting"],
            session_id=session_id,
            timestamp_epoch=1700000000.0
        )
        await self.storage.insert_concept(
            course_id=course_id,
            concept_name="binary_search",
            definition=None,
            related_concepts=["divide_and_conquer"],
            session_id=session_id,
            timestamp_epoch=1700000050.0
        )
        session_id_2 = f"test_session_2_{uuid.uuid4().hex[:6]}"
        await self.storage.insert_concept(
            course_id=course_id,
            concept_name="binary_search",
            definition=None,
            related_concepts=["logarithmic_time"],
            session_id=session_id_2,
            timestamp_epoch=1700000100.0
        )
        concepts = await self.storage.get_concepts_for_course(course_id)
        matching = [c for c in concepts if c["concept_name"] == "binary_search"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["definition"], "Search algorithm on sorted array")
        self.assertIn("sorting", matching[0]["related_concepts"])
        self.assertIn("divide_and_conquer", matching[0]["related_concepts"])
        self.assertIn("logarithmic_time", matching[0]["related_concepts"])


if __name__ == "__main__":
    unittest.main()
