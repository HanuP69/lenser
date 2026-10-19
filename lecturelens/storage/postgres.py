import asyncio
import json
import logging
import sys
from typing import Any, Optional

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import psycopg
from psycopg.rows import dict_row

from lecturelens.schemas.events import GeneralNoteSegment, TimelineEntry

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lectures (
    lecture_id TEXT PRIMARY KEY,
    course_id TEXT,
    title TEXT,
    scheduled_at TIMESTAMPTZ,
    session_id TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    profile_json JSONB
);

CREATE TABLE IF NOT EXISTS timeline_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    timestamp_epoch DOUBLE PRECISION NOT NULL,
    elapsed_seconds DOUBLE PRECISION NOT NULL,
    correlation_id TEXT,
    content TEXT,
    metadata JSONB,
    importance_flags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS general_notes (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    lecture_id TEXT,
    sequence_number INT NOT NULL,
    heading TEXT,
    content TEXT,
    key_points JSONB,
    concept_tags TEXT[],
    slide_references JSONB,
    timestamp_epoch DOUBLE PRECISION,
    start_time TEXT,
    end_time TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id SERIAL PRIMARY KEY,
    course_id TEXT,
    concept_name TEXT,
    definition TEXT,
    related_concepts TEXT[],
    session_id TEXT,
    timestamp_epoch DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT concepts_course_id_session_id_concept_name_key UNIQUE(course_id, session_id, concept_name)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    source_type TEXT,
    source_id TEXT,
    chunk_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_timeline_events_session ON timeline_events (session_id, timestamp_epoch);
CREATE INDEX IF NOT EXISTS idx_general_notes_session_seq ON general_notes (session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_concepts_course ON concepts (course_id);
"""

MIGRATE_CONCEPTS_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_name = 'concepts'
    ) THEN
        ALTER TABLE concepts DROP CONSTRAINT IF EXISTS concepts_course_id_concept_name_key;
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'concepts_course_id_session_id_concept_name_key'
        ) THEN
            ALTER TABLE concepts ADD CONSTRAINT concepts_course_id_session_id_concept_name_key
            UNIQUE (course_id, session_id, concept_name);
        END IF;
    END IF;
END $$;
"""

CREATE_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"
ADD_VECTOR_COLUMN_SQL = "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding vector(768);"


class PostgresStorage:
    def __init__(self, connection_url: str) -> None:
        self.connection_url = connection_url
        self._connection: Optional[psycopg.AsyncConnection[dict_row]] = None

    async def get_connection(self) -> psycopg.AsyncConnection[dict_row]:
        if self._connection is None or self._connection.closed:
            self._connection = await psycopg.AsyncConnection.connect(
                self.connection_url,
                row_factory=dict_row,
                autocommit=False
            )
        return self._connection

    async def init_schema(self) -> None:
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(CREATE_VECTOR_EXTENSION_SQL)
        except Exception as vector_error:
            logger.info("pgvector extension not installed or available: %s", vector_error)

        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(CREATE_TABLES_SQL)
                await cur.execute(MIGRATE_CONCEPTS_SQL)

        try:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(ADD_VECTOR_COLUMN_SQL)
        except Exception as vector_col_error:
            logger.debug("vector column skipped: %s", vector_col_error)

    async def insert_timeline_event(self, entry: TimelineEntry | dict[str, Any]) -> bool:
        if isinstance(entry, TimelineEntry):
            event_id = entry.event_id
            session_id = entry.session_id
            source_type = entry.source_type
            timestamp_epoch = entry.timestamp_epoch
            raw_elapsed = getattr(entry, "elapsed_seconds", None)
            elapsed_seconds = float(raw_elapsed) if raw_elapsed is not None else 0.0
            correlation_id = entry.correlation_id
            content = entry.content
            metadata_val = json.dumps(entry.metadata) if entry.metadata is not None else None
            flags = entry.importance_flags
        else:
            event_id = str(entry.get("event_id") or "")
            session_id = str(entry.get("session_id") or "")
            source_type = str(entry.get("source_type") or "")
            timestamp_epoch = float(entry.get("timestamp_epoch", 0.0))
            raw_elapsed = entry.get("elapsed_seconds")
            elapsed_seconds = float(raw_elapsed) if raw_elapsed is not None else 0.0
            correlation_id = entry.get("correlation_id")
            content = str(entry.get("content", ""))
            meta_raw = entry.get("metadata")
            metadata_val = json.dumps(meta_raw) if meta_raw is not None else None
            flags = list(entry.get("importance_flags") or [])

        query = """
        INSERT INTO timeline_events (
            event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
            correlation_id, content, metadata, importance_flags
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING;
        """
        conn = await self.get_connection()
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (
                        event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
                        correlation_id, content, metadata_val, flags
                    )
                )
                return (cur.rowcount or 0) > 0

    async def insert_general_note(self, note: GeneralNoteSegment | dict[str, Any]) -> bool:
        if isinstance(note, GeneralNoteSegment):
            session_id = note.session_id
            lecture_id = note.lecture_id
            sequence_number = note.sequence_number
            heading = note.heading
            content = note.content
            key_points_val = json.dumps(note.key_points)
            concept_tags = note.concept_tags
            slide_refs_val = json.dumps(note.slide_references)
            timestamp_epoch = note.timestamp_epoch
            start_time = note.start_time
            end_time = note.end_time
        else:
            session_id = str(note.get("session_id") or "")
            lecture_id = note.get("lecture_id")
            sequence_number = int(note.get("sequence_number", 0))
            heading = str(note.get("heading") or "")
            content = str(note.get("content") or "")
            key_points_val = json.dumps(note.get("key_points") or [])
            concept_tags = list(note.get("concept_tags") or [])
            slide_refs_val = json.dumps(note.get("slide_references") or [])
            timestamp_epoch = float(note.get("timestamp_epoch", 0.0))
            start_time = str(note.get("start_time") or "")
            end_time = str(note.get("end_time") or "")

        query = """
        INSERT INTO general_notes (
            session_id, lecture_id, sequence_number, heading, content,
            key_points, concept_tags, slide_references, timestamp_epoch,
            start_time, end_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id, sequence_number) DO UPDATE SET
            heading = EXCLUDED.heading,
            content = EXCLUDED.content,
            key_points = EXCLUDED.key_points,
            concept_tags = EXCLUDED.concept_tags,
            slide_references = EXCLUDED.slide_references,
            timestamp_epoch = EXCLUDED.timestamp_epoch,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time;
        """
        conn = await self.get_connection()
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (
                        session_id, lecture_id, sequence_number, heading, content,
                        key_points_val, concept_tags, slide_refs_val, timestamp_epoch,
                        start_time, end_time
                    )
                )
                return (cur.rowcount or 0) > 0

    async def insert_concept(
        self,
        course_id: Optional[str],
        concept_name: str,
        definition: Optional[str],
        related_concepts: list[str],
        session_id: str,
        timestamp_epoch: float
    ) -> bool:
        course_key = course_id if course_id else "_default_course"
        query = """
        INSERT INTO concepts (
            course_id, concept_name, definition, related_concepts, session_id, timestamp_epoch
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_id, session_id, concept_name) DO UPDATE SET
            definition = COALESCE(NULLIF(EXCLUDED.definition, ''), concepts.definition),
            related_concepts = ARRAY(
                SELECT DISTINCT unnest(COALESCE(concepts.related_concepts, '{}'::text[]) || COALESCE(EXCLUDED.related_concepts, '{}'::text[]))
            ),
            timestamp_epoch = EXCLUDED.timestamp_epoch;
        """
        conn = await self.get_connection()
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (course_key, concept_name, definition, related_concepts, session_id, timestamp_epoch)
                )
                return (cur.rowcount or 0) > 0

    async def get_timeline_events(
        self,
        session_id: str,
        limit: int = 1000
    ) -> list[dict[str, Any]]:
        query = """
        SELECT event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
               correlation_id, content, metadata, importance_flags, created_at
        FROM timeline_events
        WHERE session_id = %s
        ORDER BY timestamp_epoch ASC
        LIMIT %s;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            await cur.execute(query, (session_id, limit))
            return await cur.fetchall()

    async def get_general_notes_since(
        self,
        session_id: str,
        after_sequence_number: int = 0
    ) -> list[dict[str, Any]]:
        query = """
        SELECT session_id, lecture_id, sequence_number, heading, content,
               key_points, concept_tags, slide_references, timestamp_epoch,
               start_time, end_time
        FROM general_notes
        WHERE session_id = %s AND sequence_number > %s
        ORDER BY sequence_number ASC;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            await cur.execute(query, (session_id, after_sequence_number))
            return await cur.fetchall()

    async def session_exists(self, session_id: str) -> bool:
        query = """
        SELECT 1 FROM timeline_events WHERE session_id = %s
        UNION ALL
        SELECT 1 FROM general_notes WHERE session_id = %s
        LIMIT 1;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            await cur.execute(query, (session_id, session_id))
            return (await cur.fetchone()) is not None

    async def insert_embedding(
        self,
        session_id: str,
        source_type: str,
        source_id: str,
        chunk_text: str,
        vector: Optional[list[float]] = None
    ) -> bool:
        conn = await self.get_connection()
        async with conn.transaction():
            async with conn.cursor() as cur:
                if vector is not None:
                    query = """
                    INSERT INTO embeddings (session_id, source_type, source_id, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector);
                    """
                    await cur.execute(query, (session_id, source_type, source_id, chunk_text, vector))
                else:
                    query = """
                    INSERT INTO embeddings (session_id, source_type, source_id, chunk_text)
                    VALUES (%s, %s, %s, %s);
                    """
                    await cur.execute(query, (session_id, source_type, source_id, chunk_text))
                return True

    async def search_embeddings(
        self,
        session_id: str,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        query = """
        SELECT chunk_text, source_id, source_type, (embedding <=> %s::vector) AS distance
        FROM embeddings
        WHERE session_id = %s AND embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT %s;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            try:
                await cur.execute(query, (query_vector, session_id, top_k))
                return await cur.fetchall()
            except Exception as err:
                await conn.rollback()
                logger.warning("vector search unavailable in postgres: %s", err)
                return []

    async def get_concepts_for_course(self, course_id: str) -> list[dict[str, Any]]:
        query = """
        SELECT concept_id, course_id, concept_name, definition, related_concepts,
               session_id, timestamp_epoch, created_at
        FROM concepts
        WHERE course_id = %s
        ORDER BY timestamp_epoch ASC;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            await cur.execute(query, (course_id,))
            rows = await cur.fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = str(row["concept_name"])
            if name not in grouped:
                grouped[name] = dict(row)
                grouped[name]["related_concepts"] = list(row.get("related_concepts") or [])
            else:
                existing = grouped[name]
                if row.get("definition"):
                    existing["definition"] = row["definition"]
                existing_related = set(existing.get("related_concepts") or [])
                existing_related.update(row.get("related_concepts") or [])
                existing["related_concepts"] = sorted(list(existing_related))
                existing["timestamp_epoch"] = max(float(existing.get("timestamp_epoch") or 0.0), float(row.get("timestamp_epoch") or 0.0))
                existing["session_id"] = row.get("session_id") or existing.get("session_id")
        return sorted(list(grouped.values()), key=lambda x: str(x["concept_name"]))

    async def get_all_concepts_for_session(self, session_id: str) -> list[dict[str, Any]]:
        query = """
        SELECT concept_id, course_id, concept_name, definition, related_concepts,
               session_id, timestamp_epoch, created_at
        FROM concepts
        WHERE session_id = %s
        ORDER BY timestamp_epoch ASC;
        """
        conn = await self.get_connection()
        async with conn.cursor() as cur:
            await cur.execute(query, (session_id,))
            return await cur.fetchall()

    async def close(self) -> None:
        if self._connection is not None and not self._connection.closed:
            await self._connection.close()
            self._connection = None
