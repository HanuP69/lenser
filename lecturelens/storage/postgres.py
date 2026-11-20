"""PostgreSQL Storage Layer (DBMS Placement Standard).

Implements:
1. ACID Transactions via async context manager (with db.transaction(): ...).
2. 3NF Relational Schemas with Foreign Keys & Compound Unique Constraints.
3. B+ Tree Compound Indexing on (session_id, sequence_number) for fast replay range scans.
4. Clean connection management and parameterization.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import Any, Optional

from lecturelens.core.event_loop import ensure_event_loop_policy

ensure_event_loop_policy()

import psycopg
from psycopg.rows import dict_row

from lecturelens.schemas.events import GeneralNoteSegment, TimelineEntry

logger = logging.getLogger("lecturelens.storage.postgres")

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lectures (
    lecture_id TEXT PRIMARY KEY,
    course_id TEXT REFERENCES courses(course_id) ON DELETE CASCADE,
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
    definitions JSONB,
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
    UNIQUE(course_id, session_id, concept_name)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    source_type TEXT,
    source_id TEXT,
    chunk_text TEXT,
    embedding_model TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- B+ Tree Compound Indexes for Fast Range Lookups
CREATE INDEX IF NOT EXISTS idx_timeline_events_session ON timeline_events (session_id, timestamp_epoch);
CREATE INDEX IF NOT EXISTS idx_general_notes_session_seq ON general_notes (session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_concepts_course ON concepts (course_id);
"""

DEFAULT_COURSE_KEY = "_default_course"


class PostgresStorage:
    """Async PostgreSQL storage engine with ACID transactions & connection safety."""

    def __init__(self, connection_url: str, query_limit: int = 1000) -> None:
        self.connection_url = connection_url
        self.query_limit = query_limit
        self._connection: Optional[psycopg.AsyncConnection[dict_row]] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_connection(self) -> psycopg.AsyncConnection[dict_row]:
        if self._connection is None or self._connection.closed:
            conn_url = self.connection_url
            if "connect_timeout=" not in conn_url:
                separator = "&" if "?" in conn_url else "?"
                conn_url = f"{conn_url}{separator}connect_timeout=2"
            self._connection = await psycopg.AsyncConnection.connect(
                conn_url,
                row_factory=dict_row,
                autocommit=True
            )
        return self._connection

    @asynccontextmanager
    async def transaction(self):
        """ACID Transaction context manager: Auto-commit on success, auto-rollback on error."""
        async with self._lock:
            conn = await self.get_connection()
            async with conn.transaction():
                yield conn

    async def init_schema(self, vector_dimensions: int = 768) -> None:
        """Initialize 3NF relational schemas, indexes, and optional pgvector extension."""
        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as vector_error:
            logger.info("pgvector extension not installed or available: %s", vector_error)

        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(CREATE_TABLES_SQL)
                await cur.execute("ALTER TABLE general_notes ADD COLUMN IF NOT EXISTS definitions JSONB;")

        # Deduplicate embeddings and enforce unique index
        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT '';")
                    await cur.execute("""
                        DELETE FROM embeddings a USING embeddings b
                        WHERE a.id < b.id AND a.session_id = b.session_id AND a.source_id = b.source_id
                          AND NOT EXISTS (
                              SELECT 1 FROM pg_indexes
                              WHERE tablename = 'embeddings'
                                AND indexname = 'idx_embeddings_session_source'
                          );
                    """)
                    await cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_session_source ON embeddings (session_id, source_id);")
        except Exception as schema_err:
            logger.error("failed to initialize embeddings schema: %s", schema_err)
            raise RuntimeError(f"failed to initialize embeddings schema: {schema_err}") from schema_err

        # Setup vector column if extension exists
        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding vector({vector_dimensions});"
                    )
        except Exception:
            pass

    async def insert_timeline_event(self, entry: TimelineEntry | dict[str, Any]) -> bool:
        """Insert a timeline event (Audio chunk or Slide capture) idempotently."""
        d = entry.model_dump() if isinstance(entry, TimelineEntry) else dict(entry)
        meta = json.dumps(d.get("metadata")) if d.get("metadata") is not None else None
        flags = list(d.get("importance_flags") or [])

        query = """
        INSERT INTO timeline_events (
            event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
            correlation_id, content, metadata, importance_flags
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING;
        """
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (
                        str(d.get("event_id") or ""),
                        str(d.get("session_id") or ""),
                        str(d.get("source_type") or ""),
                        float(d.get("timestamp_epoch", 0.0)),
                        float(d.get("elapsed_seconds") or 0.0),
                        d.get("correlation_id"),
                        str(d.get("content", "")),
                        meta,
                        flags
                    )
                )
                return (cur.rowcount or 0) > 0

    async def insert_general_note(self, note: GeneralNoteSegment | dict[str, Any]) -> bool:
        """Insert or update a textbook note segment with monotonic sequence number."""
        d = note.model_dump() if isinstance(note, GeneralNoteSegment) else dict(note)
        query = """
        INSERT INTO general_notes (
            session_id, lecture_id, sequence_number, heading, content,
            key_points, concept_tags, slide_references, definitions,
            timestamp_epoch, start_time, end_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id, sequence_number) DO UPDATE SET
            lecture_id = EXCLUDED.lecture_id,
            heading = EXCLUDED.heading,
            content = EXCLUDED.content,
            key_points = EXCLUDED.key_points,
            concept_tags = EXCLUDED.concept_tags,
            slide_references = EXCLUDED.slide_references,
            definitions = EXCLUDED.definitions,
            timestamp_epoch = EXCLUDED.timestamp_epoch,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time;
        """
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (
                        str(d.get("session_id") or ""),
                        d.get("lecture_id"),
                        int(d.get("sequence_number", 0)),
                        str(d.get("heading") or ""),
                        str(d.get("content") or ""),
                        json.dumps(d.get("key_points") or []),
                        list(d.get("concept_tags") or []),
                        json.dumps(d.get("slide_references") or []),
                        json.dumps(d.get("definitions") or []),
                        float(d.get("timestamp_epoch", 0.0)),
                        str(d.get("start_time") or ""),
                        str(d.get("end_time") or "")
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
        """Upsert a discovered academic concept into the course graph."""
        course_key = course_id if course_id else DEFAULT_COURSE_KEY
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
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (course_key, concept_name, definition, related_concepts, session_id, timestamp_epoch)
                )
                return (cur.rowcount or 0) > 0

    async def get_timeline_events(
        self,
        session_id: str,
        limit: Optional[int] = None,
        descending: bool = False
    ) -> list[dict[str, Any]]:
        """Fetch timeline events utilizing B+ Tree index on (session_id, timestamp_epoch)."""
        order = "DESC" if descending else "ASC"
        fetch_limit = limit if limit is not None else self.query_limit
        query = f"""
        SELECT event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
               correlation_id, content, metadata, importance_flags, created_at
        FROM timeline_events
        WHERE session_id = %s
        ORDER BY timestamp_epoch {order}
        LIMIT %s;
        """
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id, fetch_limit))
                rows = await cur.fetchall()
                return list(reversed(rows)) if descending else rows

    async def get_general_notes_since(
        self,
        session_id: str,
        after_sequence_number: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch notes for replay using the compound B+ Tree index on (session_id, sequence_number)."""
        query = """
        SELECT session_id, lecture_id, sequence_number, heading, content,
               key_points, concept_tags, slide_references, timestamp_epoch,
               start_time, end_time, definitions
        FROM general_notes
        WHERE session_id = %s AND sequence_number > %s
        ORDER BY sequence_number ASC;
        """
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id, after_sequence_number))
                return await cur.fetchall()

    async def session_exists(self, session_id: str) -> bool:
        """Check if any events or notes exist for this session."""
        query = """
        SELECT 1 FROM timeline_events WHERE session_id = %s
        UNION ALL
        SELECT 1 FROM general_notes WHERE session_id = %s
        LIMIT 1;
        """
        async with self._lock:
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
        vector: Optional[list[float]] = None,
        embedding_model: str = ""
    ) -> bool:
        """Store chunk text and optional vector embeddings."""
        query = """
        INSERT INTO embeddings (session_id, source_type, source_id, chunk_text, embedding, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id, source_id) DO UPDATE SET
            source_type = EXCLUDED.source_type,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model;
        """
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (session_id, source_type, source_id, chunk_text, vector, embedding_model)
                )
                return True

    async def delete_embeddings_not_in(self, session_id: str, valid_source_ids: list[str]) -> int:
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                if valid_source_ids:
                    await cur.execute(
                        "DELETE FROM embeddings WHERE session_id = %s "
                        "AND (source_id IS NULL OR NOT (source_id = ANY(%s)));",
                        (session_id, valid_source_ids)
                    )
                else:
                    await cur.execute("DELETE FROM embeddings WHERE session_id = %s;", (session_id,))
                return cur.rowcount

    async def delete_embeddings(self, session_id: str, source_ids: list[str]) -> int:
        if not source_ids:
            return 0
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM embeddings WHERE session_id = %s AND source_id = ANY(%s);",
                    (session_id, source_ids)
                )
                return cur.rowcount or 0

    async def delete_session_embeddings(self, session_id: str, exclude_model: Optional[str] = None) -> int:
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                if exclude_model:
                    await cur.execute(
                        "DELETE FROM embeddings WHERE session_id = %s AND (embedding_model != %s OR embedding_model IS NULL);",
                        (session_id, exclude_model)
                    )
                else:
                    await cur.execute("DELETE FROM embeddings WHERE session_id = %s;", (session_id,))
                return cur.rowcount or 0

    async def search_embeddings(
        self,
        session_id: str,
        query_vector: list[float],
        top_k: int = 5,
        embedding_model: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if embedding_model:
            query = """
            SELECT chunk_text, source_id, source_type, (embedding <=> %s::vector) AS distance
            FROM embeddings
            WHERE session_id = %s AND embedding IS NOT NULL AND embedding_model = %s
            ORDER BY distance ASC LIMIT %s;
            """
            params = (query_vector, session_id, embedding_model, top_k)
        else:
            query = """
            SELECT chunk_text, source_id, source_type, (embedding <=> %s::vector) AS distance
            FROM embeddings
            WHERE session_id = %s AND embedding IS NOT NULL
            ORDER BY distance ASC LIMIT %s;
            """
            params = (query_vector, session_id, top_k)

        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                try:
                    await cur.execute(query, params)
                    return await cur.fetchall()
                except Exception as err:
                    logger.debug("vector search unavailable or fallback: %s", err)
                    return []

    async def get_concepts_for_course(self, course_id: str) -> list[dict[str, Any]]:
        course_key = course_id if course_id else DEFAULT_COURSE_KEY
        query = "SELECT * FROM concepts WHERE course_id = %s ORDER BY timestamp_epoch ASC;"
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (course_key,))
                return await cur.fetchall()

    async def get_all_concepts_for_session(self, session_id: str) -> list[dict[str, Any]]:
        query = "SELECT * FROM concepts WHERE session_id = %s ORDER BY timestamp_epoch ASC;"
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id,))
                return await cur.fetchall()

    async def close(self) -> None:
        """Safely close database connection."""
        async with self._lock:
            if self._connection is not None and not self._connection.closed:
                await self._connection.close()
                self._connection = None
