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
    CONSTRAINT concepts_course_id_session_id_concept_name_key UNIQUE(course_id, session_id, concept_name)
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

# The sentinel used for concepts that were captured without a course. The
# write and the read path have to agree on it.
DEFAULT_COURSE_KEY = "_default_course"


class PostgresStorage:
    def __init__(self, connection_url: str, query_limit: int = 1000) -> None:
        self.connection_url = connection_url
        self.query_limit = query_limit
        self._connection: Optional[psycopg.AsyncConnection[dict_row]] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_connection(self) -> psycopg.AsyncConnection[dict_row]:
        if self._connection is None or self._connection.closed:
            self._connection = await psycopg.AsyncConnection.connect(
                self.connection_url,
                row_factory=dict_row,
                autocommit=True
            )
        return self._connection

    @asynccontextmanager
    async def transaction(self):
        async with self._lock:
            conn = await self.get_connection()
            async with conn.transaction():
                yield conn

    async def init_schema(self, vector_dimensions: int = 768) -> None:
        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(CREATE_VECTOR_EXTENSION_SQL)
        except Exception as vector_error:
            logger.info("pgvector extension not installed or available: %s", vector_error)

        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(CREATE_TABLES_SQL)
                await cur.execute(MIGRATE_CONCEPTS_SQL)
                # The definitions extracted by the structuring agent had no
                # column, so they were silently dropped on persist.
                await cur.execute(
                    "ALTER TABLE general_notes ADD COLUMN IF NOT EXISTS definitions JSONB;"
                )

        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT '';")
                    # The duplicate cleanup only makes sense before the
                    # unique index exists. Running the self join on every
                    # start (and on every embedding batch) was pure cost.
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

        try:
            async with self.transaction() as conn:
                async with conn.cursor() as cur:
                    current_dim: Optional[int] = None
                    try:
                        await cur.execute("""
                            SELECT atttypmod FROM pg_attribute
                            WHERE attrelid = 'embeddings'::regclass AND attname = 'embedding';
                        """)
                        row = await cur.fetchone()
                        if row and row.get("atttypmod") is not None and row["atttypmod"] > 0:
                            current_dim = int(row["atttypmod"])
                    except Exception as inspect_error:
                        # The column cannot be inspected, so assume it still
                        # has to be created.
                        logger.debug("could not inspect the embeddings vector column: %s", inspect_error)
                    if current_dim is not None:
                        if current_dim != vector_dimensions:
                            logger.warning(
                                "vector dimension changed from %d to %d; altering column",
                                current_dim, vector_dimensions
                            )
                            await cur.execute(f"ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector({vector_dimensions});")
                    else:
                        sql = f"ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding vector({vector_dimensions});"
                        await cur.execute(sql)
        except Exception as vector_col_error:
            # A failed dimension migration leaves the column at the old
            # width, so every later insert fails with a dimension mismatch.
            # This was logged at debug level and went unnoticed.
            logger.error("could not align the embeddings vector column: %s", vector_col_error)
            raise RuntimeError(
                f"could not align the embeddings vector column: {vector_col_error}"
            ) from vector_col_error

    async def delete_embeddings_not_in(self, session_id: str, valid_source_ids: list[str]) -> int:
        """Remove embeddings for a session that are not in the valid set.

        An empty ``valid_source_ids`` means "keep nothing for this session", so
        every embedding row belonging to the session is removed. Rows with a
        NULL ``source_id`` are removed as well, because ``NULL = ANY(...)``
        evaluates to NULL and would otherwise survive the filter.
        """
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                if valid_source_ids:
                    await cur.execute(
                        "DELETE FROM embeddings WHERE session_id = %s "
                        "AND (source_id IS NULL OR NOT (source_id = ANY(%s)));",
                        (session_id, valid_source_ids)
                    )
                else:
                    await cur.execute(
                        "DELETE FROM embeddings WHERE session_id = %s;",
                        (session_id,)
                    )
                return cur.rowcount

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
        async with self.transaction() as conn:
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
            definitions_val = json.dumps(note.definitions)
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
            definitions_val = json.dumps(note.get("definitions") or [])
            timestamp_epoch = float(note.get("timestamp_epoch", 0.0))
            start_time = str(note.get("start_time") or "")
            end_time = str(note.get("end_time") or "")

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
                        session_id, lecture_id, sequence_number, heading, content,
                        key_points_val, concept_tags, slide_refs_val, definitions_val,
                        timestamp_epoch, start_time, end_time
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
        order = "DESC" if descending else "ASC"
        query = f"""
        SELECT event_id, session_id, source_type, timestamp_epoch, elapsed_seconds,
               correlation_id, content, metadata, importance_flags, created_at
        FROM timeline_events
        WHERE session_id = %s
        ORDER BY timestamp_epoch {order}
        LIMIT %s;
        """
        fetch_limit = limit if limit is not None else self.query_limit
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id, fetch_limit))
                rows = await cur.fetchall()
                if descending:
                    rows = list(reversed(rows))
                return rows

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
        async with self._lock:
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
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                if vector is not None:
                    query = """
                    INSERT INTO embeddings (session_id, source_type, source_id, chunk_text, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (session_id, source_id) DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        created_at = NOW();
                    """
                    await cur.execute(query, (session_id, source_type, source_id, chunk_text, vector, embedding_model))
                else:
                    query = """
                    INSERT INTO embeddings (session_id, source_type, source_id, chunk_text, embedding_model)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, source_id) DO UPDATE SET
                        source_type = EXCLUDED.source_type,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding_model = EXCLUDED.embedding_model,
                        created_at = NOW();
                    """
                    await cur.execute(query, (session_id, source_type, source_id, chunk_text, embedding_model))
                return True

    async def delete_embeddings(
        self,
        session_id: str,
        source_ids: list[str]
    ) -> int:
        if not source_ids:
            return 0
        query = """
        DELETE FROM embeddings
        WHERE session_id = %s AND source_id = ANY(%s);
        """
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id, source_ids))
                return cur.rowcount or 0

    async def delete_session_embeddings(
        self,
        session_id: str,
        exclude_model: Optional[str] = None
    ) -> int:
        if exclude_model:
            query = "DELETE FROM embeddings WHERE session_id = %s AND (embedding_model != %s OR embedding_model IS NULL);"
            params = (session_id, exclude_model)
        else:
            query = "DELETE FROM embeddings WHERE session_id = %s;"
            params = (session_id,)
        async with self.transaction() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
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
            ORDER BY distance ASC
            LIMIT %s;
            """
            params = (query_vector, session_id, embedding_model, top_k)
        else:
            query = """
            SELECT chunk_text, source_id, source_type, (embedding <=> %s::vector) AS distance
            FROM embeddings
            WHERE session_id = %s AND embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s;
            """
            params = (query_vector, session_id, top_k)
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                try:
                    await cur.execute(query, params)
                    return await cur.fetchall()
                except Exception as err:
                    # Only a missing pgvector extension is expected here. Any
                    # other failure (a dimension mismatch, a missing column, a
                    # permission error) used to be swallowed into an empty
                    # result, which is indistinguishable from "no similar
                    # chunks exist" and produced quietly wrong answers.
                    message = str(err).lower()
                    if "vector" not in message or "extension" not in message:
                        logger.error("postgres vector search failed: %s", err)
                        raise
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
        # insert_concept stores a session without a course under the
        # "_default_course" sentinel. The read path has to use the same
        # mapping, otherwise those concepts are unreachable.
        course_key = course_id if course_id else DEFAULT_COURSE_KEY
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (course_key,))
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
        async with self._lock:
            conn = await self.get_connection()
            async with conn.cursor() as cur:
                await cur.execute(query, (session_id,))
                return await cur.fetchall()

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None and not self._connection.closed:
                await self._connection.close()
                self._connection = None
