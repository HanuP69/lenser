import logging
import math
import re
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

CREATE_STUDENT_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_name TEXT PRIMARY KEY,
    mastery_level REAL DEFAULT 0.5,
    review_count INTEGER DEFAULT 0,
    last_reviewed_epoch REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS misconceptions (
    misconception_id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_name TEXT NOT NULL,
    error_pattern TEXT NOT NULL,
    correction TEXT NOT NULL,
    detected_at_epoch REAL NOT NULL,
    confidence REAL DEFAULT 0.8,
    FOREIGN KEY(concept_name) REFERENCES concepts(concept_name)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_name TEXT NOT NULL,
    lecture_id TEXT NOT NULL,
    timestamp_epoch REAL NOT NULL,
    evidence_type TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    FOREIGN KEY(concept_name) REFERENCES concepts(concept_name)
);

CREATE INDEX IF NOT EXISTS idx_misconceptions_concept ON misconceptions(concept_name);
CREATE INDEX IF NOT EXISTS idx_evidence_concept ON evidence(concept_name);
"""


def normalize_concept_name(name: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"[^\w_]", "", cleaned)


def compute_decayed_mastery(mastery: float, last_reviewed_epoch: float, half_life_days: float = 7.0) -> float:
    if last_reviewed_epoch <= 0:
        return mastery
    elapsed_seconds = max(0.0, time.time() - last_reviewed_epoch)
    elapsed_days = elapsed_seconds / 86400.0
    decay_factor = 0.5 ** (elapsed_days / max(0.1, half_life_days))
    baseline = 0.5
    return baseline + (mastery - baseline) * decay_factor


class StudentKnowledgeGraph:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection and always close it.

        ``with sqlite3.connect(...)`` is a transaction context manager: it
        commits or rolls back but does not close the connection, so every
        call leaked a file handle.
        """
        conn = self.get_connection()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(CREATE_STUDENT_GRAPH_SCHEMA)

    def normalize(self, name: str) -> str:
        return normalize_concept_name(name)

    def record_concept(self, concept_name: str, mastery_level: float = 0.5) -> None:
        norm_name = self.normalize(concept_name)
        query = """
        INSERT INTO concepts (concept_name, mastery_level, review_count, last_reviewed_epoch)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(concept_name) DO UPDATE SET
            review_count = review_count + 1,
            last_reviewed_epoch = excluded.last_reviewed_epoch;
        """
        with self.connection() as conn:
            conn.execute(query, (norm_name, mastery_level, time.time()))

    def update_mastery(self, concept_name: str, delta: float) -> float:
        norm_name = self.normalize(concept_name)
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT mastery_level, last_reviewed_epoch FROM concepts WHERE concept_name = ?",
                (norm_name,)
            )
            row = cursor.fetchone()
            if row is None:
                current = 0.5
                new_level = max(0.0, min(1.0, current + delta))
                insert_query = """
                INSERT INTO concepts (concept_name, mastery_level, review_count, last_reviewed_epoch)
                VALUES (?, ?, 1, ?);
                """
                conn.execute(insert_query, (norm_name, new_level, time.time()))
                return new_level
            current = float(row["mastery_level"])
            last_reviewed = row["last_reviewed_epoch"]
            if last_reviewed is not None:
                # The graph decays mastery towards the baseline over time and
                # the summary reports the decayed value. The delta used to be
                # applied to the stale stored value, which over-credited the
                # student relative to the model the interface shows.
                current = compute_decayed_mastery(current, float(last_reviewed))
            new_level = max(0.0, min(1.0, current + delta))
            conn.execute(
                "UPDATE concepts SET mastery_level = ?, review_count = review_count + 1, last_reviewed_epoch = ? WHERE concept_name = ?",
                (new_level, time.time(), norm_name)
            )
            return new_level

    def record_misconception(
        self,
        concept_name: str,
        error_pattern: str,
        correction: str,
        timestamp_epoch: Optional[float] = None,
        confidence: float = 0.8
    ) -> None:
        norm_name = self.normalize(concept_name)
        epoch = timestamp_epoch if timestamp_epoch is not None else time.time()
        query = """
        INSERT INTO misconceptions (concept_name, error_pattern, correction, detected_at_epoch, confidence)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.connection() as conn:
            conn.execute(query, (norm_name, error_pattern, correction, epoch, confidence))
        self.update_mastery(norm_name, -0.2)

    def record_evidence(
        self,
        concept_name: str,
        lecture_id: str,
        timestamp_epoch: float,
        evidence_type: str,
        excerpt: str
    ) -> None:
        norm_name = self.normalize(concept_name)
        self.record_concept(norm_name)
        query = """
        INSERT INTO evidence (concept_name, lecture_id, timestamp_epoch, evidence_type, excerpt)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.connection() as conn:
            conn.execute(query, (norm_name, lecture_id, timestamp_epoch, evidence_type, excerpt))

    def get_known_misconceptions(self, concept_name: Optional[str] = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if concept_name is not None:
                query = "SELECT * FROM misconceptions WHERE concept_name = ? ORDER BY detected_at_epoch DESC;"
                cursor = conn.execute(query, (self.normalize(concept_name),))
            else:
                query = "SELECT * FROM misconceptions ORDER BY detected_at_epoch DESC;"
                cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_student_summary(self) -> dict[str, Any]:
        with self.connection() as conn:
            concepts_cur = conn.execute("SELECT * FROM concepts ORDER BY mastery_level ASC;")
            concepts = []
            for row in concepts_cur.fetchall():
                c_dict = dict(row)
                c_dict["decayed_mastery"] = compute_decayed_mastery(
                    float(c_dict["mastery_level"]),
                    float(c_dict.get("last_reviewed_epoch", 0.0))
                )
                concepts.append(c_dict)
            misconceptions_cur = conn.execute("SELECT COUNT(*) as count FROM misconceptions;")
            misconceptions_count = int(misconceptions_cur.fetchone()["count"])
            evidence_cur = conn.execute("SELECT COUNT(*) as count FROM evidence;")
            evidence_count = int(evidence_cur.fetchone()["count"])
        return {
            "concepts_tracked": len(concepts),
            "concepts": concepts,
            "total_misconceptions": misconceptions_count,
            "total_evidence_entries": evidence_count
        }
