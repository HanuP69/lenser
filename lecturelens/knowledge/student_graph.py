import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

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


class StudentKnowledgeGraph:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def init_schema(self) -> None:
        with self.get_connection() as conn:
            conn.executescript(CREATE_STUDENT_GRAPH_SCHEMA)

    def record_concept(self, concept_name: str, mastery_level: float = 0.5) -> None:
        norm_name = concept_name.strip().lower()
        query = """
        INSERT INTO concepts (concept_name, mastery_level, review_count, last_reviewed_epoch)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(concept_name) DO UPDATE SET
            review_count = review_count + 1,
            last_reviewed_epoch = excluded.last_reviewed_epoch;
        """
        with self.get_connection() as conn:
            conn.execute(query, (norm_name, mastery_level, time.time()))

    def update_mastery(self, concept_name: str, delta: float) -> float:
        norm_name = concept_name.strip().lower()
        self.record_concept(norm_name)
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT mastery_level FROM concepts WHERE concept_name = ?", (norm_name,))
            row = cursor.fetchone()
            current = float(row["mastery_level"]) if row else 0.5
            new_level = max(0.0, min(1.0, current + delta))
            conn.execute(
                "UPDATE concepts SET mastery_level = ?, last_reviewed_epoch = ? WHERE concept_name = ?",
                (new_level, time.time(), norm_name)
            )
            return new_level

    def record_misconception(
        self,
        concept_name: str,
        error_pattern: str,
        correction: str,
        timestamp_epoch: Optional[float] = None
    ) -> None:
        norm_name = concept_name.strip().lower()
        self.record_concept(norm_name)
        epoch = timestamp_epoch if timestamp_epoch is not None else time.time()
        query = """
        INSERT INTO misconceptions (concept_name, error_pattern, correction, detected_at_epoch)
        VALUES (?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(query, (norm_name, error_pattern, correction, epoch))
        self.update_mastery(norm_name, -0.2)

    def record_evidence(
        self,
        concept_name: str,
        lecture_id: str,
        timestamp_epoch: float,
        evidence_type: str,
        excerpt: str
    ) -> None:
        norm_name = concept_name.strip().lower()
        self.record_concept(norm_name)
        query = """
        INSERT INTO evidence (concept_name, lecture_id, timestamp_epoch, evidence_type, excerpt)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(query, (norm_name, lecture_id, timestamp_epoch, evidence_type, excerpt))

    def get_known_misconceptions(self, concept_name: Optional[str] = None) -> list[dict[str, Any]]:
        with self.get_connection() as conn:
            if concept_name is not None:
                query = "SELECT * FROM misconceptions WHERE concept_name = ? ORDER BY detected_at_epoch DESC;"
                cursor = conn.execute(query, (concept_name.strip().lower(),))
            else:
                query = "SELECT * FROM misconceptions ORDER BY detected_at_epoch DESC;"
                cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_student_summary(self) -> dict[str, Any]:
        with self.get_connection() as conn:
            concepts_cur = conn.execute("SELECT * FROM concepts ORDER BY mastery_level ASC;")
            concepts = [dict(row) for row in concepts_cur.fetchall()]
            misconceptions_cur = conn.execute("SELECT COUNT(*) as count FROM misconceptions;")
            misconceptions_count = int(misconceptions_cur.fetchone()["count"])
        return {
            "concepts_tracked": len(concepts),
            "concepts": concepts,
            "total_misconceptions": misconceptions_count
        }
