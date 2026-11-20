import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
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
    elapsed_days = max(0.0, time.time() - last_reviewed_epoch) / 86400.0
    return 0.5 + (mastery - 0.5) * (0.5 ** (elapsed_days / max(0.1, half_life_days)))


class StudentKnowledgeGraph:
    """SQLite-backed Student Knowledge Graph tracking concept mastery and misconceptions."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
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
        name = self.normalize(concept_name)
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO concepts (concept_name, mastery_level, review_count, last_reviewed_epoch)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(concept_name) DO UPDATE SET
                       review_count = review_count + 1,
                       last_reviewed_epoch = excluded.last_reviewed_epoch;""",
                (name, mastery_level, time.time())
            )

    def update_mastery(self, concept_name: str, delta: float) -> float:
        name = self.normalize(concept_name)
        with self.connection() as conn:
            row = conn.execute("SELECT mastery_level, last_reviewed_epoch FROM concepts WHERE concept_name = ?", (name,)).fetchone()
            current = 0.5 if row is None else compute_decayed_mastery(float(row["mastery_level"]), float(row["last_reviewed_epoch"] or 0.0))
            new_level = max(0.0, min(1.0, current + delta))
            if row is None:
                conn.execute("INSERT INTO concepts (concept_name, mastery_level, review_count, last_reviewed_epoch) VALUES (?, ?, 1, ?)", (name, new_level, time.time()))
            else:
                conn.execute("UPDATE concepts SET mastery_level = ?, review_count = review_count + 1, last_reviewed_epoch = ? WHERE concept_name = ?", (new_level, time.time(), name))
            return new_level

    def record_misconception(self, concept_name: str, error_pattern: str, correction: str, timestamp_epoch: Optional[float] = None, confidence: float = 0.8) -> None:
        name = self.normalize(concept_name)
        epoch = time.time() if timestamp_epoch is None else timestamp_epoch
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO misconceptions (concept_name, error_pattern, correction, detected_at_epoch, confidence) VALUES (?, ?, ?, ?, ?)",
                (name, error_pattern, correction, epoch, confidence)
            )
        self.update_mastery(name, -0.2)

    def record_evidence(self, concept_name: str, lecture_id: str, timestamp_epoch: float, evidence_type: str, excerpt: str) -> None:
        name = self.normalize(concept_name)
        self.record_concept(name)
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO evidence (concept_name, lecture_id, timestamp_epoch, evidence_type, excerpt) VALUES (?, ?, ?, ?, ?)",
                (name, lecture_id, timestamp_epoch, evidence_type, excerpt)
            )

    def get_known_misconceptions(self, concept_name: Optional[str] = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if concept_name is not None:
                cur = conn.execute("SELECT * FROM misconceptions WHERE concept_name = ? ORDER BY detected_at_epoch DESC", (self.normalize(concept_name),))
            else:
                cur = conn.execute("SELECT * FROM misconceptions ORDER BY detected_at_epoch DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_student_summary(self) -> dict[str, Any]:
        with self.connection() as conn:
            concepts = [
                {**dict(r), "decayed_mastery": compute_decayed_mastery(float(r["mastery_level"]), float(r["last_reviewed_epoch"] or 0.0))}
                for r in conn.execute("SELECT * FROM concepts ORDER BY mastery_level ASC").fetchall()
            ]
            misc_cnt = conn.execute("SELECT COUNT(*) as c FROM misconceptions").fetchone()["c"]
            ev_cnt = conn.execute("SELECT COUNT(*) as c FROM evidence").fetchone()["c"]
        return {"concepts_tracked": len(concepts), "concepts": concepts, "total_misconceptions": int(misc_cnt), "total_evidence_entries": int(ev_cnt)}
