import difflib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Optional


FUZZY_MATCH_RATIO: float = 0.88
LEADING_ARTICLES: tuple[str, ...] = ("the ", "a ", "an ")


def singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_concept_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"[-_/]", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    for article in LEADING_ARTICLES:
        if lowered.startswith(article):
            lowered = lowered[len(article):]
    return " ".join(singularize_word(word) for word in lowered.split(" ")).strip()


class ConceptGraph:
    def __init__(self, db_path: Path) -> None:
        self.db_path: Path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def create_tables(self) -> None:
        with closing(self.connect()) as connection:
            existing = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'"
            ).fetchone()
            if existing is not None:
                columns = connection.execute("PRAGMA table_info(concepts)").fetchall()
                column_names: list[str] = []
                for column in columns:
                    column_names.append(column["name"])
                if "course_name" not in column_names or "id" not in column_names:
                    self.migrate_legacy_table(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    course_name TEXT,
                    timestamp_epoch REAL,
                    definition TEXT,
                    related_concepts TEXT,
                    UNIQUE(concept_name, session_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(concept_name)"
            )
            connection.commit()

    def migrate_legacy_table(self, connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE concepts RENAME TO concepts_legacy")
        connection.execute(
            """
            CREATE TABLE concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                course_name TEXT,
                timestamp_epoch REAL,
                definition TEXT,
                related_concepts TEXT,
                UNIQUE(concept_name, session_id)
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO concepts
                (concept_name, session_id, timestamp_epoch, definition, related_concepts)
            SELECT concept_name, session_id, timestamp_epoch, definition, related_concepts
            FROM concepts_legacy
            """
        )
        connection.execute("DROP TABLE concepts_legacy")

    def row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        related: list[str] = []
        raw_related = row["related_concepts"]
        if raw_related:
            try:
                parsed = json.loads(raw_related)
                if isinstance(parsed, list):
                    for item in parsed:
                        related.append(str(item))
            except ValueError:
                related = []
        return {
            "concept_name": row["concept_name"],
            "session_id": row["session_id"],
            "course_name": row["course_name"],
            "timestamp_epoch": row["timestamp_epoch"],
            "definition": row["definition"],
            "related_concepts": related
        }

    def add_or_update_concept(
        self,
        concept_name: str,
        session_id: str,
        timestamp_epoch: float,
        definition: str,
        related_concepts: list[str],
        course_name: Optional[str] = None
    ) -> None:
        normalized_name = normalize_concept_name(concept_name)
        if len(normalized_name) == 0:
            return
        related_json = json.dumps(related_concepts)
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO concepts
                    (concept_name, session_id, course_name, timestamp_epoch, definition, related_concepts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(concept_name, session_id) DO UPDATE SET
                    course_name = excluded.course_name,
                    timestamp_epoch = excluded.timestamp_epoch,
                    definition = excluded.definition,
                    related_concepts = excluded.related_concepts
                """,
                (normalized_name, session_id, course_name, timestamp_epoch, definition, related_json)
            )
            connection.commit()

    def find_concept(self, concept_name: str) -> list[dict[str, Any]]:
        normalized_name = normalize_concept_name(concept_name)
        if len(normalized_name) == 0:
            return []
        with closing(self.connect()) as connection:
            stored_names = [
                row[0] for row in connection.execute("SELECT DISTINCT concept_name FROM concepts").fetchall()
            ]
            matching_names = [
                stored for stored in stored_names
                if self.names_match(normalized_name, normalize_concept_name(stored))
            ]
            if len(matching_names) == 0:
                return []
            placeholders = ",".join("?" for _ in matching_names)
            rows = connection.execute(
                f"SELECT * FROM concepts WHERE concept_name IN ({placeholders}) ORDER BY timestamp_epoch ASC",
                matching_names
            ).fetchall()
        return [self.row_to_dict(row) for row in rows]

    @staticmethod
    def names_match(left: str, right: str) -> bool:
        if left == right:
            return True
        if min(len(left), len(right)) < 6:
            return False
        return difflib.SequenceMatcher(None, left, right).ratio() >= FUZZY_MATCH_RATIO

    def find_concept_in_other_sessions(
        self,
        concept_name: str,
        current_session_id: str
    ) -> list[dict[str, Any]]:
        matches = self.find_concept(concept_name)
        others: list[dict[str, Any]] = []
        for match in matches:
            if match["session_id"] != current_session_id:
                others.append(match)
        return others

    def get_all_concepts_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM concepts WHERE session_id = ? ORDER BY timestamp_epoch ASC",
                (session_id,)
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(self.row_to_dict(row))
        return results

    def get_all_concepts(self) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM concepts ORDER BY session_id ASC, timestamp_epoch ASC"
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(self.row_to_dict(row))
        return results

    def get_course_overview(self) -> dict[str, list[dict[str, Any]]]:
        overview: dict[str, list[dict[str, Any]]] = {}
        for concept in self.get_all_concepts():
            session_key = concept["session_id"]
            if session_key not in overview:
                overview[session_key] = []
            overview[session_key].append(concept)
        return overview

    def delete_session(self, session_id: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute("DELETE FROM concepts WHERE session_id = ?", (session_id,))
            connection.commit()
