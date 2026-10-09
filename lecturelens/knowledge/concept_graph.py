import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class ConceptGraph:
    def __init__(self, database_path: Path) -> None:
        self.database_path: Path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()

    def get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def create_tables(self) -> None:
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT UNIQUE,
                session_id TEXT,
                timestamp_epoch REAL,
                definition TEXT,
                related_concepts TEXT
            )
            """
        )
        connection.commit()
        connection.close()

    def add_or_update_concept(
        self,
        concept_name: str,
        session_id: str,
        timestamp_epoch: float,
        definition: str,
        related_concepts: list[str]
    ) -> None:
        clean_name = concept_name.strip().lower()
        if len(clean_name) == 0:
            return

        related_json = json.dumps(related_concepts)
        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO concepts (concept_name, session_id, timestamp_epoch, definition, related_concepts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(concept_name) DO UPDATE SET
                session_id = excluded.session_id,
                timestamp_epoch = excluded.timestamp_epoch,
                definition = excluded.definition,
                related_concepts = excluded.related_concepts
            """,
            (clean_name, session_id, timestamp_epoch, definition, related_json)
        )
        connection.commit()
        connection.close()

    def find_concept(self, concept_name: str) -> Optional[dict[str, Any]]:
        clean_name = concept_name.strip().lower()
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM concepts WHERE concept_name = ?",
            (clean_name,)
        )
        row = cursor.fetchone()
        connection.close()
        if row is None:
            return None

        result = dict(row)
        result["related_concepts"] = json.loads(result["related_concepts"])
        return result

    def get_all_concepts_for_session(self, session_id: str) -> list[dict[str, Any]]:
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM concepts WHERE session_id = ? ORDER BY concept_name ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        connection.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            entry["related_concepts"] = json.loads(entry["related_concepts"])
            results.append(entry)
        return results

    def get_all_concepts(self) -> list[dict[str, Any]]:
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM concepts ORDER BY concept_name ASC")
        rows = cursor.fetchall()
        connection.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            entry["related_concepts"] = json.loads(entry["related_concepts"])
            results.append(entry)
        return results

    def get_course_overview(self) -> dict[str, list[dict[str, Any]]]:
        all_concepts = self.get_all_concepts()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for concept in all_concepts:
            session_key = concept["session_id"]
            if session_key not in grouped:
                grouped[session_key] = []
            grouped[session_key].append(concept)
        return grouped
