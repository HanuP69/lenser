import sqlite3
import tempfile
import unittest
from pathlib import Path

from lecturelens.knowledge.concept_graph import ConceptGraph
from tests.support import pluck


class TestConceptGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_concepts.db"
        self.graph = ConceptGraph(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_and_find_concept(self) -> None:
        self.graph.add_or_update_concept(
            concept_name="Dynamic Programming",
            session_id="session_algos_1",
            timestamp_epoch=1700000000.0,
            definition="Method for solving complex problems by breaking them down.",
            related_concepts=["Memoization", "Recursion"],
            course_name="algorithms"
        )
        found = self.graph.find_concept("dynamic programming")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["concept_name"], "dynamic programming")
        self.assertEqual(found[0]["session_id"], "session_algos_1")
        self.assertEqual(found[0]["course_name"], "algorithms")
        self.assertIn("Memoization", found[0]["related_concepts"])

    def test_same_concept_in_two_sessions_is_kept_for_both(self) -> None:
        self.graph.add_or_update_concept("BFS", "session_1", 100.0, "Breadth first search.", [])
        self.graph.add_or_update_concept("BFS", "session_2", 200.0, "BFS using a queue.", ["Queue"])
        found = self.graph.find_concept("bfs")
        self.assertEqual(pluck(found, "session_id"), ["session_1", "session_2"])
        others = self.graph.find_concept_in_other_sessions("BFS", "session_2")
        self.assertEqual(pluck(others, "session_id"), ["session_1"])

    def test_update_within_same_session_replaces_definition(self) -> None:
        self.graph.add_or_update_concept("BFS", "session_1", 100.0, "old", [])
        self.graph.add_or_update_concept("BFS", "session_1", 150.0, "new", ["Queue"])
        found = self.graph.find_concept("bfs")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["definition"], "new")
        self.assertEqual(found[0]["related_concepts"], ["Queue"])

    def test_overview_groups_by_session(self) -> None:
        self.graph.add_or_update_concept("A", "s1", 1.0, "d", [])
        self.graph.add_or_update_concept("B", "s1", 2.0, "d", [])
        self.graph.add_or_update_concept("C", "s2", 3.0, "d", [])
        overview = self.graph.get_course_overview()
        self.assertEqual(sorted(overview.keys()), ["s1", "s2"])
        self.assertEqual(len(overview["s1"]), 2)

    def test_delete_session_only_removes_that_session(self) -> None:
        self.graph.add_or_update_concept("A", "s1", 1.0, "d", [])
        self.graph.add_or_update_concept("A", "s2", 2.0, "d", [])
        self.graph.delete_session("s1")
        self.assertEqual(pluck(self.graph.find_concept("a"), "session_id"), ["s2"])

    def test_blank_names_are_ignored(self) -> None:
        self.graph.add_or_update_concept("   ", "s1", 1.0, "d", [])
        self.assertEqual(self.graph.get_all_concepts(), [])

    def test_legacy_schema_is_migrated_without_losing_rows(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(str(legacy_path))
        connection.execute(
            "CREATE TABLE concepts (concept_name TEXT UNIQUE, session_id TEXT, timestamp_epoch REAL, definition TEXT, related_concepts TEXT)"
        )
        connection.execute("INSERT INTO concepts VALUES ('heap', 'old_session', 5.0, 'a tree', '[\"tree\"]')")
        connection.commit()
        connection.close()

        migrated = ConceptGraph(legacy_path)
        found = migrated.find_concept("heap")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["related_concepts"], ["tree"])
        migrated.add_or_update_concept("heap", "new_session", 9.0, "again", [])
        self.assertEqual(len(migrated.find_concept("heap")), 2)


if __name__ == "__main__":
    unittest.main()
