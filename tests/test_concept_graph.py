import tempfile
import unittest
from pathlib import Path
from lecturelens.knowledge.concept_graph import ConceptGraph


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
            definition="Method for solving complex problems by breaking them down into simpler subproblems.",
            related_concepts=["Memoization", "Recursion"]
        )

        found = self.graph.find_concept("dynamic programming")
        self.assertIsNotNone(found)
        self.assertEqual(found["concept_name"], "dynamic programming")
        self.assertEqual(found["session_id"], "session_algos_1")
        self.assertIn("Memoization", found["related_concepts"])

    def test_update_concept(self) -> None:
        self.graph.add_or_update_concept(
            concept_name="BFS",
            session_id="session_1",
            timestamp_epoch=100.0,
            definition="Breadth first search.",
            related_concepts=[]
        )
        self.graph.add_or_update_concept(
            concept_name="BFS",
            session_id="session_2",
            timestamp_epoch=200.0,
            definition="Breadth first search algorithm using queue.",
            related_concepts=["Queue"]
        )

        found = self.graph.find_concept("bfs")
        self.assertEqual(found["session_id"], "session_2")
        self.assertIn("Queue", found["related_concepts"])

    def test_get_session_concepts(self) -> None:
        self.graph.add_or_update_concept(
            concept_name="Merge Sort",
            session_id="session_sorting",
            timestamp_epoch=10.0,
            definition="Divide and conquer sorting.",
            related_concepts=[]
        )
        self.graph.add_or_update_concept(
            concept_name="Quick Sort",
            session_id="session_sorting",
            timestamp_epoch=20.0,
            definition="Partition based sorting.",
            related_concepts=[]
        )

        session_concepts = self.graph.get_all_concepts_for_session("session_sorting")
        self.assertEqual(len(session_concepts), 2)


if __name__ == "__main__":
    unittest.main()
