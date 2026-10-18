import asyncio
import tempfile
import unittest
from pathlib import Path

from lecturelens.knowledge.student_chat import StudentChatAgent
from lecturelens.knowledge.student_graph import StudentKnowledgeGraph


class MockChatRouter:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.recorded_prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        self.recorded_prompts.append(messages[0].content)
        return self.answer


class TestStudentKnowledgeGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "student_graph.db"
        self.graph = StudentKnowledgeGraph(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_concept_and_update_mastery(self) -> None:
        self.graph.record_concept("Dynamic Programming", mastery_level=0.4)
        new_level = self.graph.update_mastery("dynamic programming", 0.3)
        self.assertAlmostEqual(new_level, 0.7)
        summary = self.graph.get_student_summary()
        self.assertEqual(summary["concepts_tracked"], 1)

    def test_record_misconception_lowers_mastery_and_records_history(self) -> None:
        self.graph.record_concept("Pointers", mastery_level=0.8)
        self.graph.record_misconception(
            concept_name="Pointers",
            error_pattern="Thought pointer points to variable value instead of address",
            correction="A pointer stores the memory address of the target variable"
        )
        misconceptions = self.graph.get_known_misconceptions("pointers")
        self.assertEqual(len(misconceptions), 1)
        self.assertIn("memory address", misconceptions[0]["correction"])
        summary = self.graph.get_student_summary()
        self.assertEqual(summary["total_misconceptions"], 1)
        self.assertLess(summary["concepts"][0]["mastery_level"], 0.8)

    def test_record_evidence(self) -> None:
        self.graph.record_evidence(
            concept_name="Quicksort",
            lecture_id="lec_algo_04",
            timestamp_epoch=1700000100.0,
            evidence_type="quiz_error",
            excerpt="Chose worst-case pivot"
        )
        with self.graph.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM evidence WHERE concept_name = 'quicksort'")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["lecture_id"], "lec_algo_04")


class TestStudentChatAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "student_graph.db"
        self.graph = StudentKnowledgeGraph(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_chat_extracts_citations_and_cleans_tags(self) -> None:
        tutor_response = (
            "As explained at [02:15], binary search halves the array each step. "
            "At [05:40], the professor notes that the array must be sorted first."
        )
        router = MockChatRouter(tutor_response)
        agent = StudentChatAgent(router, self.graph)
        answer, citations = await agent.ask(
            question="Why does binary search need sorted data?",
            timeline_events=[
                {"elapsed_seconds": 135.0, "source_type": "transcript", "content": "halving array"},
                {"elapsed_seconds": 340.0, "source_type": "slide", "content": "sorted prerequisite"}
            ]
        )
        self.assertEqual(answer, tutor_response)
        self.assertIn("[02:15]", citations)
        self.assertIn("[05:40]", citations)

    async def test_chat_detects_misconception_and_records_to_graph(self) -> None:
        tutor_response_with_tag = (
            "Actually, an empty tree is still a valid tree! As covered at [01:10], trees can have 0 nodes.\n"
            '[MISCONCEPTION: {"concept": "trees", "error": "assumed trees must have >= 1 node", "correction": "trees can be empty with 0 nodes"}]'
        )
        router = MockChatRouter(tutor_response_with_tag)
        agent = StudentChatAgent(router, self.graph)
        answer, _ = await agent.ask("Does every tree have to have at least one node?")
        self.assertNotIn("[MISCONCEPTION:", answer)
        self.assertIn("empty tree is still a valid tree", answer)

        misconceptions = self.graph.get_known_misconceptions("trees")
        self.assertEqual(len(misconceptions), 1)
        self.assertEqual(misconceptions[0]["error_pattern"], "assumed trees must have >= 1 node")


if __name__ == "__main__":
    unittest.main()
