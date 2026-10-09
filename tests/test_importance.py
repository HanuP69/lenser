import unittest
from lecturelens.core.importance import detect_importance_flags


class TestImportance(unittest.TestCase):
    def test_exam_detection(self) -> None:
        text = "This topic is very important and will be on the final exam."
        flags = detect_importance_flags(text)
        self.assertIn("exam_relevant", flags)

    def test_definition_detection(self) -> None:
        text = "A graph is defined as a pair of vertices and edges."
        flags = detect_importance_flags(text)
        self.assertIn("definition", flags)

    def test_example_detection(self) -> None:
        text = "For example, consider an array with five elements."
        flags = detect_importance_flags(text)
        self.assertIn("example", flags)

    def test_confusion_detection(self) -> None:
        text = "Wait what, I don't understand how the pivot is selected."
        flags = detect_importance_flags(text)
        self.assertIn("confusion", flags)

    def test_multiple_flags_detection(self) -> None:
        text = "Remember this key point: the term memoization means caching recursive results."
        flags = detect_importance_flags(text)
        self.assertIn("exam_relevant", flags)
        self.assertIn("definition", flags)

    def test_no_flags_detection(self) -> None:
        text = "Let us open the notebook and proceed."
        flags = detect_importance_flags(text)
        self.assertEqual(len(flags), 0)


if __name__ == "__main__":
    unittest.main()
