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

    def test_hindi_and_hinglish_detection(self) -> None:
        text_exam = "Ye concept bahut important hai aur exam me aayega dhyan do."
        self.assertIn("exam_relevant", detect_importance_flags(text_exam))

        text_def = "Is concept ka matlab aur paribhasha samjho."
        self.assertIn("definition", detect_importance_flags(text_def))

        text_ex = "Jaise ki maan lo hamare paas ek array hai."
        self.assertIn("example", detect_importance_flags(text_ex))

        text_conf = "Mujhe samajh nahi aaya doubt hai."
        self.assertIn("confusion", detect_importance_flags(text_conf))


if __name__ == "__main__":
    unittest.main()


class TestImportanceFalsePositives(unittest.TestCase):
    def test_example_is_not_exam(self) -> None:
        flags = detect_importance_flags("For example, consider sorting.")
        self.assertNotIn("exam_relevant", flags)
        self.assertIn("example", flags)

    def test_latest_is_not_test(self) -> None:
        self.assertEqual(detect_importance_flags("The latest version is out."), [])

    def test_finally_is_not_final(self) -> None:
        self.assertEqual(detect_importance_flags("Finally, we are done."), [])

    def test_unimportant_is_not_important(self) -> None:
        self.assertEqual(detect_importance_flags("That detail is unimportant."), [])

    def test_real_exam_mentions(self) -> None:
        self.assertIn("exam_relevant", detect_importance_flags("This will be on the exam."))
        self.assertIn("exam_relevant", detect_importance_flags("There is a quiz on Friday."))
        self.assertIn("exam_relevant", detect_importance_flags("Midterms are next week."))

    def test_curly_apostrophe_confusion(self) -> None:
        self.assertIn("confusion", detect_importance_flags("I don\u2019t understand this step."))

    def test_bare_test_false_positive_avoided(self) -> None:
        self.assertEqual(detect_importance_flags("This is a good test of the system."), [])

    def test_dhyaan_variants_detected(self) -> None:
        self.assertIn("exam_relevant", detect_importance_flags("Is point par dhyaan do."))
        self.assertIn("exam_relevant", detect_importance_flags("Is point par dhyan do."))

    def test_unit_test_passes_not_flagged(self) -> None:
        self.assertEqual(detect_importance_flags("All unit test passes in our suite."), [])

    def test_upcoming_and_surprise_test_flagged(self) -> None:
        self.assertIn("exam_relevant", detect_importance_flags("The test will cover graph algorithms."))
        self.assertIn("exam_relevant", detect_importance_flags("There is a test next Monday."))
        self.assertIn("exam_relevant", detect_importance_flags("Get ready for a surprise test."))
