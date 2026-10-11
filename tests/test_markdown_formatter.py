import unittest

from lecturelens.core.markdown_formatter import (
    clean_mojibake_and_emojis,
    format_clean_markdown,
    normalize_headings,
    normalize_lists_and_definitions,
    normalize_math_blocks
)


class TestMarkdownFormatter(unittest.TestCase):
    def test_clean_mojibake_and_emojis(self) -> None:
        raw = "ðŸ“ Quadratic Equations and Polynomials \U0001f4cc \U0001f4d6"
        cleaned = clean_mojibake_and_emojis(raw)
        self.assertNotIn("ðŸ“", cleaned)
        self.assertNotIn("\U0001f4cc", cleaned)
        self.assertNotIn("\U0001f4d6", cleaned)
        self.assertIn("Quadratic Equations", cleaned)

    def test_normalize_math_blocks(self) -> None:
        raw = "The formula is: $$P(x) = 0$$ where P is polynomial."
        normalized = normalize_math_blocks(raw)
        self.assertIn("\n\n$$\nP(x) = 0\n$$\n\n", normalized)

    def test_normalize_multiple_math_blocks_on_same_line(self) -> None:
        raw = "Given equations: $$\\text{Area} = l \\times w$$ $$7,000 = (w+1)w$$ done."
        normalized = normalize_math_blocks(raw)
        self.assertEqual(normalized.count("$$"), 4)
        self.assertIn("\n\n$$\n\\text{Area} = l \\times w\n$$\n\n", normalized)
        self.assertIn("\n\n$$\n7,000 = (w+1)w\n$$\n\n", normalized)

    def test_normalize_headings_roman_and_continuation(self) -> None:
        raw = "I. Review of Coefficients\nTopic (Cont.)\n### Next Section (Continued)"
        normalized = normalize_headings(raw)
        self.assertIn("## Review of Coefficients", normalized)
        self.assertNotIn("(Cont.)", normalized)
        self.assertNotIn("(Continued)", normalized)

    def test_normalize_lists_and_definitions(self) -> None:
        raw = "$A = 2$\n$B = 4$\n$C = 2$\n\nIdentify:\n$A$: First\n$B$: Second"
        normalized = normalize_lists_and_definitions(raw)
        self.assertIn("- $A = 2$", normalized)
        self.assertIn("- $B = 4$", normalized)
        self.assertIn("- $C = 2$", normalized)
        self.assertIn("- $A$: First", normalized)
        self.assertIn("- $B$: Second", normalized)

    def test_end_to_end_format_clean_markdown(self) -> None:
        sample = (
            "$A = 2$\n$B = 4$\n$C = 2$\n\n"
            "ðŸ“ Quadratic Equations (Cont.)\n"
            "I. Overview\n"
            "Formula: $$x^2 + bx + c = 0$$ is quadratic.\n"
            "Identify:\n$A$: Squared coefficient\n"
        )
        cleaned = format_clean_markdown(sample)
        self.assertNotIn("ðŸ“", cleaned)
        self.assertNotIn("(Cont.)", cleaned)
        self.assertIn("## Overview", cleaned)
        self.assertIn("- $A = 2$", cleaned)
        self.assertIn("- $A$: Squared coefficient", cleaned)
        self.assertIn("\n\n$$\nx^2 + bx + c = 0\n$$\n\n", cleaned)

    def test_strip_scratchpad_and_thoughts(self) -> None:
        raw = (
            "01:51:40\n"
            "Expert academic note taker. O. Henry's 'Hearts and Hands'.\n"
            "Sub-topic 1: The Persona\n"
            "Heading 1: The Persona\n"
            "Drafting Section 1: Detail text here.\n"
            "Refining Section 1: Better academic words.\n"
            "Markdown headings? Yes.\n"
            "No Roman numerals? Yes.\n"
            "Ignore the DP slide? Yes.\n"
            "## The Reinforcement of the False Persona\n\n"
            "Actual content of the lecture notes starts here."
        )
        cleaned = format_clean_markdown(raw)
        self.assertNotIn("Drafting Section 1", cleaned)
        self.assertNotIn("Refining Section 1", cleaned)
        self.assertNotIn("Markdown headings? Yes", cleaned)
        self.assertNotIn("Ignore the DP slide", cleaned)
        self.assertNotIn("Sub-topic 1", cleaned)
        self.assertTrue(cleaned.startswith("## The Reinforcement of the False Persona"))
        self.assertIn("Actual content of the lecture notes starts here.", cleaned)


if __name__ == "__main__":
    unittest.main()
