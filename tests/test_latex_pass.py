import unittest

from lecturelens.agents.structuring_agent import (
    LATEX_ENFORCEMENT_PROMPT,
    latex_result_is_usable,
    text_has_mathematics
)
from lecturelens.core.markdown_formatter import format_clean_markdown


class TestLatexPrompt(unittest.TestCase):
    def test_the_prompt_can_hold_a_summary(self) -> None:
        """The prompt holds LaTeX with braces in it.

        Putting it together with str.format read {Index} as a field and
        the pass failed for every section that had a formula in it.
        """
        summary = "## A section\n\nThe cost is O(1) per lookup."
        prompt = LATEX_ENFORCEMENT_PROMPT.replace("%%SUMMARY%%", summary)
        self.assertIn(summary, prompt)
        self.assertIn("\\text{Index}", prompt)


class TestLatexDetection(unittest.TestCase):
    def test_mathematics_is_found(self) -> None:
        self.assertTrue(text_has_mathematics("The cost is O(1) per lookup."))
        self.assertTrue(text_has_mathematics("the function H(x) maps a key"))
        self.assertTrue(text_has_mathematics("Redis ranked 64.8% in the survey"))
        self.assertTrue(text_has_mathematics(r"the cost is $O(1)$ per lookup"))
        self.assertTrue(text_has_mathematics("the size is 2^n"))

    def test_text_without_mathematics_is_not_found(self) -> None:
        self.assertFalse(text_has_mathematics("Redis is an in-memory data store."))
        self.assertFalse(text_has_mathematics("A hash table maps keys to values."))


class TestLatexResult(unittest.TestCase):
    def test_a_rewrite_that_keeps_the_notes_is_used(self) -> None:
        original = "## A section\n\nThe cost is O(1) per lookup."
        rewritten = "## A section\n\nThe cost is $O(1)$ per lookup."
        self.assertTrue(latex_result_is_usable(original, rewritten))

    def test_a_rewrite_that_lost_the_headings_is_not_used(self) -> None:
        original = "## A section\n\nThe cost is O(1) per lookup."
        rewritten = "The cost is $O(1)$ per lookup."
        self.assertFalse(latex_result_is_usable(original, rewritten))

    def test_a_rewrite_that_lost_the_text_is_not_used(self) -> None:
        original = "## A section\n\nThe cost is O(1) per lookup, and more text."
        self.assertFalse(latex_result_is_usable(original, "short"))

    def test_a_rewrite_that_added_a_lot_is_not_used(self) -> None:
        original = "## A section\n\nThe cost is O(1) per lookup."
        self.assertFalse(latex_result_is_usable(original, original * 5))


class TestIndentedContinuation(unittest.TestCase):
    def test_an_indented_emphasis_is_not_a_code_block(self) -> None:
        """The model writes the continuation of a numbered item with an
        indent. Markdown turned it into a code block, so the asterisks of
        the emphasis were shown as they are."""
        text = (
            "3.  **Mapping:** the index is computed like this:\n"
            "\n"
            "$$\n"
            "\\text{Index} = H(\\text{Key}) \\mod M\n"
            "$$\n"
            "\n"
            "    *(where $M$ is the size of the hash table)*.\n"
        )
        cleaned = format_clean_markdown(text)
        self.assertIn("\n*(where $M$ is the size of the hash table)*.", cleaned)
        self.assertNotIn("    *(where", cleaned)

    def test_a_nested_list_is_kept(self) -> None:
        text = "- A point\n    - A nested point\n"
        cleaned = format_clean_markdown(text)
        self.assertIn("    - A nested point", cleaned)

    def test_a_real_code_block_is_kept(self) -> None:
        text = "```\n    indented code\n```\n"
        cleaned = format_clean_markdown(text)
        self.assertIn("    indented code", cleaned)


if __name__ == "__main__":
    unittest.main()
