import unittest

from lecturelens.core.notes_html import render_notes_html


class TestNotesHtml(unittest.TestCase):
    def test_the_page_has_the_title(self) -> None:
        page = render_notes_html("# Heading", title="Lecture Notes - cs101")
        self.assertIn("<title>Lecture Notes - cs101</title>", page)
        self.assertIn("<h1>Heading</h1>", page)

    def test_a_formula_is_handed_to_mathjax(self) -> None:
        page = render_notes_html("$$\n\\text{Index} = H(\\text{Key}) \\mod M\n$$\n")
        # MathJax is told to look for these delimiters.
        self.assertIn("\\[", page)
        self.assertIn("\\text{Index}", page)
        self.assertIn("mathjax", page.lower())

    def test_an_inline_formula_is_handed_to_mathjax(self) -> None:
        page = render_notes_html("The cost is $O(1)$ per lookup.")
        self.assertIn("\\(", page)
        self.assertIn("O(1)", page)

    def test_the_time_mark_becomes_a_label(self) -> None:
        page = render_notes_html("### 00:03:36\n\n## A section\n")
        self.assertIn('class="time-mark">00:03:36<', page)
        self.assertNotIn("<h3>00:03:36</h3>", page)

    def test_a_table_is_rendered(self) -> None:
        page = render_notes_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
        self.assertIn("<table>", page)
        self.assertIn("<th>a</th>", page)

    def test_code_blocks_are_kept(self) -> None:
        page = render_notes_html("```text\nKey -> Hash\n```\n")
        self.assertIn("<pre>", page)
        self.assertIn("Key -&gt; Hash", page)

    def test_the_formulas_are_not_left_as_latex(self) -> None:
        """A viewer that does not draw formulas shows them as they are.

        The page is only useful when the formulas are drawn, so the
        renderer has to be there.
        """
        page = render_notes_html("$$\n\\text{Index}\n$$\n")
        self.assertIn("<div class=\"math block\">", page)

    def test_a_broken_title_does_not_break_the_page(self) -> None:
        page = render_notes_html("text", title="<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", page)


if __name__ == "__main__":
    unittest.main()
