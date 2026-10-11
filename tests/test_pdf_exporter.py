import tempfile
import unittest
from pathlib import Path

try:
    from lecturelens.exports.pdf_exporter import convert_inline_markup, create_pdf_from_markdown
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@unittest.skipUnless(REPORTLAB_AVAILABLE, "reportlab not installed")
class TestPdfExporter(unittest.TestCase):
    def test_inline_markup_escapes_and_bolds(self) -> None:
        self.assertEqual(convert_inline_markup("a < b **bold** & c"), "a &lt; b <b>bold</b> &amp; c")
        self.assertIn("[!]", convert_inline_markup("\u26a0\ufe0f review"))

    def test_pdf_is_created_from_notes(self) -> None:
        markdown_text = (
            "### 10:00:00\n\n> **Important Exam Topic**\n> - Remember this\n\n"
            "> **Review**: pivot choice\n\nBinary search is **fast**.\n\n- item one\n- item two\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "notes.pdf"
            create_pdf_from_markdown(markdown_text, output, "Lecture notes: demo")
            data = output.read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertGreater(len(data), 1000)

    def test_pdf_with_tables_code_and_math(self) -> None:
        markdown_text = (
            "# Complex Lecture\n\n"
            "| Algorithm | Time | Space |\n"
            "|---|---|---|\n"
            "| Binary Search | $O(\\log n)$ | $O(1)$ |\n"
            "| Merge Sort | $O(n \\log n)$ | $O(n)$ |\n\n"
            "$$\n"
            "T(n) = 2T(n/2) + O(n)\n"
            "$$\n\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    return -1\n"
            "```\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "tables_notes.pdf"
            create_pdf_from_markdown(markdown_text, output, "Algorithms")
            data = output.read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertGreater(len(data), 1000)


if __name__ == "__main__":
    unittest.main()
