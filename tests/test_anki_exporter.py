import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import genanki
    from lecturelens.exports.anki_exporter import compute_stable_deck_id, create_anki_deck
    GENANKI_AVAILABLE = True
except ImportError:
    GENANKI_AVAILABLE = False


def read_note_fields(apkg_path: Path, work_dir: Path) -> str:
    with zipfile.ZipFile(apkg_path) as archive:
        archive.extract("collection.anki2", str(work_dir))
    database_path = work_dir / "collection.anki2"
    try:
        connection = sqlite3.connect(str(database_path))
        rows = connection.execute("SELECT flds FROM notes").fetchall()
        connection.close()
        texts: list[str] = []
        for row in rows:
            texts.append(row[0])
        return "\n".join(texts)
    except sqlite3.DatabaseError:
        return database_path.read_text(encoding="utf-8")


class TestAnkiExporter(unittest.TestCase):
    def setUp(self) -> None:
        if not GENANKI_AVAILABLE:
            self.skipTest("genanki not installed")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.output_path = self.work_dir / "test_deck.apkg"

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def test_create_anki_deck_file(self) -> None:
        flashcards = [
            {"question": "What is the time complexity of QuickSort average case?", "answer": "O(n log n)"},
            {"question": "What data structure is used in Dijkstra's algorithm?", "answer": "Min-priority queue"}
        ]
        count = create_anki_deck("Algorithms Test Deck", flashcards, self.output_path)
        self.assertEqual(count, 2)
        self.assertTrue(self.output_path.exists())
        self.assertTrue(zipfile.is_zipfile(self.output_path))

    def test_deck_id_is_stable_across_processes(self) -> None:
        code = (
            "import sys; sys.path.insert(0, '.');"
            "from lecturelens.exports.anki_exporter import compute_stable_deck_id;"
            "print(compute_stable_deck_id('LectureLens algos'))"
        )
        import os
        environment = dict(os.environ)
        outputs = set()
        for seed in ("1", "2", "3"):
            environment["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=environment
            )
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1)
        self.assertEqual(int(outputs.pop()), compute_stable_deck_id("LectureLens algos"))

    def test_deck_id_range_and_distinct_names(self) -> None:
        first = compute_stable_deck_id("a")
        second = compute_stable_deck_id("b")
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(first, 1 << 30)
        self.assertLess(first, 1 << 31)

    def test_html_is_escaped_and_duplicates_dropped(self) -> None:
        flashcards = [
            {"question": "Is a < b & c > d?", "answer": "<b>yes</b>"},
            {"question": "Is a < b & c > d?", "answer": "duplicate"},
            {"question": "", "answer": "no question"},
            {"question": "no answer", "answer": "  "}
        ]
        count = create_anki_deck("Escape deck", flashcards, self.output_path)
        self.assertEqual(count, 1)
        fields = read_note_fields(self.output_path, self.work_dir)
        self.assertIn("&lt;", fields)
        self.assertNotIn("<b>yes</b>", fields)
        self.assertNotIn("duplicate", fields)

    def test_math_converted_to_anki_mathjax(self) -> None:
        flashcards = [
            {"question": "What is $e^{i\\pi} + 1$?", "answer": "It is $$0$$."}
        ]
        count = create_anki_deck("Math deck", flashcards, self.output_path)
        self.assertEqual(count, 1)
        fields = read_note_fields(self.output_path, self.work_dir)
        self.assertIn(r"\(e^{i\pi} + 1\)", fields)
        self.assertIn(r"\[0\]", fields)

    def test_currency_is_not_converted_to_math(self) -> None:
        flashcards = [
            {"question": "The item costs $5 and later $10.", "answer": "Total is $15."}
        ]
        count = create_anki_deck("Currency deck", flashcards, self.output_path)
        self.assertEqual(count, 1)
        fields = read_note_fields(self.output_path, self.work_dir)
        self.assertIn("$5", fields)
        self.assertIn("$10", fields)
        self.assertNotIn(r"\(5", fields)
        self.assertNotIn(r"\)", fields)


if __name__ == "__main__":
    unittest.main()
