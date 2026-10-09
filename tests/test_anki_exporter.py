import tempfile
import unittest
from pathlib import Path

try:
    import genanki
    from lecturelens.exports.anki_exporter import create_anki_deck
    GENANKI_AVAILABLE = True
except ImportError:
    GENANKI_AVAILABLE = False


class TestAnkiExporter(unittest.TestCase):
    def setUp(self) -> None:
        if not GENANKI_AVAILABLE:
            self.skipTest("genanki not installed")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name) / "test_deck.apkg"

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def test_create_anki_deck_file(self) -> None:
        flashcards = [
            {
                "question": "What is the time complexity of QuickSort average case?",
                "answer": "O(n log n)"
            },
            {
                "question": "What data structure is used in Dijkstra's algorithm?",
                "answer": "Min-priority queue (min-heap)"
            }
        ]

        create_anki_deck(
            deck_name="Algorithms Test Deck",
            flashcards=flashcards,
            output_file_path=self.output_path
        )

        self.assertTrue(self.output_path.exists())
        file_size = self.output_path.stat().st_size
        self.assertGreater(file_size, 0)


if __name__ == "__main__":
    unittest.main()
