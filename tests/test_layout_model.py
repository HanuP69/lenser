"""The parts of the screen, and what is done with them.

The model that finds the parts of a screen is a file that is downloaded,
so the tests here do not need it. They check the parts of the code that
decide what a part means and what is done with it.
"""

import unittest

from lecturelens.core.layout_model import (
    KIND_BY_LABEL,
    KIND_CHROME,
    KIND_CODE,
    KIND_FIGURE,
    KIND_FIGURE_TITLE,
    KIND_FORMULA,
    KIND_TEXT,
    KIND_TITLE,
    Region,
)


class TestTheMeaningOfTheParts(unittest.TestCase):
    def test_a_picture_is_a_figure(self) -> None:
        self.assertEqual(KIND_BY_LABEL["image"], KIND_FIGURE)
        self.assertEqual(KIND_BY_LABEL["chart"], KIND_FIGURE)

    def test_the_window_around_the_lecture_is_chrome(self) -> None:
        for label in ("header", "header_image", "footer", "footer_image"):
            self.assertEqual(KIND_BY_LABEL[label], KIND_CHROME, label)

    def test_the_label_under_a_picture_is_not_the_title_of_the_slide(self) -> None:
        """A caption that is the label of a picture is still a caption,
        but it must not be mistaken for the title of the slide."""
        self.assertEqual(KIND_BY_LABEL["figure_title"], KIND_FIGURE_TITLE)
        self.assertNotEqual(KIND_BY_LABEL["figure_title"], KIND_TITLE)

    def test_the_title_of_the_slide_is_a_title(self) -> None:
        self.assertEqual(KIND_BY_LABEL["paragraph_title"], KIND_TITLE)
        self.assertEqual(KIND_BY_LABEL["doc_title"], KIND_TITLE)

    def test_the_words_of_the_slide_are_text(self) -> None:
        self.assertEqual(KIND_BY_LABEL["text"], KIND_TEXT)
        self.assertEqual(KIND_BY_LABEL["table"], KIND_TEXT)

    def test_a_formula_is_a_formula(self) -> None:
        self.assertEqual(KIND_BY_LABEL["display_formula"], KIND_FORMULA)
        self.assertEqual(KIND_BY_LABEL["inline_formula"], KIND_FORMULA)

    def test_code_is_code(self) -> None:
        self.assertEqual(KIND_BY_LABEL["algorithm"], KIND_CODE)

    def test_a_part_that_is_not_known_is_other(self) -> None:
        self.assertNotIn("something_new", KIND_BY_LABEL)


class TestRegion(unittest.TestCase):
    def test_the_size_of_a_part(self) -> None:
        region = Region(kind=KIND_FIGURE, box=(10, 20, 110, 220), score=0.9, label="image")
        self.assertEqual(region.width, 100)
        self.assertEqual(region.height, 200)
        self.assertEqual(region.area, 20000)

    def test_a_part_with_no_size_has_no_area(self) -> None:
        region = Region(kind=KIND_FIGURE, box=(10, 20, 10, 20), score=0.1, label="image")
        self.assertEqual(region.area, 0)


if __name__ == "__main__":
    unittest.main()
