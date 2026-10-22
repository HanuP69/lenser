"""The picture of a diagram, from the screen to the notes.

A description of a diagram teaches nothing, so the picture itself is
kept. The tests here follow it on its way: it is cut out of the screen,
it is named, the notes point at it, and the page of the notes has it
inside it.
"""

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image, ImageDraw

from lecturelens.core.layout_model import (
    KIND_CHROME,
    KIND_FIGURE,
    KIND_FIGURE_TITLE,
    KIND_TEXT,
    KIND_TITLE,
    Region,
)
from lecturelens.core.local_ocr import read_text
from lecturelens.core.notes_html import render_notes_html


def make_png(size: tuple[int, int] = (400, 300)) -> bytes:
    image = Image.new("RGB", size, "white")
    drawer = ImageDraw.Draw(image)
    drawer.rectangle([40, 40, 360, 260], fill="navy")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    """A reader that always says the same thing."""

    def __init__(self, texts: list[str], boxes: list[list[list[float]]]) -> None:
        self.texts = texts
        self.boxes = boxes
        self.calls = 0

    def __call__(self, image: Any) -> Any:
        self.calls += 1

        class Output:
            pass

        output = Output()
        output.txts = self.texts
        output.scores = [0.99] * len(self.texts)
        output.boxes = self.boxes
        return output


class TestTheWindowIsLeftOut(unittest.TestCase):
    def test_the_text_of_the_window_is_not_in_the_text_of_the_slide(self) -> None:
        engine = FakeEngine(
            texts=["Neural Networks", "lecturelens-playback", "6.100L Lecture 1"],
            boxes=[
                [[40, 30], [360, 30], [360, 60], [40, 60]],
                [[0, 0], [200, 0], [200, 20], [0, 20]],
                [[100, 280], [300, 280], [300, 300], [100, 300]],
            ]
        )
        regions = [
            Region(kind=KIND_TITLE, box=(40, 30, 360, 60), score=0.9, label="paragraph_title"),
            Region(kind=KIND_CHROME, box=(0, 0, 200, 20), score=0.6, label="header"),
            Region(kind=KIND_CHROME, box=(100, 280, 300, 300), score=0.7, label="footer"),
        ]
        with mock.patch("lecturelens.core.local_ocr.get_engine", return_value=engine):
            result = read_text(Image.new("RGB", (400, 300), "white"), regions)
        self.assertIn("Neural Networks", result["text"])
        self.assertNotIn("lecturelens-playback", result["text"])
        self.assertNotIn("6.100L Lecture 1", result["text"])

    def test_the_whole_screen_is_read_when_the_parts_are_not_known(self) -> None:
        engine = FakeEngine(
            texts=["Neural Networks", "lecturelens-playback"],
            boxes=[
                [[40, 30], [360, 30], [360, 60], [40, 60]],
                [[0, 0], [200, 0], [200, 20], [0, 20]],
            ]
        )
        with mock.patch("lecturelens.core.local_ocr.get_engine", return_value=engine):
            result = read_text(Image.new("RGB", (400, 300), "white"))
        self.assertIn("lecturelens-playback", result["text"])


class TestThePictureIsCutOut(unittest.TestCase):
    def make_agent(self, folder: str) -> Any:
        from lecturelens.agents.screen_agent import ScreenAgent
        from lecturelens.config.settings import Settings
        from lecturelens.core.clock import SessionClock

        class FakeBus:
            async def publish_resilient(self, stream_name: str, event: Any) -> str:
                return "1-1"

        settings = Settings()
        settings.output_dir = Path(folder)
        settings.screen_layout_model = False
        settings.screen_local_ocr = False
        return ScreenAgent(settings, FakeBus(), SessionClock(), "figure_test", None)

    def test_the_biggest_picture_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_FIGURE, box=(10, 10, 60, 60), score=0.9, label="image"),
                Region(kind=KIND_FIGURE, box=(100, 100, 300, 250), score=0.8, label="image"),
            ]
            path = agent.save_the_figure(make_png(), regions, 7)
            self.assertTrue(path)
            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).name, "figure_0007.png")
            # The picture is the big one, not the small one.
            with Image.open(path) as saved:
                self.assertEqual(saved.size, (200, 150))

    def test_a_screen_without_a_picture_saves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_TEXT, box=(10, 10, 60, 60), score=0.9, label="text"),
            ]
            self.assertEqual(agent.save_the_figure(make_png(), regions, 1), "")

    def test_a_picture_that_is_too_small_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_FIGURE, box=(10, 10, 60, 60), score=0.9, label="image"),
            ]
            self.assertEqual(agent.save_the_figure(make_png(), regions, 1), "")

    def test_the_caption_is_the_title_of_the_slide(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_TITLE, box=(40, 30, 360, 60), score=0.9, label="paragraph_title"),
                Region(kind=KIND_FIGURE, box=(40, 90, 360, 260), score=0.8, label="image"),
            ]
            lines = [
                {"text": "Neural Networks", "box": [40, 30, 360, 60]},
                {"text": "a picture of a neuron", "box": [40, 90, 360, 110]},
            ]
            self.assertEqual(agent.caption_of_the_figure(regions, lines), "Neural Networks")

    def test_a_slide_without_a_title_has_no_caption(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_FIGURE, box=(40, 90, 360, 260), score=0.8, label="image"),
            ]
            self.assertEqual(agent.caption_of_the_figure(regions, []), "")

    def test_a_slide_without_a_title_uses_its_own_words(self) -> None:
        """A caption that is made up teaches the reader nothing, so the
        words of the slide are used when the slide has no title."""
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            regions = [
                Region(kind=KIND_FIGURE, box=(40, 90, 360, 260), score=0.8, label="image"),
            ]
            caption = agent.caption_of_the_figure(
                regions, [], "A picture of a neuron and its parts"
            )
            self.assertEqual(caption, "A picture of a neuron and its parts")


class TestTheNotesPointAtThePicture(unittest.TestCase):
    def make_agent(self, folder: str) -> Any:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.config.settings import Settings

        class FakeRouter:
            async def ainvoke(self, messages: Any) -> Any:
                raise AssertionError("the model must not be asked")

        settings = Settings()
        settings.output_dir = Path(folder)
        return StructuringAgent(
            settings=settings,
            session_id="figure_test",
            llm_router=FakeRouter()
        )

    def test_the_picture_of_a_slide_is_in_the_notes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            figure_path = str(agent.session_dir / "assets" / "figure_0001.png")
            entries = [
                {
                    "source_type": "slide",
                    "timestamp_epoch": 100.0,
                    "metadata": {
                        "figure_file_path": figure_path,
                        "figure_caption": "A picture of a neuron",
                    },
                }
            ]
            lines = agent.collect_figure_references(entries)
            self.assertEqual(len(lines), 2)
            self.assertIn("![A picture of a neuron](assets/figure_0001.png)", lines[0])

    def test_a_slide_without_a_picture_adds_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            entries = [
                {
                    "source_type": "slide",
                    "timestamp_epoch": 100.0,
                    "metadata": {},
                }
            ]
            self.assertEqual(agent.collect_figure_references(entries), [])

    def test_the_same_picture_is_only_once_in_the_notes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            path = str(Path(folder) / "assets" / "figure_0001.png")
            entries = [
                {"source_type": "slide", "timestamp_epoch": 100.0, "metadata": {"figure_file_path": path}},
                {"source_type": "slide", "timestamp_epoch": 200.0, "metadata": {"figure_file_path": path}},
            ]
            self.assertEqual(len(agent.collect_figure_references(entries)), 1)


class TestThePageHasThePictureInsideIt(unittest.TestCase):
    def test_the_picture_is_put_inside_the_page(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder) / "assets"
            assets.mkdir(parents=True)
            Image.new("RGB", (20, 20), "red").save(assets / "figure_0001.png")
            markdown = "### 00:01:00\n\n![A neuron](assets/figure_0001.png)\n"
            page = render_notes_html(markdown, base_dir=Path(folder))
            self.assertIn("data:image/png;base64,", page)
            self.assertNotIn('src="assets/figure_0001.png"', page)

    def test_a_picture_that_is_not_there_leaves_the_link_alone(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            markdown = "![A neuron](assets/figure_9999.png)\n"
            page = render_notes_html(markdown, base_dir=Path(folder))
            self.assertIn('src="assets/figure_9999.png"', page)

    def test_a_picture_of_the_web_is_not_read_from_the_disk(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            markdown = "![A neuron](https://example.invalid/figure.png)\n"
            page = render_notes_html(markdown, base_dir=Path(folder))
            self.assertIn('src="https://example.invalid/figure.png"', page)


class TestTheLabelOfAPicture(unittest.TestCase):
    def test_the_label_of_a_picture_is_used_when_there_is_no_title(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.agents.screen_agent import ScreenAgent
            from lecturelens.config.settings import Settings
            from lecturelens.core.clock import SessionClock

            class FakeBus:
                async def publish_resilient(self, stream_name: str, event: Any) -> str:
                    return "1-1"

            settings = Settings()
            settings.output_dir = Path(folder)
            settings.screen_layout_model = False
            settings.screen_local_ocr = False
            agent = ScreenAgent(settings, FakeBus(), SessionClock(), "figure_test", None)
            regions = [
                Region(kind=KIND_FIGURE_TITLE, box=(40, 260, 360, 280), score=0.8, label="figure_title"),
            ]
            lines = [{"text": "Feature points selected", "box": [40, 260, 360, 280]}]
            self.assertEqual(
                agent.caption_of_the_figure(regions, lines),
                "Feature points selected"
            )


if __name__ == "__main__":
    unittest.main()
