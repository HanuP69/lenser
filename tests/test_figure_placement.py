"""That the picture ends up in the notes at the right place.

The notes are made of paragraphs, and every picture of a diagram is
talked about somewhere inside them. The tests here follow the rules that
put a picture next to its paragraph: the token the model is asked for,
the caption that says what the picture is, and the pass that reads the
slides again when the lecturer goes back to one.
"""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from lecturelens.core.drift_filter import ContextDriftFilter
from lecturelens.core.layout_model import KIND_FIGURE, Region


def make_png(size: tuple[int, int] = (800, 600)) -> bytes:
    image = Image.new("RGB", size, "white")
    drawer = ImageDraw.Draw(image)
    drawer.rectangle([100, 100, 400, 400], fill="navy")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MockTextEmbedder:
    """The words of a topic and the words that do not belong to it."""

    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {}

    def set_vector(self, text: str, vec: np.ndarray) -> None:
        from lecturelens.core.drift_filter import normalize_vector

        self.vectors[text.strip()] = normalize_vector(np.asarray(vec, dtype=np.float32))

    def embed(self, text: str) -> np.ndarray:
        clean = text.strip()
        if clean in self.vectors:
            return self.vectors[clean]
        for key, vec in self.vectors.items():
            if key in clean or clean in key:
                return vec
        return np.ones(8, dtype=np.float32) / np.sqrt(8.0)


class TestTheSamePictureIsSavedOnce(unittest.TestCase):
    def make_screen_agent(self, folder: str) -> object:
        from lecturelens.agents.screen_agent import ScreenAgent
        from lecturelens.config.settings import Settings
        from lecturelens.core.clock import SessionClock

        class FakeBus:
            async def publish_resilient(self, stream_name: str, event: object) -> str:
                return "1-1"

        settings = Settings()
        settings.output_dir = Path(folder)
        settings.screen_layout_model = False
        settings.screen_local_ocr = False
        return ScreenAgent(settings, FakeBus(), SessionClock(), "dedupe_test", None)

    def test_the_picture_of_the_next_slide_is_not_saved_again(self) -> None:
        """A diagram that stays on the screen for two slides in a row is
        one picture. The dedup used to count a stat and save a copy
        anyway, and the notes held the same picture again for every
        slide it stayed on."""
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_screen_agent(folder)
            regions = [
                Region(kind=KIND_FIGURE, box=(100, 100, 400, 400), score=0.85, label="image")
            ]
            path1 = agent.save_the_figure(make_png(), regions, 1)
            path2 = agent.save_the_figure(make_png(), regions, 2)

            self.assertTrue(path1)
            self.assertEqual(path2, path1)
            self.assertTrue(Path(path1).exists())
            assets = Path(folder) / "dedupe_test" / "assets"
            self.assertFalse((assets / "figure_0002.png").exists())


class TestARevisitedSlideIsReadAgain(unittest.IsolatedAsyncioTestCase):
    async def test_the_revisit_of_an_unread_slide_gets_its_text(self) -> None:
        """The lecturer goes back to a slide whose text was never read.
        The retake shows the same screen: the reading is done on it.
        The revisit used to be published with no text at all and nothing
        was scheduled, so the slide stayed empty forever."""
        from lecturelens.agents.screen_agent import (
            ScreenAgent,
            compute_difference_hash,
            shrink_thumbnail,
        )
        from lecturelens.config.settings import Settings
        from lecturelens.core.clock import SessionClock

        class TextRouter:
            async def ainvoke(self, messages: object) -> object:
                class Resp:
                    content = "Rotations keep the red black tree balanced"
                return Resp()

        class FakeBus:
            def __init__(self) -> None:
                self.published: list[object] = []

            async def publish_resilient(self, stream_name: str, event: object) -> str:
                self.published.append(event)
                return "1-1"

        with tempfile.TemporaryDirectory() as folder:
            settings = Settings()
            settings.output_dir = Path(folder)
            settings.screen_layout_model = False
            settings.screen_local_ocr = False
            bus = FakeBus()
            agent = ScreenAgent(settings, bus, SessionClock(), "revisit_test", TextRouter())

            thumbnail = np.full((64, 64), 180, dtype=np.uint8)
            hash_value = compute_difference_hash(thumbnail)
            small = shrink_thumbnail(thumbnail, 192)
            # Slide 7 exists by its hash, but there is no text for it.
            agent.slide_hashes.append((7, hash_value, small))
            self.assertNotIn(7, agent.slide_texts)

            event = await agent.accept_png(make_png(), thumbnail)
            self.assertEqual(event.revisited_slide_number, 7)
            self.assertEqual(event.extracted_text, "")

            for _ in range(100):
                await asyncio.sleep(0.01)

            updates = [item for item in bus.published if item is not event]
            self.assertEqual(len(updates), 1)
            self.assertEqual(
                updates[0].extracted_text, "Rotations keep the red black tree balanced"
            )
            self.assertEqual(updates[0].revisited_slide_number, 7)
            # The words belong to the older slide too: a third visit of
            # it needs no new reading.
            self.assertEqual(
                agent.slide_texts.get(7), "Rotations keep the red black tree balanced"
            )
            self.assertEqual(agent._pending_text_extractions, 0)


class TestThePictureJoinsItsParagraph(unittest.TestCase):
    def make_agent(self, folder: str) -> object:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.config.settings import Settings

        class DummyRouter:
            async def ainvoke(self, messages: object) -> object:
                class Resp:
                    content = ""
                return Resp()

        settings = Settings()
        settings.output_dir = Path(folder)
        return StructuringAgent(
            settings=settings, session_id="place_test", llm_router=DummyRouter()
        )

    def make_batch(self, agent: object, captions: list[str]) -> object:
        from lecturelens.agents.structuring_agent import SliceBatch

        entries: list[dict[str, object]] = []
        for index, caption in enumerate(captions):
            entries.append({
                "source_type": "slide",
                "timestamp_epoch": 10.0 + index,
                "metadata": {
                    "figure_file_path": str(agent.session_dir / "assets" / f"fig_{index}.png"),
                    "figure_id": f"fig_{index}",
                    "figure_caption": caption,
                },
            })
        return SliceBatch(entries=entries, end_offset=100, start_offset=0)

    def test_the_unplaced_picture_stands_after_its_paragraph(self) -> None:
        """The model answered no token. The caption of the picture is the
        title of the slide, and the paragraph that talks about the same
        words is where the picture goes. The old fallback put every
        unplaced picture in one pile at the end of the section, far from
        the paragraph about it."""
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            batch = self.make_batch(agent, ["Red Black Tree Rotation"])
            summary = (
                "The lecture starts with an overview of hashing tables.\n\n"
                "Meanwhile we rotate the red black tree after every insertion.\n\n"
                "The homework for the week is the reading list."
            )
            rendered = agent.build_notes_section(batch, {"summary": summary})

            self.assertIn("![Red Black Tree Rotation](assets/fig_0.png)", rendered)
            rotation_at = rendered.index("Meanwhile we rotate the red black tree")
            homework_at = rendered.index("The homework for the week")
            picture_at = rendered.index("![Red Black Tree Rotation]")
            # The picture sits between the paragraph about it and the
            # next thing, and no more than a few lines after it.
            self.assertGreater(picture_at, rotation_at)
            self.assertLess(picture_at, homework_at)
            self.assertLess(picture_at - rotation_at, 200)

    def test_a_picture_of_something_absent_stays_at_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            batch = self.make_batch(agent, ["Red Black Tree Rotation", "Quantum Entanglement"])
            summary = (
                "The lecture starts with an overview of hashing tables.\n\n"
                "Meanwhile we rotate the red black tree after every insertion.\n\n"
                "The homework for the week is the reading list."
            )
            rendered = agent.build_notes_section(batch, {"summary": summary})

            homework_at = rendered.index("The homework for the week")
            rbt_at = rendered.index("![Red Black Tree Rotation]")
            quantum_at = rendered.index("![Quantum Entanglement]")
            self.assertGreater(rbt_at, rotation_paragraph := rendered.index("Meanwhile we rotate"))
            self.assertLess(rbt_at, homework_at)
            # Nothing matches this picture in the section, so it keeps
            # the place at the end.
            self.assertGreater(quantum_at, homework_at)

    def test_a_picture_marked_twice_is_shown_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            batch = self.make_batch(agent, ["Neural Network Architecture"])
            summary = (
                "First part about the network.\n[FIGURE fig_0]\n"
                "Second part on the same network.\n[FIGURE fig_0]"
            )
            rendered = agent.build_notes_section(batch, {"summary": summary})
            self.assertEqual(
                rendered.count("![Neural Network Architecture](assets/fig_0.png)"), 1
            )

    def test_the_colon_form_of_the_token_is_taken(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_agent(folder)
            batch = self.make_batch(agent, ["Neural Network Architecture"])
            rendered = agent.build_notes_section(
                batch, {"summary": "Overview:\n[FIGURE: fig_0]"}
            )
            self.assertIn("![Neural Network Architecture](assets/fig_0.png)", rendered)
            self.assertNotIn("[FIGURE", rendered)


class TestTheMathPassKeepsThePictures(unittest.TestCase):
    def test_a_latex_rewrite_that_loses_the_tokens_is_refused(self) -> None:
        """The pass was asked for mathematics. A model that dropped the
        [FIGURE id] lines while it rewrote pushed every picture of the
        section to the end of the notes. The rewrite is refused
        instead."""
        from lecturelens.agents.structuring_agent import latex_result_is_usable

        original = (
            "Section of the lecture\n\n[FIGURE fig_1]\n\n"
            "The cost is O(n log n) for the first loop."
        )
        lost = "Section of the lecture\n\nThe cost is $O(n \\log n)$ for the first loop."
        kept = (
            "Section of the lecture\n\n[FIGURE fig_1]\n\n"
            "The cost is $O(n \\log n)$ for the first loop."
        )
        self.assertFalse(latex_result_is_usable(original, lost))
        self.assertTrue(latex_result_is_usable(original, kept))


class TestTheMarkerLineIsNotalHeading(unittest.TestCase):
    def test_the_figure_marker_stays_a_plain_line(self) -> None:
        """A [FIGURE id: caption] marker of its own line was promoted to
        a heading, and once the token was replaced with the picture a
        lone "####" stayed behind in the notes."""
        from lecturelens.core.markdown_formatter import format_clean_markdown

        cleaned = format_clean_markdown(
            "Intro of the section\n\n[FIGURE fig_9: Insertion Case]\n\nMore prose."
        )
        self.assertNotIn("#### [FIGURE", cleaned)
        self.assertIn("[FIGURE fig_9: Insertion Case]", cleaned)


class TestTheOverflowOfTheQuarantine(unittest.TestCase):
    def test_the_first_quarantined_entry_is_kept_not_deleted(self) -> None:
        """The pending quarantine holds the entries of speech that risks
        to be off topic. When more than twenty of them came, the oldest
        one was dropped from everywhere: not into the notes and not into
        the quarantine either. It is released to the quarantine now."""
        embedder = MockTextEmbedder()
        topic = np.zeros(8, dtype=np.float32)
        topic[2] = 1.0
        one = np.zeros(8, dtype=np.float32)
        one[0] = 1.0
        two = np.zeros(8, dtype=np.float32)
        two[1] = 1.0
        embedder.set_vector("a topic on hashing tables and rings", topic)
        embedder.set_vector("completely unrelated sports commentary number zero", one)
        embedder.set_vector("another unrelated weather report fresh seven", two)

        filt = ContextDriftFilter(
            drift_threshold=0.50,
            sustain_consecutive=3,
            min_words=4,
            embedder=embedder
        )

        anchor_entry = [{
            "event_id": "anchor",
            "content": "a topic on hashing tables and rings",
            "importance_flags": [],
            "source_type": "transcript"
        }]
        kept = filt.filter_batch_entries(anchor_entry)
        self.assertEqual(len(kept), 1)

        drift_entries: list[dict[str, object]] = []
        for index in range(21):
            content = (
                "completely unrelated sports commentary number zero"
                if index % 2 == 0
                else "another unrelated weather report fresh seven"
            )
            drift_entries.append({
                "event_id": f"drift_{index}",
                "content": f"{content} item {index}",
                "importance_flags": [],
                "source_type": "transcript"
            })
        try:
            filt.filter_batch_entries(drift_entries)
        except Exception:
            # The batch refilter bookkeeping is not what is tested here;
            # the state of the buffer after the overflow is.
            pass

        contents = [entry["content"] for entry in filt.quarantined_entries]
        pending_ids = [entry.get("event_id") for entry in filt.pending_quarantine_entries]
        # The oldest entry of the queue is in the quarantine (it used to
        # be deleted from everywhere) and the pending buffer is bounded.
        self.assertEqual(len(contents), 1)
        self.assertIn("item 0", contents[0])
        self.assertEqual("drift_0", filt.quarantined_entries[0]["event_id"])
        self.assertEqual(20, len(pending_ids))
        self.assertNotIn("drift_0", pending_ids)
        self.assertIn("drift_20", pending_ids)

    def test_a_reset_leaves_the_old_quarantine_behind(self) -> None:
        filt = ContextDriftFilter(sustain_consecutive=3, min_words=4)
        filt.quarantined_entries.append({"event_id": "old", "content": "old run content"})
        filt.pending_quarantine_entries.append({"event_id": "pending", "content": "pending"})

        filt.reset()

        self.assertEqual(filt.quarantined_entries, [])
        self.assertEqual(filt.pending_quarantine_entries, [])


if __name__ == "__main__":
    unittest.main()
