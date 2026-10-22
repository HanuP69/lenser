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

import numpy as np
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


class TestFigurePipelineEnhancements(unittest.IsolatedAsyncioTestCase):
    def make_screen_agent(self, folder: str, router: Any = None) -> Any:
        from lecturelens.agents.screen_agent import ScreenAgent
        from lecturelens.config.settings import Settings
        from lecturelens.core.clock import SessionClock

        class FakeBus:
            def __init__(self) -> None:
                self.published: list[Any] = []

            async def publish_resilient(self, stream_name: str, event: Any) -> str:
                self.published.append(event)
                return "1-1"

            async def publish(self, stream_name: str, event: Any) -> str:
                self.published.append(event)
                return "1-1"

        settings = Settings()
        settings.output_dir = Path(folder)
        settings.screen_layout_model = True
        settings.screen_local_ocr = False
        return ScreenAgent(settings, FakeBus(), SessionClock(), "fig_enhance_test", router)

    async def test_layout_on_full_frame_vs_crop(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_screen_agent(folder)
            full_frame = make_png((800, 600))
            crop_frame = make_png((200, 100))

            full_regions = [
                Region(kind=KIND_FIGURE, box=(100, 100, 400, 400), score=0.85, label="image")
            ]
            crop_regions = [
                Region(kind=KIND_TEXT, box=(10, 10, 50, 50), score=0.9, label="text")
            ]

            async def mock_find_parts(png_bytes: bytes) -> list[Any]:
                if len(png_bytes) == len(full_frame):
                    return full_regions
                return crop_regions

            agent.find_the_parts = mock_find_parts
            event = await agent.accept_png(
                full_frame,
                np.zeros((64, 64), dtype=np.uint8),
                crop_png=crop_frame
            )
            self.assertTrue(event.figure_file_path)
            self.assertTrue(Path(event.figure_file_path).exists())

    def test_repeating_chrome_webcam_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_screen_agent(folder)
            frame_png = make_png((800, 600))
            webcam_region = [
                Region(kind=KIND_FIGURE, box=(600, 10, 780, 160), score=0.95, label="image")
            ]

            p1 = agent.save_the_figure(frame_png, webcam_region, 1)
            self.assertTrue(p1)
            p2 = agent.save_the_figure(frame_png, webcam_region, 2)
            self.assertTrue(p2)
            p3 = agent.save_the_figure(frame_png, webcam_region, 3)
            self.assertEqual(p3, "")

    def test_dedup_across_progressive_builds(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            agent = self.make_screen_agent(folder)
            frame1 = make_png((800, 600))
            frame2 = make_png((800, 600))
            fig_regions = [
                Region(kind=KIND_FIGURE, box=(100, 100, 400, 400), score=0.85, label="image")
            ]

            path1 = agent.save_the_figure(frame1, fig_regions, 1)
            self.assertTrue(path1)
            stat_before = agent.control.stats.get("slide_figure_deduped", 0)
            path2 = agent.save_the_figure(frame2, fig_regions, 2)
            self.assertTrue(path2)
            stat_after = agent.control.stats.get("slide_figure_deduped", 0)
            self.assertGreater(stat_after, stat_before)

    def test_token_placement_and_invalid_token_dropping(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.agents.structuring_agent import StructuringAgent, SliceBatch
            from lecturelens.config.settings import Settings

            class DummyRouter:
                async def ainvoke(self, messages: Any) -> Any:
                    return ""

            settings = Settings()
            settings.output_dir = Path(folder)
            agent = StructuringAgent(settings=settings, session_id="test_tok", llm_router=DummyRouter())

            fig_path = str(agent.session_dir / "assets" / "fig_100.png")
            entries = [
                {
                    "source_type": "slide",
                    "timestamp_epoch": 10.0,
                    "metadata": {
                        "figure_file_path": fig_path,
                        "figure_id": "fig_100",
                        "figure_caption": "Neural Network Architecture"
                    }
                }
            ]
            batch = SliceBatch(entries=entries, end_offset=100, start_offset=0)
            summary_with_tokens = (
                "Here is an overview of the network:\n"
                "[FIGURE fig_100]\n"
                "And here is an invalid token that should be dropped:\n"
                "[FIGURE hallucinated_id]"
            )
            result_state = {"summary": summary_with_tokens}
            rendered = agent.build_notes_section(batch, result_state)

            self.assertIn("![Neural Network Architecture](assets/fig_100.png)", rendered)
            self.assertNotIn("[FIGURE fig_100]", rendered)
            self.assertNotIn("hallucinated_id", rendered)

    def test_prompt_injection_in_figure_caption_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.agents.structuring_agent import StructuringAgent, SliceBatch
            from lecturelens.config.settings import Settings

            class DummyRouter:
                async def ainvoke(self, messages: Any) -> Any:
                    return ""

            settings = Settings()
            settings.output_dir = Path(folder)
            agent = StructuringAgent(settings=settings, session_id="test_inj", llm_router=DummyRouter())

            fig_path = str(agent.session_dir / "assets" / "fig_200.png")
            entries = [
                {
                    "source_type": "slide",
                    "timestamp_epoch": 10.0,
                    "metadata": {
                        "figure_file_path": fig_path,
                        "figure_id": "fig_200",
                        "figure_caption": "Evil Caption](http://attacker.com)\n\n# Injected Heading"
                    }
                }
            ]
            batch = SliceBatch(entries=entries, end_offset=100, start_offset=0)
            rendered = agent.build_notes_section(batch, {"summary": "Explaining the diagram:\n[FIGURE fig_200]"})

            self.assertNotIn("Injected Heading", rendered.splitlines()[0])
            self.assertIn("fig_200.png", rendered)
            self.assertNotIn("Evil Caption](http://attacker.com)", rendered)

    def test_unreferenced_figure_appended_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.agents.structuring_agent import StructuringAgent, SliceBatch
            from lecturelens.config.settings import Settings

            class DummyRouter:
                async def ainvoke(self, messages: Any) -> Any:
                    return ""

            settings = Settings()
            settings.output_dir = Path(folder)
            agent = StructuringAgent(settings=settings, session_id="test_fallback", llm_router=DummyRouter())

            fig_path = str(agent.session_dir / "assets" / "fig_300.png")
            entries = [
                {
                    "source_type": "slide",
                    "timestamp_epoch": 10.0,
                    "metadata": {
                        "figure_file_path": fig_path,
                        "figure_id": "fig_300",
                        "figure_caption": "Unreferenced Diagram"
                    }
                }
            ]
            batch = SliceBatch(entries=entries, end_offset=100, start_offset=0)
            rendered = agent.build_notes_section(batch, {"summary": "Note content without tokens."})
            self.assertIn("![Unreferenced Diagram](assets/fig_300.png)", rendered)

    def test_gateway_asset_serving_and_auth(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.config.settings import Settings
            from lecturelens.gateway.server import GatewayServer

            settings = Settings()
            settings.output_dir = Path(folder)
            settings.gateway_api_key = "secret_key_123"

            class DummyBus:
                pass

            gateway = GatewayServer(settings, DummyBus())
            session_id = "test_asset_sess"
            sess_assets = Path(folder) / session_id / "assets"
            sess_assets.mkdir(parents=True, exist_ok=True)
            (sess_assets / "diagram.png").write_bytes(b"PNG_BYTES_HERE")

            class DummyReq:
                def __init__(self, path: str, headers: dict[str, str]) -> None:
                    self.path = path
                    self.headers = headers

            req_unauth = DummyReq(f"/api/assets/{session_id}/diagram.png", {})
            resp_unauth = gateway.handle_http_request(None, req_unauth)
            self.assertIsNotNone(resp_unauth)
            self.assertEqual(resp_unauth.status_code, 401)

            req_ok = DummyReq(f"/api/assets/{session_id}/diagram.png", {"Authorization": "Bearer secret_key_123"})
            resp_ok = gateway.handle_http_request(None, req_ok)
            self.assertIsNotNone(resp_ok)
            self.assertEqual(resp_ok.status_code, 200)
            self.assertEqual(resp_ok.body, b"PNG_BYTES_HERE")

            req_query_ok = DummyReq(f"/api/assets/{session_id}/diagram.png?key=secret_key_123", {})
            resp_query_ok = gateway.handle_http_request(None, req_query_ok)
            self.assertIsNotNone(resp_query_ok)
            self.assertEqual(resp_query_ok.status_code, 200)

            req_missing = DummyReq(f"/api/assets/{session_id}/missing.png", {"Authorization": "Bearer secret_key_123"})
            resp_missing = gateway.handle_http_request(None, req_missing)
            self.assertIsNotNone(resp_missing)
            self.assertEqual(resp_missing.status_code, 404)

            req_traversal = DummyReq(f"/api/assets/{session_id}/..%2F..%2Fevil.png", {"Authorization": "Bearer secret_key_123"})
            resp_traversal = gateway.handle_http_request(None, req_traversal)
            self.assertIsNotNone(resp_traversal)
            self.assertIn(resp_traversal.status_code, (400, 404))

    async def test_student_client_hinglish_asset_link_protection(self) -> None:
        from lecturelens.client.student_client import PersonalizationEngine, StudentProfile
        from lecturelens.schemas.events import GeneralNoteSegment

        class HinglishRouter:
            async def ainvoke(self, messages: Any) -> Any:
                prompt_text = str(messages[0].content)
                token = "__LL_ASSET_LINK_0__"
                self.assertIn(token, prompt_text)
                class Msg:
                    content = f"Yeh lecture notes hain:\n{token}\nSamajh aaya sabko."
                return Msg()

        profile = StudentProfile(student_id="s1", preferred_language="Hinglish", detail_level="concise")
        engine = PersonalizationEngine(profile, HinglishRouter())

        segment = GeneralNoteSegment(
            session_id="s1",
            sequence_number=1,
            timestamp_epoch=100.0,
            start_time="00:01:40",
            end_time="00:02:40",
            heading="Section 1 (00:01:40)",
            content="Summary content:\n![Neuron Architecture](assets/figure_0001.png)\nDetails here.",
            concept_tags=["neural_network"]
        )

        rewritten = await engine.rewrite_segment(segment)
        self.assertIn("![Neuron Architecture](assets/figure_0001.png)", rewritten)
        self.assertNotIn("__LL_ASSET_LINK_0__", rewritten)

    async def test_crash_replay_republishing_without_resynthesis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.agents.structuring_agent import StructuringAgent, SliceBatch
            from lecturelens.config.settings import Settings

            class CountingRouter:
                def __init__(self) -> None:
                    self.calls = 0

                async def ainvoke(self, messages: Any) -> Any:
                    self.calls += 1
                    class Resp:
                        content = "Synthesized content."
                    return Resp()

            class MockBus:
                def __init__(self) -> None:
                    self.published: list[Any] = []

                async def publish_resilient(self, stream: str, msg: Any) -> str:
                    self.published.append(msg)
                    return "1-1"

            settings = Settings()
            settings.output_dir = Path(folder)
            router = CountingRouter()
            bus = MockBus()
            agent = StructuringAgent(settings=settings, session_id="replay_crash_test", llm_router=router, redis_bus=bus)

            agent.notes_path.write_text("Already written clean notes from earlier run.\n", encoding="utf-8")
            agent.save_notes_marker(200)

            entries = [{"source_type": "transcript", "timestamp_epoch": 10.0, "content": "hello"}]
            batch = SliceBatch(entries=entries, end_offset=200, start_offset=0)

            await agent.process_single_batch(batch, "")

            self.assertEqual(router.calls, 0)
            self.assertEqual(len(bus.published), 1)
            self.assertEqual(bus.published[0].content, "Already written clean notes from earlier run.\n")

    async def test_replay_kill_mid_stream_matches_clean_run(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as clean_dir, tempfile.TemporaryDirectory() as crash_dir:
            from lecturelens.agents.structuring_agent import StructuringAgent
            from lecturelens.config.settings import Settings

            clean_settings = Settings()
            clean_settings.output_dir = Path(clean_dir)
            clean_settings.max_slice_characters = 60

            class DeterministicRouter:
                async def ainvoke(self, messages: Any) -> Any:
                    content = str(messages[0].content)
                    topic = "Topic A" if "part1" in content else "Topic B"
                    class R:
                        content = f"Notes on {topic}."
                    return R()

            clean_agent = StructuringAgent(
                settings=clean_settings,
                session_id="lecture_clean",
                llm_router=DeterministicRouter()
            )

            lines = [
                json.dumps({"source_type": "transcript", "timestamp_epoch": 10.0, "content": "part1 content words here"}),
                json.dumps({"source_type": "transcript", "timestamp_epoch": 20.0, "content": "part1 more content here"}),
                json.dumps({"source_type": "transcript", "timestamp_epoch": 30.0, "content": "part2 content words here"}),
                json.dumps({"source_type": "transcript", "timestamp_epoch": 40.0, "content": "part2 more content here"})
            ]
            clean_agent.timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            await clean_agent.process_all_pending()
            clean_notes = clean_agent.notes_path.read_text(encoding="utf-8")
            self.assertTrue(clean_notes)

            crash_settings = Settings()
            crash_settings.output_dir = Path(crash_dir)
            crash_settings.max_slice_characters = 60

            crash_agent = StructuringAgent(
                settings=crash_settings,
                session_id="lecture_crash",
                llm_router=DeterministicRouter()
            )
            crash_agent.timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            batches = crash_agent.read_pending_batches()
            self.assertGreaterEqual(len(batches), 2)
            await crash_agent.process_single_batch(batches[0], "")

            recovered_agent = StructuringAgent(
                settings=crash_settings,
                session_id="lecture_crash",
                llm_router=DeterministicRouter()
            )
            await recovered_agent.process_all_pending()
            recovered_notes = recovered_agent.notes_path.read_text(encoding="utf-8")

            self.assertEqual(clean_notes, recovered_notes)


if __name__ == "__main__":
    unittest.main()
