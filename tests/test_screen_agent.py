import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from lecturelens.agents import screen_agent
from lecturelens.agents.screen_agent import ScreenAgent, build_thumbnail_from_png
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.image_utils import (
    SlideChangeGate,
    compute_changed_fraction,
    compute_difference_hash,
    hamming_distance
)


def make_slide_png(lines: list[str], size: tuple[int, int] = (1280, 720)) -> bytes:
    image = Image.new("RGB", size, "white")
    drawer = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        drawer.rectangle([80, 100 + index * 90, 80 + len(line) * 22, 140 + index * 90], fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def thumb(lines: list[str]) -> np.ndarray:
    return build_thumbnail_from_png(make_slide_png(lines), 640)


class TestImageUtils(unittest.TestCase):
    def test_changed_fraction_ignores_tiny_cursor_but_sees_new_bullet(self) -> None:
        base = thumb(["Title line", "Bullet one"])
        cursor = base.copy()
        cursor[100:110, 200:208] = 0
        with_bullet = thumb(["Title line", "Bullet one", "A brand new bullet"])
        self.assertLess(compute_changed_fraction(base, cursor), 0.004)
        self.assertGreater(compute_changed_fraction(base, with_bullet), 0.004)

    def test_shape_mismatch_counts_as_changed(self) -> None:
        self.assertEqual(compute_changed_fraction(np.zeros((5, 5)), np.zeros((6, 6))), 1.0)

    def test_hash_distance(self) -> None:
        first = thumb(["Alpha", "Beta"])
        second = thumb(["Alpha", "Beta"])
        different = thumb(["Totally different", "Content here", "More", "Lines", "Added"])
        self.assertEqual(hamming_distance(compute_difference_hash(first), compute_difference_hash(second)), 0)
        self.assertGreater(hamming_distance(compute_difference_hash(first), compute_difference_hash(different)), 3)


class TestSlideChangeGate(unittest.TestCase):
    def make_gate(self) -> SlideChangeGate:
        return SlideChangeGate(change_threshold=0.004, pixel_delta=25, min_interval_seconds=5.0)

    def test_first_frame_is_accepted_and_static_screen_is_not(self) -> None:
        gate = self.make_gate()
        frame = thumb(["Intro"])
        self.assertTrue(gate.evaluate(frame, 0.0))
        for second in range(1, 30):
            self.assertFalse(gate.evaluate(frame, float(second)))

    def test_slide_change_is_accepted_once_settled(self) -> None:
        gate = self.make_gate()
        first = thumb(["Intro"])
        second = thumb(["Intro", "Next point", "More"])
        gate.evaluate(first, 0.0)
        self.assertFalse(gate.evaluate(second, 6.0))
        self.assertTrue(gate.evaluate(second, 7.0))

    def test_min_interval_throttles_rapid_changes(self) -> None:
        gate = self.make_gate()
        gate.evaluate(thumb(["A"]), 0.0)
        accepted = 0
        for index in range(1, 12):
            lines = []
            for n in range(index % 5 + 1):
                lines.append(f"line {n}")
            frame = thumb(lines + ["x" * (index * 3)])
            if gate.evaluate(frame, index * 0.5):
                accepted += 1
        self.assertLessEqual(accepted, 1)

    def test_continuous_video_still_gets_captured_after_settle_limit(self) -> None:
        gate = self.make_gate()
        gate.evaluate(thumb(["start"]), 0.0)
        accepted = 0
        for index in range(1, 40):
            frame = thumb(["video"] + ["frame"] * (index % 6 + 1) + ["y" * index])
            if gate.evaluate(frame, float(index)):
                accepted += 1
        self.assertGreaterEqual(accepted, 1)
        self.assertLessEqual(accepted, 8)


class FakeBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append(event)
        return "1-1"


def make_agent(temp_dir: str, router: Any = None) -> tuple[ScreenAgent, FakeBus]:
    settings = Settings()
    settings.output_dir = Path(temp_dir)
    bus = FakeBus()
    return ScreenAgent(settings, bus, SessionClock(), "screen_test", router), bus


class TestScreenAgent(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_saves_before_ocr_and_numbers_slides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir)
            seen_at_ocr: list[bool] = []

            async def fake_pool(function: Any, png_bytes: bytes, tesseract_path: str) -> str:
                seen_at_ocr.append(len(list(agent.slides_dir.glob("slide_*.png"))) == 1)
                return "Binary Search"

            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=fake_pool):
                event = await agent.ingest_png_bytes(make_slide_png(["Binary Search"]))
            self.assertEqual(event.slide_number, 1)
            self.assertEqual(event.extracted_text, "Binary Search")
            self.assertEqual(seen_at_ocr, [True])
            self.assertEqual(len(bus.published), 1)

    async def test_revisited_slide_reuses_text_and_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir)
            ocr_calls: list[int] = []

            async def fake_pool(function: Any, png_bytes: bytes, tesseract_path: str) -> str:
                ocr_calls.append(1)
                return "Merge sort"

            first_png = make_slide_png(["Merge sort", "divide"])
            other_png = make_slide_png(["Quick sort", "pivot", "partition", "recurse"])
            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=fake_pool):
                first = await agent.ingest_png_bytes(first_png)
                await agent.ingest_png_bytes(other_png)
                again = await agent.ingest_png_bytes(first_png)
            self.assertEqual(first.revisited_slide_number, 0)
            self.assertEqual(again.revisited_slide_number, 1)
            self.assertEqual(again.extracted_text, "Merge sort")
            self.assertEqual(len(ocr_calls), 2)

    async def test_numbering_continues_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = make_agent(temp_dir)
            (agent.slides_dir / "slide_0009.png").write_bytes(b"x")
            restarted, _ = make_agent(temp_dir)
            self.assertEqual(restarted.slide_counter, 9)

    async def test_empty_ocr_falls_back_to_vision_router(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class Router:
                def __init__(self) -> None:
                    self.calls = 0

                async def ainvoke(self, messages: Any) -> Any:
                    self.calls += 1
                    from langchain_core.messages import AIMessage
                    return AIMessage(content="A diagram of a tree")

            router = Router()
            agent, _ = make_agent(temp_dir, router)

            async def empty_ocr(function: Any, png_bytes: bytes, tesseract_path: str) -> str:
                return ""

            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=empty_ocr):
                event = await agent.ingest_png_bytes(make_slide_png(["tree"]))
            self.assertEqual(event.extracted_text, "A diagram of a tree")
            self.assertEqual(router.calls, 1)

    async def test_ocr_failure_does_not_call_docker_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = make_agent(temp_dir)

            async def failing_ocr(function: Any, png_bytes: bytes, tesseract_path: str) -> str:
                raise RuntimeError("tesseract missing")

            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=failing_ocr):
                with mock.patch.object(screen_agent, "run_docker_ocr_request") as docker_call:
                    event = await agent.ingest_png_bytes(make_slide_png(["x"]))
            self.assertEqual(event.extracted_text, "")
            docker_call.assert_not_called()

    async def test_docker_ocr_used_through_thread_pool_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = make_agent(temp_dir)
            agent.settings.docker_ocr_url = "http://localhost:9999/ocr"

            async def failing_ocr(function: Any, png_bytes: bytes, tesseract_path: str) -> str:
                raise RuntimeError("tesseract missing")

            with mock.patch.object(screen_agent, "run_in_process_pool", side_effect=failing_ocr):
                with mock.patch.object(screen_agent, "run_docker_ocr_request", return_value="from docker"):
                    event = await agent.ingest_png_bytes(make_slide_png(["x"]))
            self.assertEqual(event.extracted_text, "from docker")


class TestScreenHelpers(unittest.TestCase):
    def test_encode_frame_round_trip(self) -> None:
        from lecturelens.agents.screen_agent import ScreenFrame, encode_frame_as_png
        raw = bytes([10, 20, 30, 255] * (4 * 3))
        frame = ScreenFrame(raw_bgra=raw, width=4, height=3, thumbnail=np.zeros((3, 4), dtype=np.uint8))
        decoded = Image.open(io.BytesIO(encode_frame_as_png(frame)))
        self.assertEqual(decoded.size, (4, 3))
        self.assertEqual(decoded.getpixel((0, 0)), (30, 20, 10))


if __name__ == "__main__":
    unittest.main()
