import asyncio
import io
import os
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

    def test_switching_from_wall_clock_to_monotonic_does_not_freeze_the_gate(self) -> None:
        """A gateway frame used to feed a wall-clock epoch into the gate while
        the local path fed a monotonic value. The gate then computed a huge
        negative interval and rejected every later slide for the session."""
        gate = self.make_gate()
        first = thumb(["Intro"])
        second = thumb(["Intro", "Next point", "More"])

        # Wall-clock epoch first (the gateway path).
        self.assertTrue(gate.evaluate(first, 1_700_000_000.0))

        # Then the local monotonic path: the slide change must still be seen.
        accepted = False
        for offset in range(0, 40):
            if gate.evaluate(second, 3_000.0 + offset):
                accepted = True
                break
        self.assertTrue(accepted, "the gate must not freeze after a clock base change")

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


class MockVisionRouter:
    def __init__(self, response_text: str = "Extracted slide text") -> None:
        self.calls = 0
        self.response_text = response_text

    async def ainvoke(self, messages: Any) -> Any:
        self.calls += 1
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.response_text)


def make_agent(temp_dir: str, router: Any = None) -> tuple[ScreenAgent, FakeBus]:
    settings = Settings()
    settings.output_dir = Path(temp_dir)
    bus = FakeBus()
    return ScreenAgent(settings, bus, SessionClock(), "screen_test", router), bus


class TestScreenAgent(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_saves_before_ocr_and_numbers_slides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            router = MockVisionRouter("Binary Search")
            agent, bus = make_agent(temp_dir, router)
            event = await agent.ingest_png_bytes(make_slide_png(["Binary Search"]))
            self.assertEqual(event.slide_number, 1)
            self.assertEqual(event.extracted_text, "Binary Search")
            self.assertEqual(router.calls, 1)
            self.assertEqual(len(list(agent.slides_dir.glob("slide_*.png"))), 1)
            self.assertEqual(len(bus.published), 1)

    async def test_revisited_slide_reuses_text_and_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class DynamicRouter:
                def __init__(self) -> None:
                    self.calls = 0

                async def ainvoke(self, messages: Any) -> Any:
                    self.calls += 1
                    from langchain_core.messages import AIMessage
                    return AIMessage(content=f"Slide content call {self.calls}")

            router = DynamicRouter()
            agent, bus = make_agent(temp_dir, router)

            first_png = make_slide_png(["Merge sort", "divide"])
            other_png = make_slide_png(["Quick sort", "pivot", "partition", "recurse"])

            first = await agent.ingest_png_bytes(first_png)
            await agent.ingest_png_bytes(other_png)
            again = await agent.ingest_png_bytes(first_png)

            self.assertEqual(first.revisited_slide_number, 0)
            self.assertEqual(again.revisited_slide_number, 1)
            self.assertEqual(again.extracted_text, first.extracted_text)
            self.assertEqual(router.calls, 2)

    async def test_numbering_continues_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = make_agent(temp_dir)
            (agent.slides_dir / "slide_0009.png").write_bytes(b"x")
            restarted, _ = make_agent(temp_dir)
            self.assertEqual(restarted.slide_counter, 9)

    async def test_vision_fallback_retry_and_empty_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class FailingRouter:
                def __init__(self) -> None:
                    self.calls = 0

                async def ainvoke(self, messages: Any) -> Any:
                    self.calls += 1
                    raise RuntimeError("API timeout")

            router = FailingRouter()
            agent, bus = make_agent(temp_dir, router)
            event = await agent.ingest_png_bytes(make_slide_png(["test failure"]))
            self.assertEqual(event.extracted_text, "")
            self.assertEqual(router.calls, 2)


class TestScreenHelpers(unittest.TestCase):
    def test_encode_frame_round_trip(self) -> None:
        from lecturelens.agents.screen_agent import ScreenFrame, encode_frame_as_png
        raw = bytes([10, 20, 30, 255] * (4 * 3))
        frame = ScreenFrame(raw_bgra=raw, width=4, height=3, thumbnail=np.zeros((3, 4), dtype=np.uint8))
        decoded = Image.open(io.BytesIO(encode_frame_as_png(frame)))
        self.assertEqual(decoded.size, (4, 3))
        self.assertEqual(decoded.getpixel((0, 0)), (30, 20, 10))


class TestVisionEmbedderAndGate(unittest.TestCase):
    def test_vision_embedder_preprocessing_and_similarity(self) -> None:
        from lecturelens.core.vision_embedder import VisionEmbedder
        embedder = VisionEmbedder(auto_download=False)
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        tensor = embedder.preprocess(dummy_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(embedder.similarity(v1, v2), 1.0)
        self.assertAlmostEqual(embedder.similarity(v1, v3), 0.0)

    def test_spatial_block_similarity(self) -> None:
        from lecturelens.core.image_utils import compute_spatial_block_similarity
        arr1 = np.ones((60, 80), dtype=np.uint8) * 100
        arr2 = np.ones((60, 80), dtype=np.uint8) * 100
        sim = compute_spatial_block_similarity(arr1, arr2)
        self.assertAlmostEqual(sim, 1.0)

        small1 = np.zeros((4, 4), dtype=np.uint8)
        small2 = np.ones((4, 4), dtype=np.uint8) * 200
        sim_small = compute_spatial_block_similarity(small1, small2)
        self.assertAlmostEqual(sim_small, 0.0)

    def test_vision_model_download_is_checksum_verified(self) -> None:
        """The ONNX model is executed in-process, so an unverified download
        was a supply-chain hole."""
        import hashlib
        import tempfile

        from lecturelens.core import vision_embedder

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "squeezenet.onnx"
            embedder = vision_embedder.VisionEmbedder(model_path=str(model_path), auto_download=False)

            payload = b"not-the-real-model" * 1000
            model_path.write_bytes(payload)
            self.assertFalse(embedder._verify_checksum(model_path))

            real_digest = hashlib.sha256(payload).hexdigest()
            with mock.patch.dict(os.environ, {"LECTURELENS_VISION_MODEL_SHA256": real_digest}):
                self.assertTrue(embedder._verify_checksum(model_path))

            pinned = vision_embedder.EXPECTED_MODEL_SHA256
            self.assertEqual(len(pinned), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in pinned))

    def test_slide_change_gate_with_mock_embedder_filters_motion(self) -> None:
        from lecturelens.core.image_utils import SlideChangeGate
        class MockEmbedder:
            def __init__(self) -> None:
                self.sim_return = 0.98

            def embed(self, arr: np.ndarray) -> np.ndarray:
                return np.ones(10, dtype=np.float32) / np.sqrt(10.0)

            def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
                return self.sim_return

        mock_emb = MockEmbedder()
        gate = SlideChangeGate(
            change_threshold=0.004,
            pixel_delta=25,
            min_interval_seconds=1.0,
            embedder=mock_emb,
            similarity_threshold=0.95
        )
        f1 = np.zeros((100, 100), dtype=np.uint8)
        self.assertTrue(gate.evaluate(f1, 0.0))

        f2 = np.zeros((100, 100), dtype=np.uint8)
        f2[:30, :30] = 200
        mock_emb.sim_return = 0.98
        self.assertFalse(gate.evaluate(f2, 2.0))

        mock_emb.sim_return = 0.80
        self.assertTrue(gate.evaluate(f2, 3.0))


if __name__ == "__main__":
    unittest.main()
