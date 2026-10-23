"""Tests for board and handwriting detection and pipeline."""

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

from lecturelens.agents.screen_agent import ScreenAgent, build_thumbnail_from_png
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.handwriting import (
    BOARD_VISION_PROMPT,
    detect_board_surface,
    filter_chrome_lines,
    is_board_wipe
)
from lecturelens.core.image_utils import SlideChangeGate


class TestBoardDetection(unittest.TestCase):
    def test_green_chalkboard_detected(self) -> None:
        board = np.zeros((200, 300, 3), dtype=np.uint8)
        board[:, :, 0] = 30
        board[:, :, 1] = 80
        board[:, :, 2] = 40
        is_board, kind = detect_board_surface(board)
        self.assertTrue(is_board)
        self.assertEqual(kind, "chalkboard_green")

    def test_slate_chalkboard_detected(self) -> None:
        slate = np.zeros((200, 300, 3), dtype=np.uint8)
        slate.fill(45)
        is_board, kind = detect_board_surface(slate)
        self.assertTrue(is_board)
        self.assertEqual(kind, "chalkboard_slate")

    def test_plain_digital_slide_not_detected_as_chalkboard(self) -> None:
        slide = np.zeros((200, 300, 3), dtype=np.uint8)
        slide[:, :, 0] = 200
        slide[:, :, 1] = 50
        slide[:, :, 2] = 50
        is_board, kind = detect_board_surface(slide)
        self.assertFalse(is_board)
        self.assertEqual(kind, "none")


class TestChromeFiltering(unittest.TestCase):
    def test_filter_chrome_lines_drops_bottom_player_controls(self) -> None:
        lines = [
            {"text": "Substitution vs Elimination", "box": [50.0, 100.0, 300.0, 140.0]},
            {"text": "Pause", "box": [10.0, 950.0, 60.0, 980.0]},
            {"text": "01:25 / 14:00", "box": [80.0, 950.0, 180.0, 980.0]},
            {"text": "YouTube", "box": [900.0, 950.0, 980.0, 980.0]}
        ]
        filtered = filter_chrome_lines(lines, image_height=1000, image_width=1000)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["text"], "Substitution vs Elimination")


class TestBoardWipeDetection(unittest.TestCase):
    def test_board_wipe_detected_on_large_diff(self) -> None:
        first = np.full((100, 100), 50, dtype=np.uint8)
        first[20:80, 20:80] = 200
        second = np.full((100, 100), 50, dtype=np.uint8)
        self.assertTrue(is_board_wipe(first, second, wipe_threshold=0.20))

    def test_small_change_is_not_board_wipe(self) -> None:
        first = np.full((100, 100), 50, dtype=np.uint8)
        second = np.full((100, 100), 50, dtype=np.uint8)
        second[10:15, 10:15] = 200
        self.assertFalse(is_board_wipe(first, second, wipe_threshold=0.20))


class DummyEmbedder:
    def is_available(self) -> bool:
        return True

    def embed(self, thumbnail: np.ndarray) -> np.ndarray:
        return np.ones(128, dtype=np.float32)

    def similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        return 0.99


class TestBoardChangeGate(unittest.TestCase):
    def test_board_change_gate_not_blocked_by_room_embedding(self) -> None:
        gate = SlideChangeGate(
            change_threshold=0.04,
            pixel_delta=25,
            min_interval_seconds=1.0,
            embedder=DummyEmbedder()
        )
        first_thumb = np.zeros((180, 320, 3), dtype=np.uint8)
        first_thumb[:, :, 0] = 30
        first_thumb[:, :, 1] = 80
        first_thumb[:, :, 2] = 40
        self.assertTrue(gate.evaluate(first_thumb, 0.0))

        second_thumb = first_thumb.copy()
        second_thumb[30:100, 50:200] = 180
        self.assertFalse(gate.evaluate(second_thumb, 2.0))
        self.assertTrue(gate.evaluate(second_thumb, 3.0))


class MockEventBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append(event)
        return "msg-1"


class CapturePromptRouter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages: Any) -> Any:
        class FakeResponse:
            def __init__(self, content: str) -> None:
                self.content = content
        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        self.prompts.append(str(item.get("text", "")))
            elif isinstance(content, str):
                self.prompts.append(content)
        return FakeResponse("Transcribed: H2O + EtOH")


def make_chalkboard_png(size: tuple[int, int] = (640, 480)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 0] = 30
    arr[:, :, 1] = 80
    arr[:, :, 2] = 40
    arr[50:80, 50:200] = 220
    im = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class TestScreenAgentBoardPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_screen_agent_uses_board_vision_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.vision_retry_delays_seconds = [0.0]
            settings.screen_local_ocr = False
            clock = SessionClock()
            bus = MockEventBus()
            router = CapturePromptRouter()
            agent = ScreenAgent(
                settings=settings,
                redis_bus=bus,
                session_clock=clock,
                session_id="board_test",
                llm_router=router
            )
            png = make_chalkboard_png()
            event = await agent.ingest_png_bytes(png, capture_epoch=1.0, enforce_gate=False)
            self.assertIsNotNone(event)
            if agent.retry_tasks:
                await asyncio.gather(*list(agent.retry_tasks))
            self.assertGreaterEqual(len(router.prompts), 1)
            self.assertIn("chalkboard and whiteboard lectures", router.prompts[0])


if __name__ == "__main__":
    unittest.main()
