import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import cv2
import numpy as np

from lecturelens.agents import camera_agent
from lecturelens.agents.camera_agent import (
    CameraAgent,
    build_board_thumbnail,
    compute_blur_score,
    compute_motion_score,
    detect_board_contour_and_warp,
    find_board_quadrilateral,
    order_quadrilateral_points
)
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock


def encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def board_scene(strokes: int = 6, corners: Optional[list[list[int]]] = None) -> np.ndarray:
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    if corners is None:
        corners = [[110, 70], [540, 90], [520, 410], [90, 380]]
    cv2.fillConvexPoly(image, np.array(corners, dtype=np.int32), (235, 235, 235))
    for index in range(strokes):
        y = 120 + index * 40
        cv2.line(image, (160, y), (460, y + 5), (20, 20, 160), 3)
    return image


class TestWarp(unittest.TestCase):
    def test_board_quadrilateral_is_found_and_warped(self) -> None:
        scene = board_scene()
        quad = find_board_quadrilateral(scene, 0.2)
        self.assertIsNotNone(quad)
        warped_bytes = detect_board_contour_and_warp(encode(scene), 0.2)
        warped = cv2.imdecode(np.frombuffer(warped_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(warped.shape[1], 380)
        self.assertGreater(warped.shape[0], 280)
        border_mean = float(np.mean(warped[5:15, 5:-5]))
        self.assertGreater(border_mean, 150.0)

    def test_noise_without_board_returns_original_bytes(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        original = encode(noise)
        self.assertEqual(detect_board_contour_and_warp(original, 0.2), original)

    def test_small_object_is_rejected(self) -> None:
        image = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.fillConvexPoly(image, np.array([[300, 200], [340, 200], [340, 240], [300, 240]], dtype=np.int32), (240, 240, 240))
        original = encode(image)
        self.assertEqual(detect_board_contour_and_warp(original, 0.2), original)

    def test_full_frame_border_is_not_treated_as_board(self) -> None:
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (1, 1), (638, 478), (0, 0, 0), 2)
        original = encode(image)
        self.assertEqual(detect_board_contour_and_warp(original, 0.2), original)

    def test_undecodable_bytes_are_returned_unchanged(self) -> None:
        self.assertEqual(detect_board_contour_and_warp(b"not an image", 0.2), b"not an image")

    def test_point_ordering(self) -> None:
        points = np.array([[500, 400], [10, 20], [490, 30], [20, 410]], dtype=np.float32)
        ordered = order_quadrilateral_points(points)
        self.assertEqual(ordered[0].tolist(), [10.0, 20.0])
        self.assertEqual(ordered[1].tolist(), [490.0, 30.0])
        self.assertEqual(ordered[2].tolist(), [500.0, 400.0])
        self.assertEqual(ordered[3].tolist(), [20.0, 410.0])


class TestScores(unittest.TestCase):
    def test_motion_and_blur(self) -> None:
        sharp = board_scene()
        blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
        self.assertEqual(compute_motion_score(encode(sharp), encode(sharp)), 0.0)
        self.assertGreater(compute_motion_score(encode(sharp), encode(board_scene(0))), 0.0)
        self.assertGreater(compute_blur_score(encode(sharp)), compute_blur_score(encode(blurred)))


class FakeBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append(event)
        return "1-1"


class FakeSource:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.index = 0
        self.closed = False

    def read_jpeg(self) -> bytes:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeRouter:
    def __init__(self, text: str = "x = 2\nCONFIDENCE: 0.9", fail: bool = False) -> None:
        self.calls = 0
        self.text = text
        self.fail = fail

    async def ainvoke(self, messages: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("all providers down")
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.text, additional_kwargs={"lecturelens_provider": "gemma"})


async def inline_process_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
    return function(*args)


def make_agent(temp_dir: str, router: Any, source: Any = None) -> tuple[CameraAgent, FakeBus]:
    settings = Settings()
    settings.output_dir = Path(temp_dir)
    settings.board_min_interval_seconds = 0.0
    settings.blur_variance_threshold = 5.0
    settings.motion_stillness_required_frames = 2
    settings.camera_poll_interval_seconds = 0.01
    bus = FakeBus()
    return CameraAgent(settings, bus, SessionClock(), "camera_test", router, source or FakeSource([encode(board_scene())])), bus


class TestCameraAgent(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_board_is_published_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            router = FakeRouter()
            agent, bus = make_agent(temp_dir, router)
            frame = encode(board_scene())
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                for _ in range(12):
                    await agent.evaluate_frame(frame)
            self.assertEqual(len(bus.published), 1)
            self.assertEqual(router.calls, 1)

    async def test_changed_board_is_published_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            router = FakeRouter()
            agent, bus = make_agent(temp_dir, router)
            first = encode(board_scene(2))
            second = encode(board_scene(7))
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                for _ in range(5):
                    await agent.evaluate_frame(first)
                for _ in range(5):
                    await agent.evaluate_frame(second)
            self.assertEqual(len(bus.published), 2)

    async def test_event_has_real_provider_and_parsed_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir, FakeRouter())
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                event = await agent.process_still_frame(encode(board_scene()))
            self.assertEqual(event.provider_used, "gemma")
            self.assertAlmostEqual(event.confidence, 0.9)
            self.assertEqual(event.raw_ocr_text, "x = 2")
            self.assertTrue(Path(event.image_file_path).exists())
            raw_files = list(agent.board_dir.glob("board_raw_*.jpg"))
            self.assertEqual(len(raw_files), 1)

    async def test_vision_failure_still_saves_frames_and_publishes_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir, FakeRouter(fail=True))
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                event = await agent.process_still_frame(encode(board_scene()))
            self.assertEqual(event.raw_ocr_text, "")
            self.assertEqual(event.provider_used, "none")
            self.assertEqual(len(list(agent.board_dir.glob("board_*.jpg"))), 2)

    async def test_warp_failure_falls_back_to_raw_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir, FakeRouter())

            async def broken_pool(function: Any, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("pool broke")

            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=broken_pool):
                event = await agent.process_still_frame(encode(board_scene()))
            self.assertIsNotNone(event)

    async def test_moving_hand_blocks_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir, FakeRouter())
            agent.settings.motion_threshold = 0.001
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                for index in range(10):
                    await agent.evaluate_frame(encode(board_scene(index % 7)))
            self.assertEqual(len(bus.published), 0)

    async def test_run_loop_reads_source_and_closes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = FakeSource([encode(board_scene())])
            agent, bus = make_agent(temp_dir, FakeRouter(), source)

            async def stopper() -> None:
                await asyncio.sleep(0.6)
                agent.stop()

            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=20)
            self.assertTrue(source.closed)
            self.assertEqual(len(bus.published), 1)

    async def test_capture_errors_do_not_crash_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class FailingSource:
                closed = False

                def read_jpeg(self) -> bytes:
                    raise RuntimeError("camera unplugged")

                def close(self) -> None:
                    self.closed = True

            agent, bus = make_agent(temp_dir, FakeRouter(), FailingSource())

            async def stopper() -> None:
                await asyncio.sleep(0.4)
                agent.stop()

            await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=20)
            self.assertGreaterEqual(agent.consecutive_capture_errors, 1)

    async def test_raw_numbering_continues_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, _ = make_agent(temp_dir, None)
            (agent.board_dir / "board_raw_0004.jpg").write_bytes(b"x")
            restarted, _ = make_agent(temp_dir, None)
            self.assertEqual(restarted.capture_counter, 4)

    async def test_missing_router_publishes_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent, bus = make_agent(temp_dir, None)
            with mock.patch.object(camera_agent, "run_in_process_pool", side_effect=inline_process_pool):
                event = await agent.process_still_frame(encode(board_scene()))
            self.assertEqual(event.provider_used, "none")


if __name__ == "__main__":
    unittest.main()
