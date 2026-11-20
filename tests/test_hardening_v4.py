"""Regression tests for the third review round (see FIXES.md, v0.4.0)."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_process_pool, shutdown_pools
from lecturelens.core.env_file import read_env_value, set_env_value

# The marks of a thread fallback. The spawn of a worker process does not
# share globals with the parent, so the list only grows when the function
# was executed in the parent process by the thread pool.
fallback_marks: list[int] = []


def failing_work() -> int:
    raise ValueError("the work itself is wrong")


class FileioSmoke(unittest.TestCase):
    def test_a_value_with_a_hash_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            self.assertTrue(set_env_value("LECTURELENS_WINDOW_TITLE", "Slides #1 - 2026", env_path=env_path))
            self.assertEqual(read_env_value("LECTURELENS_WINDOW_TITLE", env_path=env_path), "Slides #1 - 2026")


class TestConcurrencyReraises(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        shutdown_pools()

    async def test_the_error_of_the_work_is_not_eaten_and_not_run_twice(self) -> None:
        """A raising task must fail once, not run a second time in a thread.

        The worker exception used to land in the same catch as a broken
        pool, so the pool was reset under other coroutines and the call
        was then executed again in the thread pool. For anything with
        side effects the work happened twice and the caller saw a
        RuntimeError instead of the error.
        """
        fallback_marks.clear()
        with self.assertRaises(ValueError):
            await run_in_process_pool(failing_work, max_attempts=1)
        # The work belongs to the process pool. A thread fallback would
        # have run the function again here in the parent.
        self.assertEqual(len(fallback_marks), 0)


class TestConcurrencyReraises(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        shutdown_pools()


class TestSessionClockSteps(unittest.TestCase):
    def test_a_wall_clock_step_reanchors_and_stays_letterless(self) -> None:
        """After a wall clock step the timestamps do not flap anymore.

        The step used to leave the monotonic estimate offset for the rest
        of the session, so every timestamp came from whichever side of
        the tolerance the slew happened to be in and the timeline
        jumped back and forth.
        """
        clock = SessionClock()
        clock.reference_wall_time = time.time() - 600.0
        clock.reference_monotonic_time = time.monotonic() - 1.0
        first = clock.get_timestamp()
        self.assertAlmostEqual(first, time.time(), delta=1.0)
        # The estimate follows the new anchor, so the next timestamps do
        # not jump back to the stale pair.
        time.sleep(0.02)
        second = clock.get_timestamp()
        self.assertGreater(second, first)
        self.assertAlmostEqual(second - first, 0.06, delta=1.0)

    def test_a_wall_clock_step_does_not_reset_the_elapsed_time(self) -> None:
        clock = SessionClock(elapsed_offset_seconds=0.0)
        clock.reference_wall_time = time.time() - 90.0
        clock.reference_monotonic_time = time.monotonic() - 1.0
        elapsed = clock.get_run_elapsed_seconds()
        self.assertAlmostEqual(elapsed, 90.0, delta=2.0)
        clock.get_timestamp()
        self.assertAlmostEqual(clock.get_run_elapsed_seconds(), 90.0, delta=2.0)

    def test_a_scalar_clock_file_is_survivable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clock.json"
            path.write_text("true", encoding="utf-8")
            clock = SessionClock.load_or_create(path)
            self.assertIsNotNone(clock)


class TestKeyRotation(unittest.TestCase):
    def test_the_errors_that_rotate_keys(self) -> None:
        from lecturelens.router.llm_router import is_key_rotation_error

        class WithStatus(Exception):
            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        self.assertTrue(is_key_rotation_error(WithStatus("too many requests", 429)))
        self.assertTrue(is_key_rotation_error(WithStatus("bad gateway", 502)))
        self.assertTrue(is_key_rotation_error(WithStatus("unauthorized", 401)))
        self.assertTrue(is_key_rotation_error(WithStatus("forbidden", 403)))
        self.assertTrue(is_key_rotation_error(TimeoutError("the request timed out")))
        self.assertTrue(is_key_rotation_error(ConnectionError("connection refused")))
        self.assertTrue(is_key_rotation_error(Exception("429 quota exceeded")))
        # An error of the request itself would fail on every key the
        # same way, so it does not rotate.
        self.assertFalse(is_key_rotation_error(WithStatus("model 'x' not found", 404)))
        self.assertFalse(is_key_rotation_error(WithStatus("payload too large", 413)))
        self.assertFalse(is_key_rotation_error(ValueError("bad message shape")))


class TestProviderModeFiltering(unittest.TestCase):
    def test_hybrid_allows_every_provider_and_cloud_leaves_the_local_out(self) -> None:
        from lecturelens.router.llm_router import is_cloud_provider, provider_allowed_for_mode

        local_config = {"type": "ollama"}
        cloud_config = {"type": "openai_compatible"}
        self.assertFalse(is_cloud_provider(local_config))
        self.assertTrue(is_cloud_provider(cloud_config))
        self.assertTrue(provider_allowed_for_mode(local_config, "hybrid"))
        self.assertTrue(provider_allowed_for_mode(cloud_config, "hybrid"))
        self.assertFalse(provider_allowed_for_mode(local_config, "cloud"))
        self.assertFalse(provider_allowed_for_mode(cloud_config, "local"))


class TestCitationSourceNames(unittest.TestCase):
    def test_a_timeline_source_is_named_like_its_file(self) -> None:
        from lecturelens.knowledge.qa_engine import QAEngine

        self.assertEqual(
            QAEngine._humanize_source_id({"source_id": "timeline_0012", "source_type": "transcript"}),
            "timeline.jsonl"
        )
        self.assertEqual(QAEngine._humanize_source_id({"source_id": "notes_0abc"}), "notes.md")


class TestGatewayAssetAuth(unittest.TestCase):
    def test_an_absent_token_is_not_a_correct_token(self) -> None:
        """An asset request without a token is unauthorized, not served."""
        with tempfile.TemporaryDirectory() as folder:
            from lecturelens.config.settings import Settings
            from lecturelens.gateway.server import GatewayServer

            settings = Settings()
            settings.output_dir = Path(folder)
            settings.gateway_api_key = "secret_key_123"

            class DummyBus:
                pass

            gateway = GatewayServer(settings, DummyBus())
            request = type("Req", (), {"path": "/api/assets/some_session/diagram.png", "headers": {}})()
            response = gateway.handle_http_request(None, request)
            self.assertIsNotNone(response)
            self.assertEqual(response.status_code, 401)


class TestPublishResilientDedup(unittest.IsolatedAsyncioTestCase):
    async def test_a_published_event_is_not_delivered_again(self) -> None:
        """A second publish of the same event id answers, it does not repeat."""
        from lecturelens.config.settings import Settings
        from lecturelens.core.redis_bus import RedisBus
        from lecturelens.schemas.events import TranscriptChunk

        bus = RedisBus(Settings().redis_url)
        try:
            await bus.client.ping()
        except Exception:
            await bus.close()
            self.skipTest("Redis not available on localhost:6379")

        stream_name = "lecturelens:test_hardening_stream"
        await bus.client.delete(stream_name)
        chunk = TranscriptChunk(
            session_id="test_hardening",
            timestamp_epoch=1000.0,
            start_seconds=0.0,
            end_seconds=2.0,
            text="dedupe me",
            confidence=0.9
        )
        try:
            first = await bus.publish_resilient(stream_name, chunk)
            second = await bus.publish_resilient(stream_name, chunk)
            self.assertEqual(second, "published_earlier")
            length = await bus.client.xlen(stream_name)
            self.assertEqual(length, 1)
        finally:
            await bus.client.delete(stream_name)
            await bus.close()
        _ = first


if __name__ == "__main__":
    unittest.main()
