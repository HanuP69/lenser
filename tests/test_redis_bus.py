import asyncio
import time
import unittest

import redis.asyncio as redis_module

from lecturelens.config.settings import Settings
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TranscriptChunk


class TestRedisBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = RedisBus(Settings().redis_url)
        try:
            await self.bus.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            await self.bus.client.delete("lecturelens:test_stream:transcript")
            await self.bus.close()

    async def test_publish_and_stream_info(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        stream_name = "lecturelens:test_stream:transcript"
        chunk = TranscriptChunk(
            session_id="test_session",
            timestamp_epoch=1000.0,
            start_seconds=0.0,
            end_seconds=2.0,
            text="Hello world test",
            confidence=0.9,
            language="en",
            audio_source="mic",
            speaker_id=None,
            importance_flags=[],
            raw_audio_path=None
        )

        entry_id = await self.bus.publish(stream_name, chunk)
        self.assertIsNotNone(entry_id)
        self.assertIn("-", entry_id)

        info = await self.bus.get_stream_info(stream_name)
        self.assertGreaterEqual(info.get("length", 0), 1)

    async def test_consumer_group_creation(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        stream_name = "lecturelens:test_stream:transcript"
        group_name = "test_group"

        await self.bus.ensure_consumer_group(stream_name, group_name)
        await self.bus.ensure_consumer_group(stream_name, group_name)

        pending_count = await self.bus.get_pending_count(stream_name, group_name)
        self.assertEqual(pending_count, 0)


def make_chunk(text: str) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="test_session",
        timestamp_epoch=1000.0,
        start_seconds=0.0,
        end_seconds=2.0,
        text=text,
        confidence=0.9
    )


class TestRedisBusReliability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = RedisBus(Settings().redis_url, reclaim_interval_seconds=0.0, reclaim_idle_ms=0)
        try:
            await self.bus.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
        self.stream = "lecturelens:test_reliability:transcript"
        self.group = "test_group_reliability"
        if self.redis_available:
            await self.bus.client.delete(self.stream)

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            await self.bus.client.delete(self.stream)
            await self.bus.close()

    async def run_consumer(self, handler, stop_flag: dict, consumer_name: str = "consumer_a") -> None:
        await self.bus.consume(
            self.stream, self.group, consumer_name, handler,
            batch_size=5, block_ms=50, should_stop=lambda: stop_flag["stop"]
        )

    async def test_all_messages_arrive_and_are_acknowledged(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        for index in range(10):
            await self.bus.publish(self.stream, make_chunk(f"line {index}"))
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": True}
        await asyncio.wait_for(self.run_consumer(handler, stop_flag), timeout=10)
        self.assertEqual(len(received), 10)
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_failing_message_discarded_after_max_attempts(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        self.bus.max_delivery_attempts = 3
        await self.bus.publish(self.stream, make_chunk("poison"))
        await self.bus.publish(self.stream, make_chunk("fine"))
        received: list[str] = []

        async def handler(data: dict) -> None:
            if data["text"] == "poison":
                raise ValueError("bad message")
            received.append(data["text"])

        stop_flag = {"stop": False}
        consumer_task = asyncio.create_task(self.run_consumer(handler, stop_flag))
        for _ in range(50):
            await asyncio.sleep(0.05)
            if "fine" in received and (await self.bus.get_pending_count(self.stream, self.group)) == 0:
                break
        stop_flag["stop"] = True
        await asyncio.wait_for(consumer_task, timeout=10)
        self.assertEqual(received, ["fine"])
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_restarted_consumer_reprocesses_its_unacknowledged_messages(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        await self.bus.publish(self.stream, make_chunk("one"))
        await self.bus.ensure_consumer_group(self.stream, self.group)
        await self.bus.read_group_messages(self.stream, self.group, "consumer_a", ">", 5, None)
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 1)
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": True}
        await asyncio.wait_for(self.run_consumer(handler, stop_flag, "consumer_a"), timeout=10)
        self.assertEqual(received, ["one"])
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_other_consumers_stuck_messages_are_reclaimed(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        await self.bus.publish(self.stream, make_chunk("orphan"))
        await self.bus.ensure_consumer_group(self.stream, self.group)
        await self.bus.read_group_messages(self.stream, self.group, "dead_consumer", ">", 5, None)
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": False}
        consumer_task = asyncio.create_task(self.run_consumer(handler, stop_flag, "live_consumer"))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if received:
                break
        stop_flag["stop"] = True
        await asyncio.wait_for(consumer_task, timeout=10)
        self.assertEqual(received, ["orphan"])

    async def test_publish_resilient_returns_none_on_connection_error(self) -> None:
        bus = RedisBus(Settings().redis_url)

        async def flaky_publish(stream_name, event):
            raise redis_module.ConnectionError("down")

        bus.publish = flaky_publish
        result = await bus.publish_resilient("lecturelens:test_buffer:stream", make_chunk("a"))
        self.assertIsNone(result)
        await bus.close()

    async def test_consume_survives_read_errors_with_backoff(self) -> None:
        original_read = self.bus.read_group_messages
        calls = {"count": 0}

        async def flaky_read(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] <= 2:
                raise redis_module.ConnectionError("down")
            return []

        self.bus.read_group_messages = flaky_read

        async def noop(data: dict) -> None:
            return None

        original_ensure = self.bus.ensure_consumer_group

        async def fake_ensure(stream_name, group_name):
            return None

        self.bus.ensure_consumer_group = fake_ensure
        stop_flag = {"stop": True}
        started = time.monotonic()
        await asyncio.wait_for(self.run_consumer(noop, stop_flag), timeout=10)
        self.assertTrue(calls["count"] >= 1)
        self.assertLess(time.monotonic() - started, 10)


if __name__ == "__main__":
    unittest.main()
