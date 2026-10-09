import unittest
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TranscriptChunk


class TestRedisBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = RedisBus("redis://localhost:6379")
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


if __name__ == "__main__":
    unittest.main()
