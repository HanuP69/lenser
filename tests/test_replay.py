import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.core.replay import ReplayEngine
from lecturelens.schemas.events import TimelineEntry


class TestReplayEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.timeline_path = Path(self.temp_dir.name) / "timeline.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_sample_timeline(self, count: int = 5) -> None:
        lines: list[str] = []
        for i in range(count):
            entry = TimelineEntry(
                session_id="source_sess",
                event_id=f"evt_{i}",
                timestamp_epoch=1700000000.0 + i * 2.0,
                elapsed_seconds=float(i * 2),
                source_type="transcript" if i % 2 == 0 else "slide",
                content=f"Event content {i}"
            )
            lines.append(json.dumps(entry.model_dump()))
        self.timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def test_replay_loads_and_emits_to_callback_at_speed_zero(self) -> None:
        self.write_sample_timeline(4)
        engine = ReplayEngine(self.timeline_path, speed=0.0)
        received: list[TimelineEntry] = []

        def callback(entry: TimelineEntry) -> None:
            received.append(entry)

        count = await engine.replay(target_session_id="new_target", on_entry=callback, publish_to_timeline_stream=False)
        self.assertEqual(count, 4)
        self.assertEqual(len(received), 4)
        self.assertEqual(received[0].session_id, "new_target")
        self.assertEqual(received[0].event_id, "evt_0")
        self.assertEqual(received[3].event_id, "evt_3")

    async def test_replay_publishes_to_redis_stream(self) -> None:
        self.write_sample_timeline(2)
        mock_bus = mock.AsyncMock()
        engine = ReplayEngine(self.timeline_path, redis_bus=mock_bus, speed=0.0)
        count = await engine.replay(target_session_id="dest_sess", publish_to_timeline_stream=True)
        self.assertEqual(count, 2)
        self.assertEqual(mock_bus.publish_resilient.await_count, 2)
        call_args = mock_bus.publish_resilient.call_args_list[0]
        stream_name, entry = call_args[0]
        self.assertEqual(stream_name, "lecturelens:dest_sess:stream:timeline")
        self.assertEqual(entry.session_id, "dest_sess")

    async def test_empty_or_missing_file_returns_zero(self) -> None:
        missing_path = Path(self.temp_dir.name) / "nonexistent.jsonl"
        engine = ReplayEngine(missing_path, speed=0.0)
        count = await engine.replay()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
