import tempfile
import unittest
from pathlib import Path
from lecturelens.agents.sync_agent import SyncAgent
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TimelineEntry


class TestSyncAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.settings.sync_merge_window_seconds = 0.5
        self.settings.correlation_window_seconds = 2.0

        self.clock = SessionClock()
        self.bus = RedisBus("redis://localhost:6379")
        self.agent = SyncAgent(
            settings=self.settings,
            redis_bus=self.bus,
            session_clock=self.clock,
            session_id="test_sync_session"
        )

    async def asyncTearDown(self) -> None:
        await self.bus.close()
        self.temp_dir.cleanup()

    def test_correlation_assignment(self) -> None:
        entry_audio = TimelineEntry(
            session_id="test_sync_session",
            timestamp_epoch=100.0,
            source_type="transcript",
            content="Look at this algorithm on the board",
            correlation_id=None,
            importance_flags=[],
            metadata={}
        )
        entry_slide = TimelineEntry(
            session_id="test_sync_session",
            timestamp_epoch=101.0,
            source_type="slide",
            content="Binary Search Tree slide",
            correlation_id=None,
            importance_flags=[],
            metadata={}
        )

        self.agent.pending_buffer = [entry_audio, entry_slide]
        ready = self.agent.correlate_and_extract_ready_entries(current_time=105.0)

        self.assertEqual(len(ready), 2)
        self.assertIsNotNone(ready[0].correlation_id)
        self.assertEqual(ready[0].correlation_id, ready[1].correlation_id)


if __name__ == "__main__":
    unittest.main()
