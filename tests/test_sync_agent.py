import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from lecturelens.agents import sync_agent
from lecturelens.agents.sync_agent import BufferedEntry, SyncAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import SYNC_STOP_FLAG_NAME, set_flag
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import (
    SlideCaptured,
    TimelineEntry,
    TranscriptChunk
)
from tests.support import pluck

SESSION = "test_sync_session"


def entry(timestamp: float, source_type: str, content: str = "text") -> TimelineEntry:
    return TimelineEntry(
        session_id=SESSION,
        timestamp_epoch=timestamp,
        source_type=source_type,
        content=content,
        correlation_id=None,
        importance_flags=[],
        metadata={}
    )


class SyncTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.settings.sync_merge_window_seconds = 0.5
        self.settings.correlation_window_seconds = 5.0
        self.bus = RedisBus(self.settings.redis_url)
        self.agent = SyncAgent(self.settings, self.bus, SessionClock(), SESSION)

    async def asyncTearDown(self) -> None:
        await self.bus.close()
        self.temp_dir.cleanup()

    def read_timeline(self) -> list[dict]:
        path = self.agent.timeline_path
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        parsed: list[dict] = []
        for line in lines:
            if line.strip():
                parsed.append(json.loads(line))
        return parsed


class TestCorrelation(SyncTestBase):
    def test_simple_transcript_slide_pair(self) -> None:
        transcript = entry(100.0, "transcript")
        slide = entry(101.0, "slide")
        self.agent.assign_correlation_ids([transcript, slide])
        self.assertIsNotNone(slide.correlation_id)
        self.assertEqual(transcript.correlation_id, slide.correlation_id)

    def test_earlier_transcript_keeps_its_link_when_a_later_one_appears(self) -> None:
        first = entry(0.0, "transcript")
        second = entry(4.0, "transcript")
        slide = entry(5.0, "slide")
        self.agent.assign_correlation_ids([first, second, slide])
        self.assertEqual(first.correlation_id, slide.correlation_id)
        self.assertEqual(second.correlation_id, slide.correlation_id)

    def test_transcript_after_slide_still_correlates(self) -> None:
        first = entry(0.0, "transcript")
        slide = entry(3.0, "slide")
        later = entry(6.0, "transcript")
        self.agent.assign_correlation_ids([first, slide, later])
        self.assertEqual(later.correlation_id, slide.correlation_id)

    def test_transcript_picks_nearest_of_two_slides(self) -> None:
        slide_one = entry(0.0, "slide")
        slide_two = entry(8.0, "slide")
        transcript = entry(6.0, "transcript")
        self.agent.assign_correlation_ids([slide_one, transcript, slide_two])
        self.assertNotEqual(slide_one.correlation_id, slide_two.correlation_id)
        self.assertEqual(transcript.correlation_id, slide_two.correlation_id)

    def test_distant_entries_are_not_correlated(self) -> None:
        transcript = entry(0.0, "transcript")
        slide = entry(60.0, "slide")
        self.agent.assign_correlation_ids([transcript, slide])
        self.assertIsNone(transcript.correlation_id)
        self.assertIsNotNone(slide.correlation_id)

    def test_consecutive_slides_get_different_ids(self) -> None:
        first = entry(10.0, "slide")
        second = entry(12.0, "slide")
        self.agent.assign_correlation_ids([first, second])
        self.assertNotEqual(first.correlation_id, second.correlation_id)

    def test_correlation_works_across_flush_batches(self) -> None:
        slide = entry(100.0, "slide")
        self.agent.assign_correlation_ids([slide])
        late_transcript = entry(102.0, "transcript")
        self.agent.assign_correlation_ids([late_transcript])
        self.assertEqual(late_transcript.correlation_id, slide.correlation_id)

    def test_assignment_is_idempotent_for_retries(self) -> None:
        slide = entry(100.0, "slide")
        transcript = entry(101.0, "transcript")
        self.agent.assign_correlation_ids([slide, transcript])
        first_id = slide.correlation_id
        self.agent.assign_correlation_ids([slide, transcript])
        self.assertEqual(slide.correlation_id, first_id)
        self.assertEqual(len(self.agent.recent_anchors), 1)


class TestFlushing(SyncTestBase):
    async def test_entries_wait_for_arrival_based_merge_window(self) -> None:
        self.settings.sync_merge_window_seconds = 30.0
        old_event_time = entry(1.0, "transcript")
        await self.agent.add_entry(old_event_time)
        written = await self.agent.flush_ready_entries(force=False)
        self.assertEqual(written, 0)
        self.assertEqual(self.read_timeline(), [])

    async def test_late_arriving_older_event_is_ordered_with_its_batch(self) -> None:
        await self.agent.add_entry(entry(10.0, "transcript"))
        await self.agent.add_entry(entry(5.0, "transcript"))
        await self.agent.add_entry(entry(7.0, "slide"))
        written = await self.agent.flush_ready_entries(force=True)
        self.assertEqual(written, 3)
        stamps = pluck(self.read_timeline(), "timestamp_epoch")
        self.assertEqual(stamps, sorted(stamps))

    async def test_ready_entries_flush_after_window(self) -> None:
        self.settings.sync_merge_window_seconds = 0.05
        await self.agent.add_entry(entry(1.0, "transcript"))
        await asyncio.sleep(0.1)
        self.assertEqual(await self.agent.flush_ready_entries(force=False), 1)
        self.assertEqual(len(self.agent.pending_buffer), 0)

    async def test_write_failure_keeps_entries_buffered(self) -> None:
        await self.agent.add_entry(entry(1.0, "transcript"))
        with mock.patch.object(sync_agent, "append_text_durably", side_effect=OSError("disk full")):
            written = await self.agent.flush_ready_entries(force=True)
        self.assertEqual(written, 0)
        self.assertEqual(len(self.agent.pending_buffer), 1)
        self.assertEqual(await self.agent.flush_ready_entries(force=True), 1)
        self.assertEqual(len(self.read_timeline()), 1)

    async def test_timeline_lines_are_complete_json(self) -> None:
        for index in range(20):
            await self.agent.add_entry(entry(float(index), "transcript", f"line {index}"))
        await self.agent.flush_ready_entries(force=True)
        raw = self.agent.timeline_path.read_text(encoding="utf-8")
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(len(self.read_timeline()), 20)


class TestSyncEndToEnd(SyncTestBase):
    async def test_streams_are_consumed_correlated_and_drained_on_stop(self) -> None:
        try:
            await self.bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")
        session = "sync_e2e_session"
        agent = SyncAgent(self.settings, self.bus, SessionClock(), session)
        for suffix in ("transcript", "slide"):
            await self.bus.client.delete(f"lecturelens:{session}:stream:{suffix}")
        base = time.time()
        await self.bus.publish(
            f"lecturelens:{session}:stream:transcript",
            TranscriptChunk(
                session_id=session, timestamp_epoch=base, start_seconds=0.0, end_seconds=3.0,
                text="explaining the tree", confidence=0.9, importance_flags=["definition"]
            )
        )
        await self.bus.publish(
            f"lecturelens:{session}:stream:slide",
            SlideCaptured(
                session_id=session, timestamp_epoch=base + 1.0, slide_number=1, monitor_index=1,
                image_file_path="s.png", extracted_text="Trees", revisited_slide_number=0
            )
        )

        async def stopper() -> None:
            await asyncio.sleep(1.0)
            set_flag(agent.control.session_dir, SYNC_STOP_FLAG_NAME)

        await asyncio.wait_for(asyncio.gather(agent.run(), stopper()), timeout=30)

        lines: list[dict] = []
        for raw_line in agent.timeline_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                lines.append(json.loads(raw_line))
        self.assertEqual(len(lines), 2)
        kinds = pluck(lines, "source_type")
        self.assertEqual(kinds, ["transcript", "slide"])
        correlation_ids = set(pluck(lines, "correlation_id"))
        self.assertEqual(len(correlation_ids), 1)
        self.assertIsNotNone(next(iter(correlation_ids)))
        self.assertEqual(lines[0]["importance_flags"], ["definition"])
        self.assertEqual(lines[1]["metadata"]["slide_number"], 1)
        for suffix in ("transcript", "slide"):
            await self.bus.client.delete(f"lecturelens:{session}:stream:{suffix}")

    async def test_restart_does_not_duplicate_entries(self) -> None:
        try:
            await self.bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")
        session = "sync_restart_session"
        for suffix in ("transcript", "slide"):
            await self.bus.client.delete(f"lecturelens:{session}:stream:{suffix}")
        await self.bus.publish(
            f"lecturelens:{session}:stream:transcript",
            TranscriptChunk(
                session_id=session, timestamp_epoch=time.time(), start_seconds=0.0, end_seconds=3.0,
                text="only once", confidence=0.9
            )
        )
        for _ in range(2):
            agent = SyncAgent(self.settings, self.bus, SessionClock(), session)
            set_flag(agent.control.session_dir, SYNC_STOP_FLAG_NAME)
            await asyncio.wait_for(agent.run(), timeout=30)
            (agent.control.session_dir / "control" / SYNC_STOP_FLAG_NAME).unlink()
        lines = (self.settings.output_dir / session / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        for suffix in ("transcript", "slide"):
            await self.bus.client.delete(f"lecturelens:{session}:stream:{suffix}")

    async def test_a_slide_shares_the_correlation_id_of_a_nearby_transcript(self) -> None:
        """A slide used to look for a cross type anchor in a list that only
        ever holds slides, so it always got a fresh uuid and the link between
        a slide and the speech next to it only existed in one direction."""
        agent = SyncAgent(self.settings, self.bus, SessionClock(), "test_sync_correlation_session")
        entries = [
            entry(100.0, "transcript", "as you can see on this slide"),
            entry(101.0, "slide", "Binary Search O(log n)")
        ]
        agent.assign_correlation_ids(entries)
        self.assertIsNotNone(entries[0].correlation_id)
        self.assertEqual(entries[0].correlation_id, entries[1].correlation_id)

    async def test_a_distant_slide_gets_its_own_correlation_id(self) -> None:
        agent = SyncAgent(self.settings, self.bus, SessionClock(), "test_sync_correlation_far_session")
        entries = [
            entry(100.0, "transcript", "as you can see on this slide"),
            entry(400.0, "slide", "A completely different slide")
        ]
        agent.assign_correlation_ids(entries)
        self.assertNotEqual(entries[0].correlation_id, entries[1].correlation_id)

    async def test_flush_ready_entries_retains_buffer_when_publish_fails(self) -> None:
        agent = SyncAgent(self.settings, self.bus, SessionClock(), "test_sync_fail_session")
        ent = entry(100.0, "transcript", "test fail")
        buf = BufferedEntry(
            entry=ent,
            arrival_monotonic=time.monotonic() - 10.0,
            stream_name="s",
            group_name="g",
            message_id="1-0"
        )
        agent.pending_buffer.append(buf)
        agent.acknowledge = mock.AsyncMock()

        agent.redis_bus.publish_resilient = mock.AsyncMock(side_effect=RuntimeError("outbox disk full"))

        flushed = await agent.flush_ready_entries(force=True)
        self.assertEqual(flushed, 0)
        self.assertEqual(len(agent.pending_buffer), 1)
        self.assertFalse(agent.acknowledge.called)

        # The durable local sink is written first, so the entry is already in
        # timeline.jsonl. The retry must not write it a second time.
        self.assertTrue(agent.timeline_path.exists())
        first_pass_lines = agent.timeline_path.read_text(encoding="utf-8").splitlines()

        agent.redis_bus.publish_resilient = mock.AsyncMock(return_value="1-0")
        flushed = await agent.flush_ready_entries(force=True)
        self.assertEqual(flushed, 1)
        self.assertEqual(len(agent.pending_buffer), 0)

        # The file still holds exactly one line: the retry published to
        # Redis but did not append to the file again.
        second_pass_lines = agent.timeline_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first_pass_lines, second_pass_lines)
        self.assertEqual(len(second_pass_lines), 1)

    async def test_partial_publish_failure_does_not_republish_the_first_entries(self) -> None:
        """The per entry state was only updated after the whole loop, so the
        entries that were published before the failure were published again
        on the retry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.sync_merge_window_seconds = 0.0
            agent = SyncAgent(settings, self.bus, SessionClock(), "sync_partial_session")
            agent.acknowledge = mock.AsyncMock()

            for index in range(4):
                agent.pending_buffer.append(BufferedEntry(
                    entry=entry(100.0 + index, "transcript", f"partial {index}"),
                    arrival_monotonic=time.monotonic() - 10.0,
                    stream_name="s",
                    group_name="g",
                    message_id=f"1-{index}",
                ))

            published: list[str] = []
            call_count = {"value": 0}

            async def failing_publish(stream: str, item: object) -> str:
                call_count["value"] += 1
                if call_count["value"] > 2:
                    raise RuntimeError("redis is down")
                published.append(str(getattr(item, "content", "")))
                return "1-0"

            agent.redis_bus.publish_resilient = failing_publish
            flushed = await agent.flush_ready_entries(force=True)
            self.assertEqual(flushed, 0)
            self.assertEqual(len(published), 2)

            # The retry publishes only the entries that never made it.
            published.clear()
            call_count["value"] = 0
            agent.redis_bus.publish_resilient = mock.AsyncMock(
                side_effect=lambda stream, item: published.append(str(getattr(item, "content", ""))) or "1-0"
            )
            flushed = await agent.flush_ready_entries(force=True)
            self.assertEqual(flushed, 4)
            self.assertEqual(len(published), 2, "the already published entries must not be published again")
            self.assertNotIn("partial 0", published)
            self.assertNotIn("partial 1", published)

    async def test_crash_after_the_file_write_republishes_to_redis(self) -> None:
        """A crash between the file write and the Redis publish used to lose
        the entry: the restart restored the fingerprint from timeline.jsonl and
        treated the entry as done, so the persister never saw it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.sync_merge_window_seconds = 0.0
            session = "sync_crash_session"

            agent = SyncAgent(settings, self.bus, SessionClock(), session)
            agent.acknowledge = mock.AsyncMock()
            ent = entry(100.0, "transcript", "crash between sinks")
            agent.pending_buffer.append(BufferedEntry(
                entry=ent,
                arrival_monotonic=time.monotonic() - 10.0,
                stream_name="s",
                group_name="g",
                message_id="1-0",
            ))

            # The file write succeeds, the publish never happens.
            async def failing_publish(*args: object, **kwargs: object) -> str:
                raise RuntimeError("redis is down")

            agent.redis_bus.publish_resilient = failing_publish
            flushed = await agent.flush_ready_entries(force=True)
            self.assertEqual(flushed, 0)
            self.assertTrue(agent.timeline_path.exists())
            self.assertTrue(agent.flush_state_path.exists())

            # A restart reloads the state from disk.
            restarted = SyncAgent(settings, self.bus, SessionClock(), session)
            self.assertTrue(restarted.flush_state_tracked)

            published: list[Any] = []
            restarted.acknowledge = mock.AsyncMock()
            restarted.redis_bus.publish_resilient = mock.AsyncMock(
                side_effect=lambda stream, item: published.append(item) or "1-0"
            )

            # The entry is delivered again, as the unacked message would be.
            await restarted.add_entry(ent, stream_name="s", group_name="g", message_id="1-0")
            self.assertEqual(len(restarted.pending_buffer), 1)

            flushed = await restarted.flush_ready_entries(force=True)
            self.assertEqual(flushed, 1)
            self.assertEqual(len(published), 1, "the entry has to reach Redis after the restart")

            # The file must not have been written a second time.
            lines = restarted.timeline_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

    async def test_sync_agent_deduplicates_redelivered_event_id(self) -> None:
        agent = SyncAgent(self.settings, self.bus, SessionClock(), "test_sync_dedup_session")
        agent.acknowledge = mock.AsyncMock()
        chunk = TranscriptChunk(
            session_id="test_sync_dedup_session",
            event_id="chunk_dup_1",
            timestamp_epoch=100.0,
            start_seconds=0.0,
            end_seconds=2.0,
            text="First and only",
            confidence=0.9
        )
        entry1 = agent.transcript_to_entry(chunk)
        self.assertEqual(entry1.event_id, "chunk_dup_1")
        await agent.add_entry(entry1, stream_name="stream_tr", group_name="grp", message_id="1-0")
        self.assertEqual(len(agent.pending_buffer), 1)

        flushed = await agent.flush_ready_entries(force=True)
        self.assertEqual(flushed, 1)

        entry2 = agent.transcript_to_entry(chunk)
        await agent.add_entry(entry2, stream_name="stream_tr", group_name="grp", message_id="2-0")
        self.assertEqual(agent.duplicates_skipped, 1)
        self.assertEqual(len(agent.pending_buffer), 0)
        agent.acknowledge.assert_called_with("stream_tr", "grp", "2-0")

        lines = agent.timeline_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)

    def test_written_event_ids_bounded_by_fingerprint_history_limit(self) -> None:
        agent = SyncAgent(self.settings, self.bus, SessionClock(), "test_sync_bound_session")
        agent.fingerprint_history_limit = 5
        for i in range(10):
            agent.remember_event_id(f"evt_{i}")
        self.assertEqual(len(agent.written_event_ids), 5)
        self.assertNotIn("evt_0", agent.written_event_ids)
        self.assertNotIn("evt_4", agent.written_event_ids)
        self.assertIn("evt_5", agent.written_event_ids)
        self.assertIn("evt_9", agent.written_event_ids)


if __name__ == "__main__":
    unittest.main()
