"""Opt-in failure injection tests against a live Redis.

These are skipped unless LECTURELENS_INTEGRATION_REDIS is set, because they
need a real Redis and they kill a child process on purpose.

    LECTURELENS_INTEGRATION_REDIS=redis://127.0.0.1:6379 pytest tests/test_integration_failures.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

from lecturelens.agents.sync_agent import SyncAgent
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus

REPO_ROOT = Path(__file__).resolve().parents[1]
REDIS_URL = os.environ.get("LECTURELENS_INTEGRATION_REDIS", "").strip()


def run_fault_runner(session_id: str, output_dir: Path, count: int, fault: str) -> subprocess.CompletedProcess:
    """Run the helper process, crashing hard at the named fault point."""
    environment = dict(os.environ)
    environment["LECTURELENS_FAULT"] = fault
    environment["LECTURELENS_OUTPUT_DIR"] = str(output_dir)
    environment["LECTURELENS_REDIS_URL"] = REDIS_URL
    environment["LECTURELENS_SYNC_MERGE_WINDOW_SECONDS"] = "0"
    return subprocess.run(
        [
            sys.executable, "-m", "tests.support_sync_fault",
            session_id, str(output_dir), REDIS_URL, str(count),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def read_timeline_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def read_stream_entries(bus: RedisBus, session_id: str) -> list[dict[str, Any]]:
    stream_name = f"lecturelens:{session_id}:stream:timeline"
    entries = []
    for _, data in await bus.client.xrange(stream_name, "-", "+"):
        entries.append(data)
    return entries


@unittest.skipUnless(REDIS_URL, "set LECTURELENS_INTEGRATION_REDIS to run the failure injection tests")
class TestSyncFailureInjection(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.session_id = "itest_" + uuid.uuid4().hex[:8]
        self.settings = Settings()
        self.settings.output_dir = self.output_dir
        self.settings.redis_url = REDIS_URL
        self.settings.sync_merge_window_seconds = 0.0
        self.bus = RedisBus(REDIS_URL)
        self.timeline_path = self.output_dir / self.session_id / "timeline.jsonl"
        self.flush_state_path = self.output_dir / self.session_id / "timeline_flush_state.json"

    async def asyncTearDown(self) -> None:
        for suffix in ("transcript", "timeline"):
            stream = f"lecturelens:{self.session_id}:stream:{suffix}"
            try:
                await self.bus.client.delete(stream)
            except Exception:
                pass
        await self.bus.close()
        self.temp_dir.cleanup()

    async def test_crash_after_the_file_write_publishes_exactly_once(self) -> None:
        """The agent crashes after the durable file write and before the
        Redis publish. After the restart the entry has to be in the file
        exactly once and in the stream exactly once."""
        count = 5
        result = run_fault_runner(self.session_id, self.output_dir, count, "crash_after_file_write")
        self.assertNotEqual(result.returncode, 0, "the helper was expected to crash")

        # The file write happened, the publish did not.
        lines = read_timeline_lines(self.timeline_path)
        self.assertEqual(len(lines), count)
        self.assertTrue(self.flush_state_path.exists())
        self.assertEqual(len(await read_stream_entries(self.bus, self.session_id)), 0)

        # A restart picks the entries up again, exactly as the unacked Redis
        # messages would.
        restarted = SyncAgent(self.settings, self.bus, SessionClock(), self.session_id)
        published: list[Any] = []
        restarted.redis_bus.publish_resilient = _collect_publishes(published)
        for index in range(count):
            entry = restarted.transcript_to_entry(
                _chunk(self.session_id, index)
            )
            await restarted.add_entry(
                entry,
                stream_name=restarted.build_stream_name("transcript"),
                group_name=restarted.consumer_group,
                message_id=entry.event_id
            )
        flushed = await restarted.flush_ready_entries(force=True)
        self.assertEqual(flushed, count)

        # The file still has every entry exactly once.
        self.assertEqual(len(read_timeline_lines(self.timeline_path)), count)
        # The stream has every entry exactly once.
        self.assertEqual(len(published), count)
        # The state file is gone once nothing is pending.
        self.assertFalse(self.flush_state_path.exists())

    async def test_crash_after_the_state_pop_does_not_duplicate_the_file(self) -> None:
        """The second window: the entries are finished but the state was not
        released yet. The file must not gain a duplicate."""
        count = 4
        result = run_fault_runner(self.session_id, self.output_dir, count, "crash_after_state_pop")
        self.assertNotEqual(result.returncode, 0, "the helper was expected to crash")

        lines = read_timeline_lines(self.timeline_path)
        self.assertEqual(len(lines), count)
        # The publish already happened before this window.
        self.assertEqual(len(await read_stream_entries(self.bus, self.session_id)), count)

        restarted = SyncAgent(self.settings, self.bus, SessionClock(), self.session_id)
        published: list[Any] = []
        restarted.redis_bus.publish_resilient = _collect_publishes(published)
        for index in range(count):
            entry = restarted.transcript_to_entry(_chunk(self.session_id, index))
            await restarted.add_entry(
                entry,
                stream_name=restarted.build_stream_name("transcript"),
                group_name=restarted.consumer_group,
                message_id=entry.event_id
            )
        await restarted.flush_ready_entries(force=True)

        # No duplicates in the file.
        self.assertEqual(len(read_timeline_lines(self.timeline_path)), count)
        # And the entries that were already published are not published again.
        self.assertEqual(len(published), 0)
        self.assertFalse(self.flush_state_path.exists())


def _chunk(session_id: str, index: int) -> Any:
    from lecturelens.schemas.events import TranscriptChunk
    return TranscriptChunk(
        session_id=session_id,
        event_id=f"itest_chunk_{index}",
        timestamp_epoch=1_700_000_000.0 + index,
        start_seconds=float(index),
        end_seconds=float(index) + 1.0,
        text=f"integration entry {index}",
        confidence=0.9
    )


def _collect_publishes(sink: list[Any]) -> Any:
    async def publish(stream_name: str, event: Any) -> str:
        sink.append(event)
        return "1-0"
    return publish
