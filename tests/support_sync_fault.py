"""Helper process for the opt-in integration tests.

It runs the real sync agent flush path against a live Redis and can crash
hard at a named fault point, so a test can reproduce a crash in a specific
window. A timed kill would be flaky, so the crash is deterministic.

Usage: python -m tests.support_sync_fault <session_id> <output_dir> <redis_url> <count>
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lecturelens.agents.sync_agent import SyncAgent  # noqa: E402
from lecturelens.config.settings import Settings  # noqa: E402
from lecturelens.core.clock import SessionClock  # noqa: E402
from lecturelens.core.redis_bus import RedisBus  # noqa: E402
from lecturelens.schemas.events import TranscriptChunk  # noqa: E402


def build_entries(session_id: str, count: int) -> list[TranscriptChunk]:
    return [
        TranscriptChunk(
            session_id=session_id,
            event_id=f"itest_chunk_{index}",
            timestamp_epoch=1_700_000_000.0 + index,
            start_seconds=float(index),
            end_seconds=float(index) + 1.0,
            text=f"integration entry {index}",
            confidence=0.9
        )
        for index in range(count)
    ]


async def main() -> None:
    session_id = sys.argv[1]
    output_dir = Path(sys.argv[2])
    redis_url = sys.argv[3]
    count = int(sys.argv[4])

    settings = Settings()
    settings.output_dir = output_dir
    settings.redis_url = redis_url
    settings.sync_merge_window_seconds = 0.0

    bus = RedisBus(settings.redis_url)
    agent = SyncAgent(settings, bus, SessionClock(), session_id)
    stream_name = agent.build_stream_name("transcript")

    try:
        for chunk in build_entries(session_id, count):
            entry = agent.transcript_to_entry(chunk)
            await agent.add_entry(
                entry,
                stream_name=stream_name,
                group_name=agent.consumer_group,
                message_id=chunk.event_id
            )
        await agent.flush_ready_entries(force=True)
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
