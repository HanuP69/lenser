import asyncio
import json
import uuid
from pathlib import Path
from typing import Any
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import (
    TranscriptChunk,
    SlideCaptured,
    BoardCaptured,
    TimelineEntry
)


class SyncAgent:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.session_clock: SessionClock = session_clock
        self.session_id: str = session_id
        self.group_name: str = "group:kuudere"
        self.consumer_name: str = f"consumer:kuudere:{uuid.uuid4().hex[:8]}"

        self.transcript_stream: str = f"lecturelens:{session_id}:stream:transcript"
        self.slide_stream: str = f"lecturelens:{session_id}:stream:slide"
        self.board_stream: str = f"lecturelens:{session_id}:stream:board"

        self.pending_buffer: list[TimelineEntry] = []
        self.buffer_lock: asyncio.Lock = asyncio.Lock()
        self.is_running: bool = False

        session_dir = self.settings.get_session_output_dir(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_file_path: Path = session_dir / "timeline.jsonl"

    async def handle_transcript_message(self, message_data: dict[str, str]) -> None:
        chunk = TranscriptChunk.from_redis_dict(message_data)
        entry = TimelineEntry(
            session_id=self.session_id,
            timestamp_epoch=chunk.timestamp_epoch,
            source_type="transcript",
            content=chunk.text,
            correlation_id=None,
            importance_flags=chunk.importance_flags,
            metadata={
                "start_seconds": chunk.start_seconds,
                "end_seconds": chunk.end_seconds,
                "confidence": chunk.confidence,
                "language": chunk.language,
                "audio_source": chunk.audio_source,
                "speaker_id": chunk.speaker_id,
                "raw_audio_path": chunk.raw_audio_path
            }
        )
        async with self.buffer_lock:
            self.pending_buffer.append(entry)

    async def handle_slide_message(self, message_data: dict[str, str]) -> None:
        slide = SlideCaptured.from_redis_dict(message_data)
        entry = TimelineEntry(
            session_id=self.session_id,
            timestamp_epoch=slide.timestamp_epoch,
            source_type="slide",
            content=slide.extracted_text,
            correlation_id=None,
            importance_flags=[],
            metadata={
                "slide_number": slide.slide_number,
                "monitor_index": slide.monitor_index,
                "image_file_path": slide.image_file_path
            }
        )
        async with self.buffer_lock:
            self.pending_buffer.append(entry)

    async def handle_board_message(self, message_data: dict[str, str]) -> None:
        board = BoardCaptured.from_redis_dict(message_data)
        entry = TimelineEntry(
            session_id=self.session_id,
            timestamp_epoch=board.timestamp_epoch,
            source_type="board",
            content=board.raw_ocr_text,
            correlation_id=None,
            importance_flags=[],
            metadata={
                "image_file_path": board.image_file_path,
                "confidence": board.confidence,
                "provider_used": board.provider_used
            }
        )
        async with self.buffer_lock:
            self.pending_buffer.append(entry)

    def correlate_and_extract_ready_entries(self, current_time: float) -> list[TimelineEntry]:
        ready_entries: list[TimelineEntry] = []
        remaining_entries: list[TimelineEntry] = []

        self.pending_buffer.sort(key=lambda entry: entry.timestamp_epoch)

        for entry in self.pending_buffer:
            age_seconds = current_time - entry.timestamp_epoch
            if age_seconds >= self.settings.sync_merge_window_seconds:
                ready_entries.append(entry)
            else:
                remaining_entries.append(entry)

        self.pending_buffer = remaining_entries

        for i in range(len(ready_entries)):
            current = ready_entries[i]
            if current.correlation_id is not None:
                continue
            for j in range(i + 1, len(ready_entries)):
                other = ready_entries[j]
                time_difference = abs(current.timestamp_epoch - other.timestamp_epoch)
                if time_difference <= self.settings.correlation_window_seconds:
                    if current.source_type != other.source_type:
                        if current.correlation_id is None:
                            shared_id = str(uuid.uuid4())
                            current.correlation_id = shared_id
                        other.correlation_id = current.correlation_id

        return ready_entries

    async def flush_loop(self) -> None:
        while self.is_running:
            await asyncio.sleep(1.0)
            current_time = self.session_clock.get_timestamp()

            async with self.buffer_lock:
                ready = self.correlate_and_extract_ready_entries(current_time)

            if len(ready) == 0:
                continue

            with open(self.timeline_file_path, "a", encoding="utf-8") as file_handle:
                for entry in ready:
                    line = json.dumps(entry.to_jsonl_dict(), ensure_ascii=False)
                    file_handle.write(line + "\n")

    async def run(self) -> None:
        self.is_running = True
        flush_task = asyncio.create_task(self.flush_loop())

        transcript_task = asyncio.create_task(
            self.redis_bus.consume(
                stream_name=self.transcript_stream,
                group_name=self.group_name,
                consumer_name=self.consumer_name,
                handler=self.handle_transcript_message,
                batch_size=self.settings.redis_consumer_batch_size,
                block_ms=self.settings.redis_consumer_block_ms
            )
        )

        slide_task = asyncio.create_task(
            self.redis_bus.consume(
                stream_name=self.slide_stream,
                group_name=self.group_name,
                consumer_name=self.consumer_name,
                handler=self.handle_slide_message,
                batch_size=self.settings.redis_consumer_batch_size,
                block_ms=self.settings.redis_consumer_block_ms
            )
        )

        board_task = asyncio.create_task(
            self.redis_bus.consume(
                stream_name=self.board_stream,
                group_name=self.group_name,
                consumer_name=self.consumer_name,
                handler=self.handle_board_message,
                batch_size=self.settings.redis_consumer_batch_size,
                block_ms=self.settings.redis_consumer_block_ms
            )
        )

        try:
            await asyncio.gather(flush_task, transcript_task, slide_task, board_task)
        except asyncio.CancelledError:
            self.is_running = False

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    args, _ = parser.parse_known_args()

    settings = Settings()
    if args.session is not None and len(args.session) > 0:
        settings.session_name = args.session

    session_id = settings.session_name or "default_session"
    clock_file = settings.get_session_output_dir(session_id) / "clock.json"
    if clock_file.exists():
        session_clock = SessionClock.load_from_file(clock_file)
    else:
        session_clock = SessionClock()

    redis_bus = RedisBus(settings.redis_url)
    agent = SyncAgent(
        settings=settings,
        redis_bus=redis_bus,
        session_clock=session_clock,
        session_id=session_id
    )
    try:
        await agent.run()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
