import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl,
    RuntimeContext,
    SYNC_STOP_FLAG_NAME,
    execute_agent_main,
    run_standard_agent
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.fileio import append_text_durably
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import (
    BoardCaptured,
    SlideCaptured,
    TimelineEntry,
    TranscriptChunk
)

logger = get_logger("lecturelens.sync_agent")

ANCHOR_SOURCE_TYPES: tuple[str, ...] = ("slide", "board")


@dataclass
class BufferedEntry:
    entry: TimelineEntry
    arrival_monotonic: float
    stream_name: Optional[str] = None
    group_name: Optional[str] = None
    message_id: Optional[str] = None


@dataclass
class AnchorRecord:
    timestamp_epoch: float
    correlation_id: str
    source_type: str


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
        self.is_running: bool = False
        self.drain_requested: bool = False

        session_dir = settings.create_session_directories(session_id)
        self.timeline_path: Path = session_dir / "timeline.jsonl"
        self.control: AgentControl = AgentControl(session_dir, "sync_agent", SYNC_STOP_FLAG_NAME)
        self.pending_buffer: list[BufferedEntry] = []
        self.recent_anchors: list[AnchorRecord] = []
        self.buffer_lock: asyncio.Lock = asyncio.Lock()
        self.entries_written: int = 0

    def build_stream_name(self, source_type: str) -> str:
        return f"lecturelens:{self.session_id}:stream:{source_type}"

    def transcript_to_entry(self, event: TranscriptChunk) -> TimelineEntry:
        metadata = {
            "confidence": event.confidence,
            "language": event.language,
            "audio_source": event.audio_source,
            "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds,
            "raw_audio_path": event.raw_audio_path
        }
        return TimelineEntry(
            session_id=event.session_id,
            timestamp_epoch=event.timestamp_epoch,
            source_type="transcript",
            content=event.text,
            correlation_id=None,
            importance_flags=list(event.importance_flags),
            metadata=metadata
        )

    def slide_to_entry(self, event: SlideCaptured) -> TimelineEntry:
        metadata = {
            "slide_number": event.slide_number,
            "monitor_index": event.monitor_index,
            "image_file_path": event.image_file_path,
            "revisited_slide_number": event.revisited_slide_number
        }
        return TimelineEntry(
            session_id=event.session_id,
            timestamp_epoch=event.timestamp_epoch,
            source_type="slide",
            content=event.extracted_text,
            correlation_id=None,
            importance_flags=[],
            metadata=metadata
        )

    def board_to_entry(self, event: BoardCaptured) -> TimelineEntry:
        metadata = {
            "confidence": event.confidence,
            "provider_used": event.provider_used,
            "image_file_path": event.image_file_path
        }
        return TimelineEntry(
            session_id=event.session_id,
            timestamp_epoch=event.timestamp_epoch,
            source_type="board",
            content=event.raw_ocr_text,
            correlation_id=None,
            importance_flags=[],
            metadata=metadata
        )

    def convert_message(self, source_type: str, message_data: dict[str, str]) -> TimelineEntry:
        if source_type == "transcript":
            return self.transcript_to_entry(TranscriptChunk.from_redis_dict(message_data))
        if source_type == "slide":
            return self.slide_to_entry(SlideCaptured.from_redis_dict(message_data))
        if source_type == "board":
            return self.board_to_entry(BoardCaptured.from_redis_dict(message_data))
        raise ValueError(f"unknown source type {source_type}")

    async def add_entry(
        self,
        entry: TimelineEntry,
        stream_name: Optional[str] = None,
        group_name: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        async with self.buffer_lock:
            self.pending_buffer.append(
                BufferedEntry(entry, time.monotonic(), stream_name, group_name, message_id)
            )

    async def consume_stream(self, source_type: str) -> None:
        stream_name = self.build_stream_name(source_type)

        async def handle_message(message_data: dict[str, str], message_id: str) -> None:
            entry = self.convert_message(source_type, message_data)
            await self.add_entry(
                entry,
                stream_name=stream_name,
                group_name="group:kuudere",
                message_id=message_id
            )

        await self.redis_bus.consume(
            stream_name=stream_name,
            group_name="group:kuudere",
            consumer_name=f"consumer:kuudere:{source_type}",
            handler=handle_message,
            batch_size=self.settings.redis_consumer_batch_size,
            block_ms=self.settings.redis_consumer_block_ms,
            should_stop=lambda: self.drain_requested,
            auto_ack=False
        )

    def find_nearest_anchor(self, timestamp_epoch: float, window_seconds: float) -> Optional[AnchorRecord]:
        best_record: Optional[AnchorRecord] = None
        best_distance = window_seconds + 1.0
        for record in self.recent_anchors:
            distance = abs(record.timestamp_epoch - timestamp_epoch)
            if distance <= window_seconds and distance < best_distance:
                best_distance = distance
                best_record = record
        return best_record

    def find_cross_type_anchor(
        self,
        timestamp_epoch: float,
        source_type: str,
        window_seconds: float
    ) -> Optional[AnchorRecord]:
        best_record: Optional[AnchorRecord] = None
        best_distance = window_seconds + 1.0
        for record in self.recent_anchors:
            if record.source_type == source_type:
                continue
            distance = abs(record.timestamp_epoch - timestamp_epoch)
            if distance <= window_seconds and distance < best_distance:
                best_distance = distance
                best_record = record
        return best_record

    def prune_anchor_history(self, newest_timestamp: float, window_seconds: float) -> None:
        horizon = max(60.0, window_seconds * 6.0)
        kept: list[AnchorRecord] = []
        for record in self.recent_anchors:
            if newest_timestamp - record.timestamp_epoch <= horizon:
                kept.append(record)
        self.recent_anchors = kept

    def assign_correlation_ids(self, ordered_entries: list[TimelineEntry]) -> None:
        window_seconds = self.settings.correlation_window_seconds
        known_ids: set[str] = set()
        for record in self.recent_anchors:
            known_ids.add(record.correlation_id)

        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES:
                continue
            if entry.correlation_id is None:
                neighbor = self.find_cross_type_anchor(
                    entry.timestamp_epoch, entry.source_type, window_seconds
                )
                if neighbor is not None:
                    entry.correlation_id = neighbor.correlation_id
                else:
                    entry.correlation_id = str(uuid.uuid4())
            already_recorded = False
            for record in self.recent_anchors:
                if (
                    record.correlation_id == entry.correlation_id
                    and record.source_type == entry.source_type
                    and record.timestamp_epoch == entry.timestamp_epoch
                ):
                    already_recorded = True
                    break
            if not already_recorded:
                self.recent_anchors.append(
                    AnchorRecord(entry.timestamp_epoch, entry.correlation_id, entry.source_type)
                )

        for entry in ordered_entries:
            if entry.source_type in ANCHOR_SOURCE_TYPES:
                continue
            if entry.correlation_id is not None:
                continue
            anchor = self.find_nearest_anchor(entry.timestamp_epoch, window_seconds)
            if anchor is not None:
                entry.correlation_id = anchor.correlation_id

        if len(ordered_entries) > 0:
            self.prune_anchor_history(ordered_entries[-1].timestamp_epoch, window_seconds)

    def select_ready_entries(self, now_monotonic: float, force: bool) -> list[BufferedEntry]:
        ready: list[BufferedEntry] = []
        merge_window = self.settings.sync_merge_window_seconds
        for buffered in self.pending_buffer:
            if force or (now_monotonic - buffered.arrival_monotonic) >= merge_window:
                ready.append(buffered)
        ready.sort(key=lambda item: item.entry.timestamp_epoch)
        return ready

    def remove_flushed_entries(self, flushed: list[BufferedEntry]) -> None:
        flushed_identities: set[int] = set()
        for buffered in flushed:
            flushed_identities.add(id(buffered))
        remaining: list[BufferedEntry] = []
        for buffered in self.pending_buffer:
            if id(buffered) not in flushed_identities:
                remaining.append(buffered)
        self.pending_buffer = remaining

    def render_lines(self, entries: list[TimelineEntry]) -> str:
        lines: list[str] = []
        for entry in entries:
            lines.append(json.dumps(entry.model_dump(), ensure_ascii=False))
        return "\n".join(lines) + "\n"

    async def flush_ready_entries(self, force: bool = False) -> int:
        async with self.buffer_lock:
            ready = self.select_ready_entries(time.monotonic(), force)
            if len(ready) == 0:
                return 0
            ordered_entries: list[TimelineEntry] = []
            for buffered in ready:
                ordered_entries.append(buffered.entry)
            self.assign_correlation_ids(ordered_entries)
            text_block = self.render_lines(ordered_entries)
            try:
                await run_in_thread_pool(append_text_durably, self.timeline_path, text_block)
            except OSError as error:
                logger.error("timeline write failed, entries kept in buffer: %s", error)
                return 0
            for buffered in ready:
                if buffered.stream_name and buffered.group_name and buffered.message_id:
                    try:
                        await self.redis_bus.client.xack(
                            buffered.stream_name, buffered.group_name, buffered.message_id
                        )
                    except Exception as ack_err:
                        logger.warning("failed to ack message %s: %s", buffered.message_id, ack_err)
            self.remove_flushed_entries(ready)
            self.entries_written += len(ordered_entries)
            self.control.set_stat("timeline_entries_written", self.entries_written)
            self.control.set_stat("buffered_entries", len(self.pending_buffer))
            return len(ordered_entries)

    async def flush_loop(self) -> None:
        while True:
            try:
                await self.flush_ready_entries(force=False)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("flush loop error: %s", error)
            await asyncio.sleep(1.0)

    async def watch_stop_flag(self) -> None:
        while not self.control.stop_requested():
            await asyncio.sleep(0.5)
        self.drain_requested = True

    async def run(self) -> None:
        self.is_running = True
        consumer_tasks: list[asyncio.Task] = []
        for source_type in ("transcript", "slide", "board"):
            consumer_tasks.append(asyncio.create_task(self.consume_stream(source_type)))
        flush_task = asyncio.create_task(self.flush_loop())
        stop_watch_task = asyncio.create_task(self.watch_stop_flag())
        try:
            await asyncio.gather(*consumer_tasks)
        finally:
            self.is_running = False
            flush_task.cancel()
            stop_watch_task.cancel()
            await asyncio.gather(flush_task, stop_watch_task, return_exceptions=True)
            for task in consumer_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*consumer_tasks, return_exceptions=True)
            await self.flush_ready_entries(force=True)

    def stop(self) -> None:
        self.drain_requested = True


async def main() -> None:
    def build_agent(context: RuntimeContext) -> SyncAgent:
        return SyncAgent(
            settings=context.settings,
            redis_bus=context.redis_bus,
            session_clock=context.session_clock,
            session_id=context.session_id
        )

    await run_standard_agent(build_agent)


if __name__ == "__main__":
    execute_agent_main(main)
