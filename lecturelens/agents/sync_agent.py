import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lecturelens.config.defaults import SYNC_CONSUMER_GROUP, SYNC_CONSUMER_PREFIX
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
    SlideCaptured,
    TimelineEntry,
    TranscriptChunk
)

logger = get_logger("lecturelens.sync_agent")

ANCHOR_SOURCE_TYPES: tuple[str, ...] = ("slide",)
FINGERPRINT_HISTORY_LIMIT: int = 5000


def entry_fingerprint(entry: TimelineEntry) -> str:
    raw = f"{entry.source_type}|{entry.timestamp_epoch:.6f}|{entry.content}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


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
        self.pending_message_ids: set[tuple[str, str]] = set()
        self.written_fingerprints: set[str] = set()
        # Tracks how far each entry made it into the two sinks, so a failure
        # in one of them does not make the other one run twice.
        self.flush_states: dict[str, str] = {}
        self.written_event_ids: set[str] = set()
        self.fingerprint_order: deque = deque()
        self.event_id_order: deque = deque()
        self.duplicates_skipped: int = 0
        self.recent_anchors: list[AnchorRecord] = []
        self.buffer_lock: asyncio.Lock = asyncio.Lock()
        self.entries_written: int = 0
        self.fingerprint_history_limit: int = getattr(settings, "sync_fingerprint_history_limit", getattr(settings, "fingerprint_history_limit", FINGERPRINT_HISTORY_LIMIT))
        self.load_written_fingerprints()

    def load_written_fingerprints(self) -> None:
        if not self.timeline_path.exists():
            return
        try:
            with open(self.timeline_path, "r", encoding="utf-8") as file_handle:
                tail_lines = deque(file_handle, maxlen=self.fingerprint_history_limit)
        except OSError:
            return
        for raw_line in tail_lines:
            raw_line = raw_line.strip()
            if len(raw_line) == 0:
                continue
            try:
                restored = TimelineEntry(**json.loads(raw_line))
            except (ValueError, TypeError):
                continue
            self.remember_fingerprint(entry_fingerprint(restored))
            if restored.event_id:
                self.remember_event_id(restored.event_id)

    def remember_fingerprint(self, fingerprint: str) -> None:
        if fingerprint in self.written_fingerprints:
            return
        self.written_fingerprints.add(fingerprint)
        self.fingerprint_order.append(fingerprint)
        while len(self.fingerprint_order) > self.fingerprint_history_limit:
            self.written_fingerprints.discard(self.fingerprint_order.popleft())

    def remember_event_id(self, event_id: Optional[str]) -> None:
        if not event_id:
            return
        if event_id in self.written_event_ids:
            return
        self.written_event_ids.add(event_id)
        self.event_id_order.append(event_id)
        while len(self.event_id_order) > self.fingerprint_history_limit:
            self.written_event_ids.discard(self.event_id_order.popleft())

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
        elapsed = self.session_clock.elapsed_for_epoch(event.timestamp_epoch)
        return TimelineEntry(
            session_id=event.session_id,
            event_id=event.event_id,
            timestamp_epoch=event.timestamp_epoch,
            elapsed_seconds=elapsed,
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
        elapsed = self.session_clock.elapsed_for_epoch(event.timestamp_epoch)
        return TimelineEntry(
            session_id=event.session_id,
            event_id=event.event_id,
            timestamp_epoch=event.timestamp_epoch,
            elapsed_seconds=elapsed,
            source_type="slide",
            content=event.extracted_text,
            correlation_id=None,
            importance_flags=[],
            metadata=metadata
        )

    def convert_message(self, source_type: str, message_data: dict[str, str]) -> TimelineEntry:
        if source_type == "transcript":
            return self.transcript_to_entry(TranscriptChunk.from_redis_dict(message_data))
        if source_type == "slide":
            return self.slide_to_entry(SlideCaptured.from_redis_dict(message_data))
        raise ValueError(f"unknown source type {source_type}")

    async def add_entry(
        self,
        entry: TimelineEntry,
        stream_name: Optional[str] = None,
        group_name: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        async with self.buffer_lock:
            pending_key = (stream_name or "", message_id) if message_id is not None else None
            if pending_key is not None and pending_key in self.pending_message_ids:
                self.duplicates_skipped += 1
                return
            if entry.event_id and any(b.entry.event_id == entry.event_id for b in self.pending_buffer):
                self.duplicates_skipped += 1
                await self.acknowledge(stream_name, group_name, message_id)
                return
            if entry.event_id and entry.event_id in self.written_event_ids:
                self.duplicates_skipped += 1
                await self.acknowledge(stream_name, group_name, message_id)
                return
            fp = entry_fingerprint(entry)
            if fp in self.written_fingerprints:
                self.duplicates_skipped += 1
                await self.acknowledge(stream_name, group_name, message_id)
                return
            if pending_key is not None:
                self.pending_message_ids.add(pending_key)
            self.pending_buffer.append(
                BufferedEntry(entry, time.monotonic(), stream_name, group_name, message_id)
            )

    async def acknowledge(
        self,
        stream_name: Optional[str],
        group_name: Optional[str],
        message_id: Optional[str]
    ) -> None:
        if not (stream_name and group_name and message_id):
            return
        try:
            await self.redis_bus.client.xack(stream_name, group_name, message_id)
        except Exception as ack_error:
            logger.warning("failed to ack message %s: %s", message_id, ack_error)

    async def consume_stream(self, source_type: str) -> None:
        stream_name = self.build_stream_name(source_type)

        async def handle_message(message_data: dict[str, str], message_id: str) -> None:
            entry = self.convert_message(source_type, message_data)
            await self.add_entry(
                entry,
                stream_name=stream_name,
                group_name=SYNC_CONSUMER_GROUP,
                message_id=message_id
            )

        await self.redis_bus.consume(
            stream_name=stream_name,
            group_name=SYNC_CONSUMER_GROUP,
            consumer_name=f"{SYNC_CONSUMER_PREFIX}:{source_type}",
            handler=handle_message,
            batch_size=self.settings.redis_consumer_batch_size,
            block_ms=self.settings.redis_consumer_block_ms,
            should_stop=lambda: self.drain_requested,
            auto_ack=False,
            reclaim_idle_ms=self.reclaim_idle_milliseconds()
        )

    def reclaim_idle_milliseconds(self) -> int:
        hold_seconds = self.settings.sync_merge_window_seconds + 60.0
        return int(max(hold_seconds, 1.0) * 1000)

    def find_nearest_anchor(self, timestamp_epoch: float, window_seconds: float) -> Optional[AnchorRecord]:
        best_record: Optional[AnchorRecord] = None
        best_distance = window_seconds + 1.0
        for record in self.recent_anchors:
            distance = abs(record.timestamp_epoch - timestamp_epoch)
            if distance <= window_seconds and distance < best_distance:
                best_distance = distance
                best_record = record
        return best_record

    def find_active_anchor(self, timestamp_epoch: float, max_age_seconds: float) -> Optional[AnchorRecord]:
        best_record: Optional[AnchorRecord] = None
        for record in self.recent_anchors:
            age = timestamp_epoch - record.timestamp_epoch
            if age < 0.0 or age > max_age_seconds:
                continue
            if best_record is None or record.timestamp_epoch > best_record.timestamp_epoch:
                best_record = record
        return best_record

    def find_upcoming_anchor(self, timestamp_epoch: float, lead_seconds: float) -> Optional[AnchorRecord]:
        best_record: Optional[AnchorRecord] = None
        for record in self.recent_anchors:
            lead = record.timestamp_epoch - timestamp_epoch
            if lead <= 0.0 or lead > lead_seconds:
                continue
            if best_record is None or record.timestamp_epoch < best_record.timestamp_epoch:
                best_record = record
        return best_record

    def select_anchor_for_speech(self, timestamp_epoch: float) -> Optional[AnchorRecord]:
        window_seconds = self.settings.correlation_window_seconds
        upcoming = self.find_upcoming_anchor(timestamp_epoch, self.settings.correlation_lead_seconds)
        if upcoming is not None:
            return upcoming
        if self.settings.correlation_active_seconds > 0.0:
            active = self.find_active_anchor(timestamp_epoch, self.settings.correlation_active_seconds)
            if active is not None:
                return active
        return self.find_nearest_anchor(timestamp_epoch, window_seconds)

    @staticmethod
    def find_nearest_entry_of_other_type(
        entry: TimelineEntry,
        ordered_entries: list[TimelineEntry],
        window_seconds: float
    ) -> Optional[TimelineEntry]:
        """Find the closest entry that comes from a different source type."""
        best_entry: Optional[TimelineEntry] = None
        best_distance = window_seconds + 1.0
        for candidate in ordered_entries:
            if candidate is entry:
                continue
            if candidate.source_type == entry.source_type:
                continue
            distance = abs(candidate.timestamp_epoch - entry.timestamp_epoch)
            if distance <= window_seconds and distance < best_distance:
                best_distance = distance
                best_entry = candidate
        return best_entry

    def prune_anchor_history(self, newest_timestamp: float, window_seconds: float) -> None:
        horizon = max(60.0, window_seconds * 6.0, self.settings.correlation_active_seconds)
        kept: list[AnchorRecord] = []
        for record in self.recent_anchors:
            if newest_timestamp - record.timestamp_epoch <= horizon:
                kept.append(record)
        self.recent_anchors = kept

    def assign_correlation_ids(self, ordered_entries: list[TimelineEntry]) -> None:
        window_seconds = self.settings.correlation_window_seconds

        # A slide is linked to the closest transcript of the same batch. The
        # slide used to look for a cross type anchor in recent_anchors, but
        # that list only ever holds slides and the lookup skips entries of
        # the same type, so it could never return anything and every slide
        # got a fresh uuid. The link only existed in one direction.
        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES:
                continue
            if entry.correlation_id is not None:
                continue
            neighbour = self.find_nearest_entry_of_other_type(entry, ordered_entries, window_seconds)
            if neighbour is not None:
                if neighbour.correlation_id is None:
                    neighbour.correlation_id = str(uuid.uuid4())
                entry.correlation_id = neighbour.correlation_id

        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES:
                continue
            if entry.correlation_id is None:
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
            anchor = self.select_anchor_for_speech(entry.timestamp_epoch)
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

            # Each entry is tracked separately for the two sinks. The local
            # file and the Redis stream used to be written in one step, so a
            # failure in the second sink made the first one run again and
            # duplicate the timeline.
            fingerprints = [entry_fingerprint(buffered.entry) for buffered in ready]
            needs_file: list[BufferedEntry] = []
            needs_redis: list[BufferedEntry] = []
            for buffered, fingerprint in zip(ready, fingerprints):
                state = self.flush_states.get(fingerprint, "")
                if state not in ("file", "both"):
                    needs_file.append(buffered)
                if state not in ("redis", "both"):
                    needs_redis.append(buffered)

            timeline_stream = f"lecturelens:{self.session_id}:stream:timeline"

            # The durable local sink is written first, so a Redis failure
            # cannot make the same entries be published twice.
            if len(needs_file) > 0:
                file_block = self.render_lines([buffered.entry for buffered in needs_file])
                try:
                    await run_in_thread_pool(append_text_durably, self.timeline_path, file_block)
                except OSError as error:
                    logger.error("timeline write failed, entries kept in buffer: %s", error)
                    return 0
                for buffered in needs_file:
                    fingerprint = entry_fingerprint(buffered.entry)
                    previous = self.flush_states.get(fingerprint, "")
                    self.flush_states[fingerprint] = "both" if previous == "redis" else "file"

            if len(needs_redis) > 0:
                try:
                    for buffered in needs_redis:
                        await self.redis_bus.publish_resilient(timeline_stream, buffered.entry)
                except Exception as pub_error:
                    logger.error("publishing timeline entry failed, keeping buffer unacknowledged: %s", pub_error)
                    return 0
                for buffered in needs_redis:
                    fingerprint = entry_fingerprint(buffered.entry)
                    previous = self.flush_states.get(fingerprint, "")
                    self.flush_states[fingerprint] = "both" if previous == "file" else "redis"

            flushed: list[BufferedEntry] = []
            for buffered, fingerprint in zip(ready, fingerprints):
                if self.flush_states.get(fingerprint) != "both":
                    # Only one of the two sinks accepted the entry, so it
                    # stays in the buffer and is retried for the missing sink.
                    continue
                flushed.append(buffered)
                self.remember_fingerprint(fingerprint)
                self.flush_states.pop(fingerprint, None)
                if buffered.entry.event_id:
                    self.remember_event_id(buffered.entry.event_id)
                if buffered.message_id is not None:
                    self.pending_message_ids.discard((buffered.stream_name or "", buffered.message_id))
                await self.acknowledge(buffered.stream_name, buffered.group_name, buffered.message_id)
            self.remove_flushed_entries(flushed)
            self.entries_written += len(flushed)
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
        for source_type in ("transcript", "slide"):
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
