import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Optional
import uuid

from lecturelens.agents.base_agent import BaseAgent
from lecturelens.config.defaults import SYNC_CONSUMER_GROUP, SYNC_CONSUMER_PREFIX
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl, RuntimeContext, SYNC_STOP_FLAG_NAME, execute_agent_main, run_standard_agent
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.fileio import append_text_durably, atomic_write_text
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import SlideCaptured, TimelineEntry, TranscriptChunk

logger = get_logger("lecturelens.sync_agent")
ANCHOR_SOURCE_TYPES: tuple[str, ...] = ("slide",)
FINGERPRINT_HISTORY_LIMIT: int = 5000


def entry_fingerprint(entry: TimelineEntry) -> str:
    raw = f"{entry.source_type}|{entry.timestamp_epoch:.6f}|{entry.content}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()


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


class SyncAgent(BaseAgent):
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str
    ) -> None:
        super().__init__(settings, redis_bus, session_clock, session_id, "sync_agent")
        self.control = AgentControl(self.session_dir, "sync_agent", SYNC_STOP_FLAG_NAME)
        self.drain_requested: bool = False
        self.timeline_path: Path = self.session_dir / "timeline.jsonl"
        self.flush_state_path: Path = self.session_dir / "timeline_flush_state.json"
        self.pending_buffer: list[BufferedEntry] = []
        self.pending_message_ids: set[tuple[str, str]] = set()
        self.written_fingerprints: set[str] = set()
        self.file_fingerprints: set[str] = set()
        self.file_event_ids: set[str] = set()
        self.flush_states: dict[str, str] = {}
        self.flush_state_tracked: bool = False
        self.written_event_ids: set[str] = set()
        self.fingerprint_order: deque = deque()
        self.event_id_order: deque = deque()
        self.duplicates_skipped: int = 0
        self.recent_anchors: list[AnchorRecord] = []
        self.slide_correlations: dict[int, str] = {}
        self.buffer_lock: asyncio.Lock = asyncio.Lock()
        self.entries_written: int = 0
        self.fingerprint_history_limit: int = settings.sync_fingerprint_history_limit
        self.consumer_group: str = getattr(settings, "sync_consumer_group", SYNC_CONSUMER_GROUP)
        self.consumer_prefix: str = getattr(settings, "sync_consumer_prefix", SYNC_CONSUMER_PREFIX)
        self.load_flush_states()
        self.load_written_fingerprints()

    def load_flush_states(self) -> None:
        if not self.flush_state_path.exists():
            return
        try:
            raw = json.loads(self.flush_state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for fp, st in raw.items():
                    if isinstance(fp, str) and isinstance(st, str):
                        self.flush_states[fp] = st
                self.flush_state_tracked = True
        except (ValueError, OSError):
            pass

    def save_flush_states(self) -> None:
        self.write_flush_states(self.flush_states)

    def write_flush_states(self, states: dict[str, str]) -> None:
        if not states:
            try:
                if self.flush_state_path.exists():
                    self.flush_state_path.unlink()
            except OSError as error:
                logger.warning("could not remove timeline flush state: %s", error)
            return
        try:
            atomic_write_text(self.flush_state_path, json.dumps(states))
            self.flush_state_tracked = True
        except OSError as error:
            logger.warning("could not persist timeline flush state: %s", error)

    @staticmethod
    def trigger_fault(fault_name: str) -> None:
        if os.environ.get("LECTURELENS_FAULT", "") == fault_name:
            logger.error("fault injected: %s", fault_name)
            os._exit(1)

    def load_written_fingerprints(self) -> None:
        if not self.timeline_path.exists():
            return
        try:
            with open(self.timeline_path, "r", encoding="utf-8") as fh:
                tail_lines = deque(fh, maxlen=self.fingerprint_history_limit)
        except OSError:
            return
        for line in tail_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = TimelineEntry(**json.loads(line))
            except (ValueError, TypeError):
                continue
            fp = entry_fingerprint(entry)
            self.file_fingerprints.add(fp)
            if self.flush_states.get(fp) == "file_pending":
                self.flush_states[fp] = "file"
            elif not self.flush_state_tracked:
                self.remember_fingerprint(fp)
            if entry.event_id:
                self.file_event_ids.add(entry.event_id)
                if not self.flush_state_tracked:
                    self.remember_event_id(entry.event_id)

        stale = [fp for fp, st in self.flush_states.items() if st == "file_pending" and fp not in self.file_fingerprints]
        for fp in stale:
            del self.flush_states[fp]
        if stale:
            self.save_flush_states()

    def remember_fingerprint(self, fingerprint: str) -> None:
        if fingerprint in self.written_fingerprints:
            return
        self.written_fingerprints.add(fingerprint)
        self.fingerprint_order.append(fingerprint)
        while len(self.fingerprint_order) > self.fingerprint_history_limit:
            self.written_fingerprints.discard(self.fingerprint_order.popleft())

    def remember_event_id(self, event_id: Optional[str]) -> None:
        if not event_id or event_id in self.written_event_ids:
            return
        self.written_event_ids.add(event_id)
        self.event_id_order.append(event_id)
        while len(self.event_id_order) > self.fingerprint_history_limit:
            self.written_event_ids.discard(self.event_id_order.popleft())

    def build_stream_name(self, source_type: str) -> str:
        return f"lecturelens:{self.session_id}:stream:{source_type}"

    def transcript_to_entry(self, event: TranscriptChunk) -> TimelineEntry:
        metadata = {
            "confidence": event.confidence, "language": event.language,
            "audio_source": event.audio_source, "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds, "raw_audio_path": event.raw_audio_path
        }
        return TimelineEntry(
            session_id=event.session_id, event_id=event.event_id, timestamp_epoch=event.timestamp_epoch,
            elapsed_seconds=self.session_clock.elapsed_for_epoch(event.timestamp_epoch),
            source_type="transcript", content=event.text, correlation_id=None,
            importance_flags=list(event.importance_flags), metadata=metadata
        )

    def slide_to_entry(self, event: SlideCaptured) -> TimelineEntry:
        metadata = {
            "slide_number": event.slide_number, "monitor_index": event.monitor_index,
            "image_file_path": event.image_file_path, "revisited_slide_number": event.revisited_slide_number,
            "figure_file_path": event.figure_file_path, "figure_caption": event.figure_caption,
            "figure_id": getattr(event, "figure_id", "")
        }
        return TimelineEntry(
            session_id=event.session_id, event_id=event.event_id, timestamp_epoch=event.timestamp_epoch,
            elapsed_seconds=self.session_clock.elapsed_for_epoch(event.timestamp_epoch),
            source_type="slide", content=event.extracted_text, correlation_id=None,
            importance_flags=[], metadata=metadata
        )

    def convert_message(self, source_type: str, message_data: dict[str, str]) -> TimelineEntry:
        if source_type == "transcript":
            return self.transcript_to_entry(TranscriptChunk.from_redis_dict(message_data))
        if source_type == "slide":
            return self.slide_to_entry(SlideCaptured.from_redis_dict(message_data))
        raise ValueError(f"unknown source type {source_type}")

    async def add_entry(
        self, entry: TimelineEntry, stream_name: Optional[str] = None,
        group_name: Optional[str] = None, message_id: Optional[str] = None
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
            fp = entry_fingerprint(entry)
            if self.is_fully_flushed(entry, fp) or (fp in self.file_fingerprints and self.flush_states.get(fp, "") != "file"):
                self.duplicates_skipped += 1
                await self.acknowledge(stream_name, group_name, message_id)
                return
            if pending_key is not None:
                self.pending_message_ids.add(pending_key)
            self.pending_buffer.append(BufferedEntry(entry, time.monotonic(), stream_name, group_name, message_id))

    def is_fully_flushed(self, entry: TimelineEntry, fingerprint: str) -> bool:
        if fingerprint in self.written_fingerprints:
            return True
        if self.flush_state_tracked:
            return self.flush_states.get(fingerprint, "") == "both" or bool(entry.event_id and entry.event_id in self.written_event_ids)
        return bool(entry.event_id and entry.event_id in self.written_event_ids)

    async def acknowledge(self, stream_name: Optional[str], group_name: Optional[str], message_id: Optional[str]) -> None:
        if stream_name and group_name and message_id:
            try:
                await self.redis_bus.client.xack(stream_name, group_name, message_id)
            except Exception as e:
                logger.warning("failed to ack message %s: %s", message_id, e)

    async def consume_stream(self, source_type: str) -> None:
        stream_name = self.build_stream_name(source_type)

        async def handle_message(message_data: dict[str, str], message_id: str) -> None:
            await self.add_entry(
                self.convert_message(source_type, message_data),
                stream_name=stream_name, group_name=self.consumer_group, message_id=message_id
            )

        await self.redis_bus.consume(
            stream_name=stream_name, group_name=self.consumer_group,
            consumer_name=f"{self.consumer_prefix}:{source_type}", handler=handle_message,
            batch_size=self.settings.redis_consumer_batch_size, block_ms=self.settings.redis_consumer_block_ms,
            should_stop=lambda: self.drain_requested, auto_ack=False,
            reclaim_idle_ms=int(max(self.settings.sync_merge_window_seconds + 60.0, 1.0) * 1000)
        )

    def find_nearest_anchor(self, timestamp_epoch: float, window_seconds: float) -> Optional[AnchorRecord]:
        best, best_dist = None, window_seconds + 1.0
        for r in self.recent_anchors:
            d = abs(r.timestamp_epoch - timestamp_epoch)
            if d <= window_seconds and d < best_dist:
                best, best_dist = r, d
        return best

    def find_active_anchor(self, timestamp_epoch: float, max_age_seconds: float) -> Optional[AnchorRecord]:
        best = None
        for r in self.recent_anchors:
            age = timestamp_epoch - r.timestamp_epoch
            if 0.0 <= age <= max_age_seconds:
                if best is None or r.timestamp_epoch > best.timestamp_epoch:
                    best = r
        return best

    def find_upcoming_anchor(self, timestamp_epoch: float, lead_seconds: float) -> Optional[AnchorRecord]:
        best = None
        for r in self.recent_anchors:
            lead = r.timestamp_epoch - timestamp_epoch
            if 0.0 < lead <= lead_seconds:
                if best is None or r.timestamp_epoch < best.timestamp_epoch:
                    best = r
        return best

    def select_anchor_for_speech(self, timestamp_epoch: float) -> Optional[AnchorRecord]:
        upcoming = self.find_upcoming_anchor(timestamp_epoch, self.settings.correlation_lead_seconds)
        if upcoming is not None:
            return upcoming
        if self.settings.correlation_active_seconds > 0.0:
            active = self.find_active_anchor(timestamp_epoch, self.settings.correlation_active_seconds)
            if active is not None:
                return active
        return self.find_nearest_anchor(timestamp_epoch, self.settings.correlation_window_seconds)

    @staticmethod
    def find_nearest_entry_of_other_type(
        entry: TimelineEntry, ordered_entries: list[TimelineEntry], window_seconds: float
    ) -> Optional[TimelineEntry]:
        best, best_dist = None, window_seconds + 1.0
        for c in ordered_entries:
            if c is not entry and c.source_type != entry.source_type:
                d = abs(c.timestamp_epoch - entry.timestamp_epoch)
                if d <= window_seconds and d < best_dist:
                    best, best_dist = c, d
        return best

    def prune_anchor_history(self, newest_timestamp: float, window_seconds: float) -> None:
        horizon = max(60.0, window_seconds * 6.0, self.settings.correlation_active_seconds)
        self.recent_anchors = [r for r in self.recent_anchors if newest_timestamp - r.timestamp_epoch <= horizon]

    def assign_correlation_ids(self, ordered_entries: list[TimelineEntry]) -> None:
        win = self.settings.correlation_window_seconds
        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES:
                continue
            slide_num = int(entry.metadata.get("slide_number", 0) or 0)
            if slide_num > 0 and slide_num in self.slide_correlations:
                entry.correlation_id = self.slide_correlations[slide_num]
            elif entry.correlation_id is not None and slide_num > 0:
                self.slide_correlations[slide_num] = entry.correlation_id
            if entry.correlation_id is not None:
                continue
            neighbour = self.find_nearest_entry_of_other_type(entry, ordered_entries, win)
            if neighbour is not None:
                if neighbour.correlation_id is None:
                    neighbour.correlation_id = str(uuid.uuid4())
                entry.correlation_id = neighbour.correlation_id
                if slide_num > 0:
                    self.slide_correlations[slide_num] = entry.correlation_id

        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES:
                continue
            if entry.correlation_id is None:
                entry.correlation_id = str(uuid.uuid4())
            slide_num = int(entry.metadata.get("slide_number", 0) or 0)
            if slide_num > 0:
                self.slide_correlations[slide_num] = entry.correlation_id
            if not any(r.correlation_id == entry.correlation_id and r.source_type == entry.source_type and r.timestamp_epoch == entry.timestamp_epoch for r in self.recent_anchors):
                self.recent_anchors.append(AnchorRecord(entry.timestamp_epoch, entry.correlation_id, entry.source_type))

        for entry in ordered_entries:
            if entry.source_type not in ANCHOR_SOURCE_TYPES and entry.correlation_id is None:
                anchor = self.select_anchor_for_speech(entry.timestamp_epoch)
                if anchor is not None:
                    entry.correlation_id = anchor.correlation_id

        if ordered_entries:
            self.prune_anchor_history(ordered_entries[-1].timestamp_epoch, win)

    def select_ready_entries(self, now_monotonic: float, force: bool) -> list[BufferedEntry]:
        merge_win = self.settings.sync_merge_window_seconds
        ready = [b for b in self.pending_buffer if force or (now_monotonic - b.arrival_monotonic) >= merge_win]
        ready.sort(key=lambda b: b.entry.timestamp_epoch)
        return ready

    def remove_flushed_entries(self, flushed: list[BufferedEntry]) -> None:
        flushed_ids = {id(b) for b in flushed}
        self.pending_buffer = [b for b in self.pending_buffer if id(b) not in flushed_ids]

    def render_lines(self, entries: list[TimelineEntry]) -> str:
        return "\n".join(json.dumps(e.model_dump(), ensure_ascii=False) for e in entries) + "\n"

    async def flush_ready_entries(self, force: bool = False) -> int:
        async with self.buffer_lock:
            ready = self.select_ready_entries(time.monotonic(), force)
            if not ready:
                return 0
            ordered = [b.entry for b in ready]
            self.assign_correlation_ids(ordered)
            fingerprints = [entry_fingerprint(b.entry) for b in ready]
            needs_file = [b for b, fp in zip(ready, fingerprints) if self.flush_states.get(fp, "") not in ("file", "both")]
            needs_redis = [b for b, fp in zip(ready, fingerprints) if self.flush_states.get(fp, "") not in ("redis", "both")]

            timeline_stream = f"lecturelens:{self.session_id}:stream:timeline"
            if needs_file:
                for b in needs_file:
                    fp = entry_fingerprint(b.entry)
                    if self.flush_states.get(fp) not in ("file", "both", "redis"):
                        self.flush_states[fp] = "file_pending"
                self.save_flush_states()
                self.trigger_fault("crash_before_file_write")
                try:
                    await run_in_thread_pool(append_text_durably, self.timeline_path, self.render_lines([b.entry for b in needs_file]))
                except OSError as error:
                    logger.error("timeline write failed: %s", error)
                    return 0
                self.trigger_fault("crash_before_state_save")
                for b in needs_file:
                    fp = entry_fingerprint(b.entry)
                    self.flush_states[fp] = "both" if self.flush_states.get(fp, "") == "redis" else "file"
                self.save_flush_states()
                self.trigger_fault("crash_after_file_write")

            if needs_redis:
                for b in needs_redis:
                    try:
                        await self.redis_bus.publish_resilient(timeline_stream, b.entry)
                    except Exception as pub_error:
                        logger.error("publishing timeline entry failed: %s", pub_error)
                        self.save_flush_states()
                        return 0
                    fp = entry_fingerprint(b.entry)
                    self.flush_states[fp] = "both" if self.flush_states.get(fp, "") == "file" else "redis"
                self.save_flush_states()

            flushed, finished_fps = [], []
            for b, fp in zip(ready, fingerprints):
                if self.flush_states.get(fp) == "both":
                    flushed.append(b)
                    finished_fps.append(fp)

            if finished_fps:
                finished_set = set(finished_fps)
                self.flush_states = {fp: st for fp, st in self.flush_states.items() if fp not in finished_set}
                self.write_flush_states(self.flush_states)
                self.trigger_fault("crash_after_state_pop")

            for b, fp in zip(flushed, finished_fps):
                self.remember_fingerprint(fp)
                if b.entry.event_id:
                    self.remember_event_id(b.entry.event_id)
                if b.message_id is not None:
                    self.pending_message_ids.discard((b.stream_name or "", b.message_id))
                await self.acknowledge(b.stream_name, b.group_name, b.message_id)
            self.remove_flushed_entries(flushed)
            self.entries_written += len(flushed)
            self.control.set_stat("timeline_entries_written", self.entries_written)
            self.control.set_stat("buffered_entries", len(self.pending_buffer))
            return len(ordered)

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
        consumer_tasks = [asyncio.create_task(self.consume_stream(s)) for s in ("transcript", "slide")]
        flush_task = asyncio.create_task(self.flush_loop())
        stop_watch = asyncio.create_task(self.watch_stop_flag())
        try:
            await asyncio.gather(*consumer_tasks)
        finally:
            self.is_running = False
            flush_task.cancel()
            stop_watch.cancel()
            await asyncio.gather(flush_task, stop_watch, return_exceptions=True)
            for t in consumer_tasks:
                if not t.done():
                    t.cancel()
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
