import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import SlideCaptured, TimelineEntry, TranscriptChunk

logger = logging.getLogger(__name__)


class ReplayEngine:
    def __init__(
        self,
        timeline_path: Path,
        redis_bus: Optional[RedisBus] = None,
        speed: float = 1.0
    ) -> None:
        self.timeline_path = timeline_path
        self.redis_bus = redis_bus
        self.speed = speed

    def load_entries(self) -> list[TimelineEntry]:
        if not self.timeline_path.exists():
            return []
        entries: list[TimelineEntry] = []
        with open(self.timeline_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if hasattr(TimelineEntry, "from_jsonl_dict"):
                        entry = TimelineEntry.from_jsonl_dict(payload)
                    else:
                        entry = TimelineEntry(**payload)
                    entries.append(entry)
                except Exception as parse_error:
                    logger.warning("skipping unparseable replay line: %s", parse_error)
        entries.sort(key=lambda item: item.timestamp_epoch)
        return entries

    async def replay(
        self,
        target_session_id: Optional[str] = None,
        on_entry: Optional[Callable[[TimelineEntry], Any]] = None,
        publish_to_timeline_stream: bool = True,
        publish_to_source_streams: bool = False
    ) -> int:
        entries = self.load_entries()
        if not entries:
            return 0

        first_timestamp = entries[0].timestamp_epoch
        last_timestamp = first_timestamp
        replayed_count = 0

        for entry in entries:
            current_timestamp = entry.timestamp_epoch
            delta = max(0.0, current_timestamp - last_timestamp)
            if self.speed > 0.0 and delta > 0.0:
                delay = delta / self.speed
                if delay > 0.001:
                    await asyncio.sleep(min(delay, 5.0))
            last_timestamp = current_timestamp

            session_id = target_session_id or entry.session_id
            replayed_entry = entry.model_copy(update={"session_id": session_id})

            if on_entry is not None:
                result = on_entry(replayed_entry)
                if asyncio.iscoroutine(result):
                    await result

            if self.redis_bus is not None:
                if publish_to_timeline_stream:
                    stream_name = f"lecturelens:{session_id}:stream:timeline"
                    await self.redis_bus.publish_resilient(stream_name, replayed_entry)
                if publish_to_source_streams:
                    if replayed_entry.source_type == "transcript":
                        meta = replayed_entry.metadata or {}
                        chunk = TranscriptChunk(
                            session_id=session_id,
                            chunk_index=int(meta.get("chunk_index", 0) or 0),
                            timestamp_epoch=replayed_entry.timestamp_epoch,
                            start_seconds=float(meta.get("start_seconds", replayed_entry.elapsed_seconds or 0.0) or 0.0),
                            end_seconds=float(meta.get("end_seconds", replayed_entry.elapsed_seconds or 0.0) or 0.0),
                            text=replayed_entry.content,
                            language=str(meta.get("language", "en") or "en"),
                            confidence=float(meta.get("confidence", 1.0) or 1.0),
                            importance_flags=list(replayed_entry.importance_flags or []),
                            audio_source=str(meta.get("audio_source", "mic") or "mic"),
                            raw_audio_path=str(meta.get("raw_audio_path", "") or "")
                        )
                        source_stream = f"lecturelens:{session_id}:stream:transcript"
                        await self.redis_bus.publish_resilient(source_stream, chunk)
                    elif replayed_entry.source_type == "slide":
                        meta = replayed_entry.metadata or {}
                        slide = SlideCaptured(
                            session_id=session_id,
                            slide_number=int(meta.get("slide_number", 1) or 1),
                            timestamp_epoch=replayed_entry.timestamp_epoch,
                            image_file_path=str(meta.get("image_file_path", "") or ""),
                            extracted_text=replayed_entry.content,
                            monitor_index=int(meta.get("monitor_index", 0) or 0),
                            revisited_slide_number=int(meta.get("revisited_slide_number", 0) or 0),
                            ocr_latency_ms=float(meta.get("ocr_latency_ms", 0.0) or 0.0)
                        )
                        source_stream = f"lecturelens:{session_id}:stream:slide"
                        await self.redis_bus.publish_resilient(source_stream, slide)

            replayed_count += 1

        return replayed_count
