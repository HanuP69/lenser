import json
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class TranscriptChunk(BaseModel):
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_epoch: float
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = Field(default="en")
    audio_source: str = Field(default="mic")
    speaker_id: Optional[str] = Field(default=None)
    importance_flags: list[str] = Field(default_factory=list)
    raw_audio_path: Optional[str] = Field(default=None)

    def to_redis_dict(self) -> dict[str, str]:
        data = self.model_dump()
        result = {}
        for key in data:
            value = data[key]
            if value is None:
                result[key] = ""
            elif isinstance(value, list):
                joined = ",".join(str(item) for item in value)
                result[key] = joined
            elif isinstance(value, bool):
                result[key] = "1" if value else "0"
            else:
                result[key] = str(value)
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "TranscriptChunk":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["event_id"] = data.get("event_id", "") or str(uuid.uuid4())
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["start_seconds"] = float(data["start_seconds"])
        parsed["end_seconds"] = float(data["end_seconds"])
        parsed["text"] = data["text"]
        parsed["confidence"] = float(data.get("confidence", "1.0"))
        parsed["language"] = data.get("language", "") or "en"
        parsed["audio_source"] = data.get("audio_source", "") or "mic"

        speaker_id_value = data.get("speaker_id", "")
        parsed["speaker_id"] = None if speaker_id_value == "" else speaker_id_value

        flags_value = data.get("importance_flags", "")
        if flags_value == "":
            parsed["importance_flags"] = []
        else:
            parsed["importance_flags"] = [item for item in flags_value.split(",") if item]

        raw_path_value = data.get("raw_audio_path", "")
        parsed["raw_audio_path"] = None if raw_path_value == "" else raw_path_value

        return cls(**parsed)


class SlideCaptured(BaseModel):
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_epoch: float
    slide_number: int
    monitor_index: int = Field(default=0)
    image_hash: str = Field(default="")
    image_file_path: str
    extracted_text: str
    revisited_slide_number: int = Field(default=0)
    ocr_latency_ms: float = Field(default=0.0)

    def to_redis_dict(self) -> dict[str, str]:
        data = self.model_dump()
        result = {}
        for key in data:
            value = data[key]
            if value is None:
                result[key] = ""
            else:
                result[key] = str(value)
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "SlideCaptured":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["event_id"] = data.get("event_id", "") or str(uuid.uuid4())
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["slide_number"] = int(data["slide_number"])
        parsed["monitor_index"] = int(data.get("monitor_index", "0") or "0")
        parsed["image_hash"] = data.get("image_hash", "")
        parsed["image_file_path"] = data["image_file_path"]
        parsed["extracted_text"] = data["extracted_text"]
        parsed["revisited_slide_number"] = int(data.get("revisited_slide_number", "0") or "0")
        parsed["ocr_latency_ms"] = float(data.get("ocr_latency_ms", "0.0") or "0.0")
        return cls(**parsed)


class BoardCaptured(BaseModel):
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_epoch: float
    image_file_path: str
    raw_ocr_text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_used: str = Field(default="unknown")

    def to_redis_dict(self) -> dict[str, str]:
        data = self.model_dump()
        result = {}
        for key in data:
            value = data[key]
            if value is None:
                result[key] = ""
            else:
                result[key] = str(value)
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "BoardCaptured":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["event_id"] = data.get("event_id", "") or str(uuid.uuid4())
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["image_file_path"] = data["image_file_path"]
        parsed["raw_ocr_text"] = data["raw_ocr_text"]
        parsed["confidence"] = float(data.get("confidence", "0.0") or "0.0")
        parsed["provider_used"] = data.get("provider_used", "unknown")
        return cls(**parsed)


class GeneralNoteSegment(BaseModel):
    session_id: str
    lecture_id: Optional[str] = Field(default=None)
    sequence_number: int
    timestamp_epoch: float
    start_time: str
    end_time: str
    heading: str
    content: str
    key_points: list[str] = Field(default_factory=list)
    concept_tags: list[str] = Field(default_factory=list)
    slide_references: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _parse_serialized_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        for field in ("key_points", "concept_tags", "slide_references"):
            val = copied.get(field)
            if isinstance(val, str):
                trimmed = val.strip()
                if trimmed.startswith("[") and trimmed.endswith("]"):
                    try:
                        copied[field] = json.loads(trimmed)
                    except Exception:
                        pass
                elif len(trimmed) == 0:
                    copied[field] = []
        return copied

    def to_redis_dict(self) -> dict[str, str]:
        result = {
            "session_id": self.session_id,
            "lecture_id": self.lecture_id or "",
            "sequence_number": str(self.sequence_number),
            "timestamp_epoch": str(self.timestamp_epoch),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "heading": self.heading,
            "content": self.content,
            "key_points": json.dumps(self.key_points),
            "concept_tags": json.dumps(self.concept_tags),
            "slide_references": json.dumps(self.slide_references)
        }
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "GeneralNoteSegment":
        key_points_raw = data.get("key_points", "[]")
        concept_tags_raw = data.get("concept_tags", "[]")
        slides_raw = data.get("slide_references", "[]")
        return cls(
            session_id=data["session_id"],
            lecture_id=data.get("lecture_id") or None,
            sequence_number=int(data["sequence_number"]),
            timestamp_epoch=float(data["timestamp_epoch"]),
            start_time=data["start_time"],
            end_time=data["end_time"],
            heading=data["heading"],
            content=data["content"],
            key_points=json.loads(key_points_raw) if key_points_raw else [],
            concept_tags=json.loads(concept_tags_raw) if concept_tags_raw else [],
            slide_references=json.loads(slides_raw) if slides_raw else []
        )


class FlagMoment(BaseModel):
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_epoch: float
    elapsed_seconds: float
    flag_type: str
    cue_text: str
    importance: float = Field(default=1.0)

    def to_redis_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "timestamp_epoch": str(self.timestamp_epoch),
            "elapsed_seconds": str(self.elapsed_seconds),
            "flag_type": self.flag_type,
            "cue_text": self.cue_text,
            "importance": str(self.importance)
        }

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "FlagMoment":
        return cls(
            session_id=data["session_id"],
            event_id=data.get("event_id", "") or str(uuid.uuid4()),
            timestamp_epoch=float(data["timestamp_epoch"]),
            elapsed_seconds=float(data.get("elapsed_seconds", "0.0") or "0.0"),
            flag_type=data["flag_type"],
            cue_text=data["cue_text"],
            importance=float(data.get("importance", "1.0") or "1.0")
        )


class AnswerChunk(BaseModel):
    session_id: str
    client_id: str
    turn_id: str
    chunk_index: int
    delta_text: str
    is_final: bool = Field(default=False)
    cited_timestamps: list[float] = Field(default_factory=list)

    def to_redis_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "turn_id": self.turn_id,
            "chunk_index": str(self.chunk_index),
            "delta_text": self.delta_text,
            "is_final": "1" if self.is_final else "0",
            "cited_timestamps": json.dumps(self.cited_timestamps)
        }

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "AnswerChunk":
        return cls(
            session_id=data["session_id"],
            client_id=data["client_id"],
            turn_id=data["turn_id"],
            chunk_index=int(data["chunk_index"]),
            delta_text=data["delta_text"],
            is_final=data.get("is_final") in ("1", "true", "True"),
            cited_timestamps=json.loads(data.get("cited_timestamps", "[]"))
        )


class AvatarState(BaseModel):
    session_id: str
    timestamp_epoch: float
    state: str
    metadata: dict = Field(default_factory=dict)

    def to_redis_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "timestamp_epoch": str(self.timestamp_epoch),
            "state": self.state,
            "metadata": json.dumps(self.metadata)
        }

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "AvatarState":
        return cls(
            session_id=data["session_id"],
            timestamp_epoch=float(data["timestamp_epoch"]),
            state=data["state"],
            metadata=json.loads(data.get("metadata", "{}"))
        )


class TimelineEntry(BaseModel):
    session_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_epoch: float
    source_type: str
    content: str
    correlation_id: Optional[str] = Field(default=None)
    importance_flags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_jsonl_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_jsonl_dict(cls, data: dict) -> "TimelineEntry":
        if "event_id" not in data:
            data["event_id"] = str(uuid.uuid4())
        return cls(**data)

    def to_redis_dict(self) -> dict[str, str]:
        result = {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "timestamp_epoch": str(self.timestamp_epoch),
            "source_type": self.source_type,
            "content": self.content,
            "correlation_id": "" if self.correlation_id is None else self.correlation_id,
            "importance_flags": ",".join(self.importance_flags),
            "metadata": json.dumps(self.metadata)
        }
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "TimelineEntry":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["event_id"] = data.get("event_id", "") or str(uuid.uuid4())
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["source_type"] = data["source_type"]
        parsed["content"] = data["content"]

        correlation_value = data.get("correlation_id", "")
        parsed["correlation_id"] = None if correlation_value == "" else correlation_value

        flags_value = data.get("importance_flags", "")
        if flags_value == "":
            parsed["importance_flags"] = []
        else:
            parsed["importance_flags"] = [item for item in flags_value.split(",") if item]

        metadata_value = data.get("metadata", "{}")
        parsed["metadata"] = json.loads(metadata_value)

        return cls(**parsed)
