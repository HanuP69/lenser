import json
from typing import Optional

from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    session_id: str
    timestamp_epoch: float
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
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
                joined = ",".join(value)
                result[key] = joined
            elif isinstance(value, bool):
                if value:
                    result[key] = "1"
                else:
                    result[key] = "0"
            else:
                result[key] = str(value)
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "TranscriptChunk":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["start_seconds"] = float(data["start_seconds"])
        parsed["end_seconds"] = float(data["end_seconds"])
        parsed["text"] = data["text"]
        parsed["confidence"] = float(data["confidence"])
        parsed["language"] = data.get("language", "") or "en"
        parsed["audio_source"] = data.get("audio_source", "") or "mic"

        speaker_id_value = data.get("speaker_id", "")
        if speaker_id_value == "":
            parsed["speaker_id"] = None
        else:
            parsed["speaker_id"] = speaker_id_value

        flags_value = data.get("importance_flags", "")
        if flags_value == "":
            parsed["importance_flags"] = []
        else:
            parsed["importance_flags"] = flags_value.split(",")

        raw_path_value = data.get("raw_audio_path", "")
        if raw_path_value == "":
            parsed["raw_audio_path"] = None
        else:
            parsed["raw_audio_path"] = raw_path_value

        return cls(**parsed)


class SlideCaptured(BaseModel):
    session_id: str
    timestamp_epoch: float
    slide_number: int
    monitor_index: int = Field(default=0)
    image_file_path: str
    extracted_text: str
    revisited_slide_number: int = Field(default=0)

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
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["slide_number"] = int(data["slide_number"])
        parsed["monitor_index"] = int(data.get("monitor_index", "0"))
        parsed["image_file_path"] = data["image_file_path"]
        parsed["extracted_text"] = data["extracted_text"]
        parsed["revisited_slide_number"] = int(data.get("revisited_slide_number", "0") or "0")
        return cls(**parsed)


class BoardCaptured(BaseModel):
    session_id: str
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
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["image_file_path"] = data["image_file_path"]
        parsed["raw_ocr_text"] = data["raw_ocr_text"]
        parsed["confidence"] = float(data.get("confidence", "0.0"))
        parsed["provider_used"] = data.get("provider_used", "unknown")
        return cls(**parsed)


class TimelineEntry(BaseModel):
    session_id: str
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
        return cls(**data)

    def to_redis_dict(self) -> dict[str, str]:
        result = {}
        result["session_id"] = self.session_id
        result["timestamp_epoch"] = str(self.timestamp_epoch)
        result["source_type"] = self.source_type
        result["content"] = self.content

        if self.correlation_id is None:
            result["correlation_id"] = ""
        else:
            result["correlation_id"] = self.correlation_id

        result["importance_flags"] = ",".join(self.importance_flags)
        result["metadata"] = json.dumps(self.metadata)
        return result

    @classmethod
    def from_redis_dict(cls, data: dict[str, str]) -> "TimelineEntry":
        parsed = {}
        parsed["session_id"] = data["session_id"]
        parsed["timestamp_epoch"] = float(data["timestamp_epoch"])
        parsed["source_type"] = data["source_type"]
        parsed["content"] = data["content"]

        correlation_value = data.get("correlation_id", "")
        if correlation_value == "":
            parsed["correlation_id"] = None
        else:
            parsed["correlation_id"] = correlation_value

        flags_value = data.get("importance_flags", "")
        if flags_value == "":
            parsed["importance_flags"] = []
        else:
            parsed["importance_flags"] = flags_value.split(",")

        metadata_value = data.get("metadata", "{}")
        parsed["metadata"] = json.loads(metadata_value)

        return cls(**parsed)
