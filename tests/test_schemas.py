import unittest
from lecturelens.schemas.events import (
    TranscriptChunk,
    SlideCaptured,
    TimelineEntry,
    GeneralNoteSegment
)


class TestSchemas(unittest.TestCase):
    def test_transcript_chunk_serialization(self) -> None:
        chunk = TranscriptChunk(
            session_id="test_session_1",
            timestamp_epoch=1700000000.5,
            start_seconds=0.0,
            end_seconds=3.0,
            text="Welcome to algorithms lecture.",
            confidence=0.95,
            language="en",
            audio_source="mic",
            speaker_id="speaker_1",
            importance_flags=["definition", "exam_relevant"],
            raw_audio_path="output/raw/audio/chunk_001.wav"
        )
        redis_dict = chunk.to_redis_dict()
        self.assertEqual(redis_dict["session_id"], "test_session_1")
        self.assertEqual(redis_dict["confidence"], "0.95")
        self.assertEqual(redis_dict["importance_flags"], "definition,exam_relevant")

        deserialized = TranscriptChunk.from_redis_dict(redis_dict)
        self.assertEqual(deserialized.session_id, chunk.session_id)
        self.assertEqual(deserialized.timestamp_epoch, chunk.timestamp_epoch)
        self.assertEqual(deserialized.text, chunk.text)
        self.assertEqual(deserialized.confidence, chunk.confidence)
        self.assertEqual(deserialized.importance_flags, ["definition", "exam_relevant"])
        self.assertEqual(deserialized.raw_audio_path, chunk.raw_audio_path)

    def test_slide_captured_serialization(self) -> None:
        slide = SlideCaptured(
            session_id="test_session_2",
            timestamp_epoch=1700000010.0,
            slide_number=1,
            monitor_index=0,
            image_file_path="output/raw/slides/slide_0001.png",
            extracted_text="Binary Search O(log n)"
        )
        redis_dict = slide.to_redis_dict()
        deserialized = SlideCaptured.from_redis_dict(redis_dict)
        self.assertEqual(deserialized.slide_number, 1)
        self.assertEqual(deserialized.extracted_text, "Binary Search O(log n)")

    def test_timeline_entry_serialization(self) -> None:
        entry = TimelineEntry(
            session_id="test_session_4",
            timestamp_epoch=1700000030.0,
            source_type="transcript",
            content="Master theorem explanation",
            correlation_id="corr-123",
            importance_flags=["exam_relevant"],
            metadata={"custom_key": 42}
        )
        jsonl_dict = entry.to_jsonl_dict()
        from_jsonl = TimelineEntry.from_jsonl_dict(jsonl_dict)
        self.assertEqual(from_jsonl.correlation_id, "corr-123")
        self.assertEqual(from_jsonl.metadata["custom_key"], 42)

        redis_dict = entry.to_redis_dict()
        from_redis = TimelineEntry.from_redis_dict(redis_dict)
        self.assertEqual(from_redis.correlation_id, "corr-123")
        self.assertEqual(from_redis.metadata["custom_key"], 42)

    def test_slide_revisit_field_round_trip(self) -> None:
        slide = SlideCaptured(
            session_id="s",
            timestamp_epoch=1.0,
            slide_number=4,
            monitor_index=1,
            image_file_path="x.png",
            extracted_text="text",
            revisited_slide_number=2
        )
        restored = SlideCaptured.from_redis_dict(slide.to_redis_dict())
        self.assertEqual(restored.revisited_slide_number, 2)
        legacy = slide.to_redis_dict()
        del legacy["revisited_slide_number"]
        self.assertEqual(SlideCaptured.from_redis_dict(legacy).revisited_slide_number, 0)

    def test_general_note_segment_handles_serialized_json_strings(self) -> None:
        data = {
            "session_id": "test_sess_note",
            "sequence_number": 1,
            "timestamp_epoch": 1700000000.0,
            "start_time": "00:00:00",
            "end_time": "00:00:10",
            "heading": "Intro",
            "content": "Intro content",
            "key_points": '["Point A", "Point B"]',
            "concept_tags": '["tag1", "tag2"]',
            "slide_references": "[1, 2]"
        }
        segment = GeneralNoteSegment(**data)
        self.assertEqual(segment.key_points, ["Point A", "Point B"])
        self.assertEqual(segment.concept_tags, ["tag1", "tag2"])
        self.assertEqual(segment.slide_references, [1, 2])


if __name__ == "__main__":
    unittest.main()
