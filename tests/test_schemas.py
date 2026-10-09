import unittest
from lecturelens.schemas.events import (
    TranscriptChunk,
    SlideCaptured,
    BoardCaptured,
    TimelineEntry
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

    def test_board_captured_serialization(self) -> None:
        board = BoardCaptured(
            session_id="test_session_3",
            timestamp_epoch=1700000020.0,
            image_file_path="output/raw/board/board_0001.jpg",
            raw_ocr_text="T(n) = 2T(n/2) + O(n)",
            confidence=0.88,
            provider_used="gemma4"
        )
        redis_dict = board.to_redis_dict()
        deserialized = BoardCaptured.from_redis_dict(redis_dict)
        self.assertEqual(deserialized.raw_ocr_text, "T(n) = 2T(n/2) + O(n)")
        self.assertEqual(deserialized.provider_used, "gemma4")

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


if __name__ == "__main__":
    unittest.main()
