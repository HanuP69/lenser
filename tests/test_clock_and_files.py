import json
import tempfile
import time
import unittest
from pathlib import Path

from lecturelens.core.clock import SessionClock
from lecturelens.core.fileio import atomic_write_text, next_file_index
from lecturelens.core.session_meta import begin_run, finish_run, get_previous_duration_seconds
from lecturelens.core.validation import is_valid_session_id, sanitize_file_stem


class TestSessionClock(unittest.TestCase):
    def test_timestamps_increase(self) -> None:
        clock = SessionClock()
        first = clock.get_timestamp()
        time.sleep(0.02)
        self.assertGreater(clock.get_timestamp(), first)

    def test_offset_and_epoch_conversion(self) -> None:
        clock = SessionClock(elapsed_offset_seconds=100.0)
        self.assertGreaterEqual(clock.get_elapsed_seconds(), 100.0)
        self.assertAlmostEqual(
            clock.epoch_for_elapsed(105.0), clock.reference_wall_time + 5.0, places=6
        )

    def test_round_trip_through_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clock.json"
            original = SessionClock(elapsed_offset_seconds=7.0)
            original.save_to_file(path)
            restored = SessionClock.load_from_file(path)
            self.assertAlmostEqual(restored.reference_wall_time, original.reference_wall_time)
            self.assertEqual(restored.elapsed_offset_seconds, 7.0)

    def test_reboot_reanchors_monotonic_reference(self) -> None:
        clock = SessionClock()
        data = clock.to_dict()
        data["reference_monotonic_time"] = time.monotonic() + 100000.0
        data["reference_wall_time"] = time.time() - 30.0
        restored = SessionClock.from_dict(data)
        self.assertLess(restored.reference_monotonic_time, time.monotonic())
        self.assertAlmostEqual(restored.get_run_elapsed_seconds(), 30.0, delta=2.0)

    def test_sleep_gap_falls_back_to_wall_clock(self) -> None:
        clock = SessionClock()
        clock.reference_monotonic_time = time.monotonic() - 1.0
        clock.reference_wall_time = time.time() - 600.0
        self.assertAlmostEqual(clock.get_timestamp(), time.time(), delta=1.0)

    def test_load_or_create_handles_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clock.json"
            path.write_text("{not json", encoding="utf-8")
            clock = SessionClock.load_or_create(path)
            self.assertIsNotNone(clock)
            self.assertIn("reference_wall_time", json.loads(path.read_text(encoding="utf-8")))


class TestFileHelpers(unittest.TestCase):
    def test_next_file_index_continues_after_highest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self.assertEqual(next_file_index(directory, "chunk_", ".wav"), 1)
            (directory / "chunk_00001.wav").write_bytes(b"a")
            (directory / "chunk_00007.wav").write_bytes(b"b")
            (directory / "other.txt").write_bytes(b"c")
            self.assertEqual(next_file_index(directory, "chunk_", ".wav"), 8)

    def test_atomic_write_replaces_content_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            atomic_write_text(path, "one")
            atomic_write_text(path, "two")
            self.assertEqual(path.read_text(encoding="utf-8"), "two")
            self.assertEqual(len(list(Path(temp_dir).iterdir())), 1)


class TestValidationAndMeta(unittest.TestCase):
    def test_session_id_validation(self) -> None:
        self.assertTrue(is_valid_session_id("2026-10-07_algorithms"))
        self.assertFalse(is_valid_session_id("../etc"))
        self.assertFalse(is_valid_session_id("a/b"))
        self.assertFalse(is_valid_session_id(""))
        self.assertFalse(is_valid_session_id("a..b"))

    def test_sanitize_file_stem(self) -> None:
        self.assertEqual(sanitize_file_stem("CS 101/Intro"), "CS_101_Intro")
        self.assertEqual(sanitize_file_stem("///"), "default")

    def test_session_meta_runs_accumulate_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            self.assertEqual(get_previous_duration_seconds(session_dir), 0.0)
            begin_run(session_dir, "s1", "algo", {"mode": "local"}, ["audio_agent"])
            time.sleep(0.05)
            finish_run(session_dir)
            first_total = get_previous_duration_seconds(session_dir)
            self.assertGreater(first_total, 0.0)
            begin_run(session_dir, "s1", "algo", {"mode": "local"}, ["audio_agent"])
            finish_run(session_dir)
            self.assertGreaterEqual(get_previous_duration_seconds(session_dir), first_total)


if __name__ == "__main__":
    unittest.main()
