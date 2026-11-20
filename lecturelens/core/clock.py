import json
import time
from pathlib import Path

from lecturelens.core.fileio import atomic_write_text

DRIFT_TOLERANCE_SECONDS: float = 5.0


class SessionClock:
    def __init__(self, elapsed_offset_seconds: float = 0.0) -> None:
        self.reference_wall_time: float = time.time()
        self.reference_monotonic_time: float = time.monotonic()
        self.elapsed_offset_seconds: float = elapsed_offset_seconds
        # The anchor pair drives the timestamps. It starts on the session
        # references, and when the wall clock steps (an NTP correction, or
        # a laptop that slept) it is moved to the wall clock. The
        # references themselves stay where the session started, so the
        # elapsed lecture time is not reset by a clock step. Without the
        # separate anchor the estimate stayed offset from the wall clock
        # forever after the step and every timestamp came from whatever
        # side of the tolerance the slew happened to be in, which made
        # the timeline jumpy.
        self._anchor_wall_time: float = self.reference_wall_time
        self._anchor_monotonic_time: float = self.reference_monotonic_time

    def get_timestamp(self) -> float:
        monotonic_elapsed = time.monotonic() - self._anchor_monotonic_time
        monotonic_estimate = self._anchor_wall_time + monotonic_elapsed
        wall_now = time.time()
        if abs(wall_now - monotonic_estimate) > DRIFT_TOLERANCE_SECONDS:
            # The wall clock moved against monotonic time. One timestamp
            # jumps with it, and from here on the monotonic estimate
            # follows the new anchor instead of disagreeing with the wall
            # clock for the rest of the session.
            self._anchor_wall_time = wall_now
            self._anchor_monotonic_time = time.monotonic()
            return wall_now
        return monotonic_estimate

    def get_run_elapsed_seconds(self) -> float:
        # Measured against the session references, not the anchor: the
        # anchor only keeps the timestamps smooth, while the elapsed
        # lecture time and the epoch_for_elapsed mapping have to keep the
        # same base the session started on.
        elapsed = self.get_timestamp() - self.reference_wall_time
        if elapsed < 0.0:
            return 0.0
        return elapsed

    def get_elapsed_seconds(self) -> float:
        return self.get_run_elapsed_seconds() + self.elapsed_offset_seconds

    def epoch_for_elapsed(self, elapsed_seconds: float) -> float:
        return self.reference_wall_time + (elapsed_seconds - self.elapsed_offset_seconds)

    def elapsed_for_epoch(self, epoch_seconds: float) -> float:
        elapsed = (epoch_seconds - self.reference_wall_time) + self.elapsed_offset_seconds
        if elapsed < self.elapsed_offset_seconds:
            return self.elapsed_offset_seconds
        return elapsed

    def format_elapsed(self, elapsed_seconds: float) -> str:
        """Format an elapsed duration as HH:MM:SS."""
        total = int(max(0.0, elapsed_seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def format_epoch_as_elapsed(self, epoch_seconds: float) -> str:
        """Format an absolute timestamp as the elapsed lecture time.

        The notes used the wall clock time of the moment, so a lecture that
        started at 14:30 produced headings like "14:42:03". The elapsed
        lecture time is what a reader needs, and it is what the question
        answering citations already use.
        """
        return self.format_elapsed(self.elapsed_for_epoch(epoch_seconds))

    def to_dict(self) -> dict:
        data = {}
        data["reference_wall_time"] = self.reference_wall_time
        data["reference_monotonic_time"] = self.reference_monotonic_time
        data["elapsed_offset_seconds"] = self.elapsed_offset_seconds
        data["accumulated_seconds"] = self.get_elapsed_seconds()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SessionClock":
        clock = cls.__new__(cls)
        clock.reference_wall_time = float(data["reference_wall_time"])
        clock.reference_monotonic_time = float(data["reference_monotonic_time"])
        clock.elapsed_offset_seconds = float(data.get("elapsed_offset_seconds", 0.0))
        monotonic_now = time.monotonic()
        if monotonic_now < clock.reference_monotonic_time:
            wall_elapsed = time.time() - clock.reference_wall_time
            if wall_elapsed < 0.0:
                wall_elapsed = 0.0
            clock.reference_monotonic_time = monotonic_now - wall_elapsed
        # The anchor starts on the references that were restored. It was
        # not saved, because a step of the wall clock between the runs can
        # be watched here as well as anywhere.
        clock._anchor_wall_time = clock.reference_wall_time
        clock._anchor_monotonic_time = clock.reference_monotonic_time
        return clock

    def save_to_file(self, file_path: Path) -> None:
        json_string = json.dumps(self.to_dict())
        atomic_write_text(file_path, json_string)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "SessionClock":
        with open(file_path, "r", encoding="utf-8") as file_handle:
            json_string = file_handle.read()
        data = json.loads(json_string)
        return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, file_path: Path) -> "SessionClock":
        if file_path.exists():
            try:
                return cls.load_from_file(file_path)
            except (ValueError, TypeError, KeyError, OSError):
                pass
        clock = cls()
        clock.save_to_file(file_path)
        return clock
