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

    def get_timestamp(self) -> float:
        monotonic_elapsed = time.monotonic() - self.reference_monotonic_time
        monotonic_estimate = self.reference_wall_time + monotonic_elapsed
        wall_now = time.time()
        if abs(wall_now - monotonic_estimate) > DRIFT_TOLERANCE_SECONDS:
            return wall_now
        return monotonic_estimate

    def get_run_elapsed_seconds(self) -> float:
        elapsed = self.get_timestamp() - self.reference_wall_time
        if elapsed < 0.0:
            return 0.0
        return elapsed

    def get_elapsed_seconds(self) -> float:
        return self.get_run_elapsed_seconds() + self.elapsed_offset_seconds

    def epoch_for_elapsed(self, elapsed_seconds: float) -> float:
        return self.reference_wall_time + (elapsed_seconds - self.elapsed_offset_seconds)

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
            except (ValueError, KeyError, OSError):
                pass
        clock = cls()
        clock.save_to_file(file_path)
        return clock
