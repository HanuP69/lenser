import time
import json
from pathlib import Path


class SessionClock:
    def __init__(self) -> None:
        self.reference_wall_time: float = time.time()
        self.reference_monotonic_time: float = time.monotonic()

    def get_timestamp(self) -> float:
        elapsed = time.monotonic() - self.reference_monotonic_time
        current_timestamp = self.reference_wall_time + elapsed
        return current_timestamp

    def get_elapsed_seconds(self) -> float:
        elapsed = time.monotonic() - self.reference_monotonic_time
        return elapsed

    def to_dict(self) -> dict:
        data = {}
        data["reference_wall_time"] = self.reference_wall_time
        data["reference_monotonic_time"] = self.reference_monotonic_time
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SessionClock":
        clock = cls.__new__(cls)
        clock.reference_wall_time = data["reference_wall_time"]
        clock.reference_monotonic_time = data["reference_monotonic_time"]
        return clock

    def save_to_file(self, file_path: Path) -> None:
        data = self.to_dict()
        json_string = json.dumps(data)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "SessionClock":
        with open(file_path, "r") as file_handle:
            json_string = file_handle.read()
        data = json.loads(json_string)
        clock = cls.from_dict(data)
        return clock
