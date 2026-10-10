import json
import time
from pathlib import Path
from typing import Any, Optional

from lecturelens.core.validation import (
    VALID_AUDIO_SOURCES,
    VALID_EXPORT_FORMATS,
    VALID_LLM_MODES,
    VALID_RUN_MODES
)

HEARTBEAT_STALE_SECONDS = 10.0
LOW_DISK_BYTES = 2 * 1024 * 1024 * 1024


def validate_choice(value: str, allowed: tuple[str, ...], option_name: str) -> str:
    if value not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"{option_name} must be one of: {allowed_text} (got '{value}')")
    return value


def validate_start_options(audio_source: str, llm_mode: str, run_mode: str) -> None:
    validate_choice(audio_source, VALID_AUDIO_SOURCES, "--audio-source")
    validate_choice(llm_mode, VALID_LLM_MODES, "--llm-mode")
    validate_choice(run_mode, VALID_RUN_MODES, "--mode")


def validate_export_format(export_format: str) -> str:
    return validate_choice(export_format, VALID_EXPORT_FORMATS, "--format")


def format_elapsed(seconds_value: float) -> str:
    total_seconds = int(max(0.0, seconds_value))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def find_latest_session(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    newest_name: Optional[str] = None
    newest_time = -1.0
    for entry in output_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        marker = entry / "session_meta.json"
        if not marker.exists():
            continue
        modified = marker.stat().st_mtime
        if modified > newest_time:
            newest_time = modified
            newest_name = entry.name
    return newest_name


def classify_heartbeat(heartbeat: Optional[dict[str, Any]], now_epoch: float) -> str:
    if heartbeat is None:
        return "NO DATA"
    state = str(heartbeat.get("state", "unknown"))
    if state == "stopped":
        return "STOPPED"
    age = now_epoch - float(heartbeat.get("time", 0.0))
    if age > HEARTBEAT_STALE_SECONDS:
        return "ERROR"
    if state == "paused":
        return "PAUSED"
    if state == "live":
        return "LIVE"
    return "IDLE"


def tail_last_line(file_path: Path, max_bytes: int = 8192) -> Optional[str]:
    if not file_path.exists():
        return None
    size = file_path.stat().st_size
    if size == 0:
        return None
    with open(file_path, "rb") as file_handle:
        file_handle.seek(max(0, size - max_bytes))
        data = file_handle.read()
    lines = data.split(b"\n")
    for raw_line in reversed(lines):
        if len(raw_line.strip()) > 0:
            return raw_line.decode("utf-8", errors="replace")
    return None


def last_transcript_snippet(timeline_path: Path, max_characters: int = 120) -> str:
    if not timeline_path.exists():
        return ""
    size = timeline_path.stat().st_size
    with open(timeline_path, "rb") as file_handle:
        file_handle.seek(max(0, size - 65536))
        data = file_handle.read()
    snippet = ""
    for raw_line in data.split(b"\n"):
        if len(raw_line.strip()) == 0:
            continue
        try:
            entry = json.loads(raw_line.decode("utf-8"))
        except ValueError:
            continue
        if entry.get("source_type") == "transcript":
            snippet = str(entry.get("content", ""))
    if len(snippet) > max_characters:
        return snippet[:max_characters] + "..."
    return snippet


class LineCounter:
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.offset: int = 0
        self.count: int = 0

    def refresh(self) -> int:
        if not self.file_path.exists():
            return self.count
        size = self.file_path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.count = 0
        if size == self.offset:
            return self.count
        with open(self.file_path, "rb") as file_handle:
            file_handle.seek(self.offset)
            data = file_handle.read()
        self.count += data.count(b"\n")
        self.offset += len(data)
        return self.count


def build_agent_rows(
    health: dict[str, str],
    heartbeats: dict[str, dict[str, Any]],
    now_epoch: float
) -> list[dict[str, str]]:
    names: list[str] = []
    for name in health.keys():
        names.append(name)
    for name in heartbeats.keys():
        if name not in names:
            names.append(name)
    rows: list[dict[str, str]] = []
    for name in names:
        heartbeat = heartbeats.get(name)
        process_status = health.get(name, "-")
        status = classify_heartbeat(heartbeat, now_epoch)
        if process_status.startswith("FAILED") or process_status == "GAVE_UP":
            status = "ERROR"
        stats_text = ""
        if heartbeat is not None:
            stat_parts: list[str] = []
            for key, value in (heartbeat.get("stats", {}) or {}).items():
                stat_parts.append(f"{key}={value}")
            stats_text = " ".join(stat_parts)
        rows.append({"agent": name, "status": status, "process": process_status, "stats": stats_text})
    return rows


def get_free_disk_bytes(directory: Path) -> int:
    import shutil
    probe = directory
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(str(probe)).free)


def format_low_disk_warning(free_bytes: int, threshold_bytes: int = LOW_DISK_BYTES) -> Optional[str]:
    if free_bytes >= threshold_bytes:
        return None
    free_gigabytes = free_bytes / float(1024 ** 3)
    return (
        f"Only {free_gigabytes:.1f} GB free on the output drive. "
        "Raw audio and screenshots grow with every minute of lecture."
    )
