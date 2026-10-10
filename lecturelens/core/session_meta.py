import json
import time
from pathlib import Path
from typing import Any, Optional

from lecturelens.core.fileio import atomic_write_text


def get_meta_path(session_dir: Path) -> Path:
    return session_dir / "session_meta.json"


def load_session_meta(session_dir: Path) -> dict[str, Any]:
    meta_path = get_meta_path(session_dir)
    if not meta_path.exists():
        return {"runs": []}
    try:
        with open(meta_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (ValueError, OSError):
        return {"runs": []}
    if "runs" not in data or not isinstance(data["runs"], list):
        data["runs"] = []
    return data


def get_previous_duration_seconds(session_dir: Path) -> float:
    meta = load_session_meta(session_dir)
    total_seconds = 0.0
    for run_record in meta["runs"]:
        total_seconds += float(run_record.get("duration_seconds", 0.0))
    return total_seconds


def begin_run(
    session_dir: Path,
    session_id: str,
    course_name: Optional[str],
    configuration: dict[str, Any],
    active_agents: list[str]
) -> None:
    meta = load_session_meta(session_dir)
    meta["session_id"] = session_id
    meta["course_name"] = course_name
    run_record = {
        "started_epoch": time.time(),
        "ended_epoch": None,
        "duration_seconds": 0.0,
        "configuration": configuration,
        "agents_active": active_agents
    }
    meta["runs"].append(run_record)
    atomic_write_text(get_meta_path(session_dir), json.dumps(meta, indent=2))


def finish_run(session_dir: Path) -> None:
    meta = load_session_meta(session_dir)
    if len(meta["runs"]) == 0:
        return
    run_record = meta["runs"][-1]
    ended_epoch = time.time()
    run_record["ended_epoch"] = ended_epoch
    run_record["duration_seconds"] = max(0.0, ended_epoch - float(run_record["started_epoch"]))
    atomic_write_text(get_meta_path(session_dir), json.dumps(meta, indent=2))
