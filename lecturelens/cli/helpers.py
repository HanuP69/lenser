import json
import re
import sys
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
    if state == "failed":
        # session_has_live_agents treats "failed" as a terminal state, so a
        # crashed agent used to be reported as IDLE here.
        return "ERROR"
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


# A line of a log that says what went wrong: the name of an exception,
# or a line that was logged as an error.
ERROR_LINE_PATTERN = re.compile(
    r"(^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\b)|(\bError:\s)|(\bERROR\b)"
)


def agent_failure_reason(log_path: Path, max_bytes: int = 16384) -> Optional[str]:
    """The reason an agent died, taken from its log.

    An agent that cannot start says so in its log and nowhere else, so
    the reason was only visible when the log was opened by hand. The
    last line that says what went wrong is shown instead.
    """
    if not log_path.exists():
        return None
    size = log_path.stat().st_size
    if size == 0:
        return None
    with open(log_path, "rb") as file_handle:
        file_handle.seek(max(0, size - max_bytes))
        data = file_handle.read()
    for raw_line in reversed(data.split(b"\n")):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if len(line) == 0:
            continue
        if "Traceback (most recent call last)" in line:
            continue
        # The frames of the traceback say where it happened, not what
        # went wrong.
        if line.startswith(("File ", "  ", "\t")):
            continue
        if ERROR_LINE_PATTERN.search(line):
            return line
    return None


def safe_for_console(text: str, limit: Optional[int] = None) -> str:
    """Text that the console is able to print.

    The console of a machine uses the code page of the machine, and a
    lecture in Urdu or Hindi has characters that cp1252 cannot encode.
    Printing one of them ends the command with an encoding error, which
    took the whole session down. The characters that cannot be shown are
    dropped instead.
    """
    if not text:
        return ""
    cleaned = text
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        cleaned.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        cleaned = cleaned.encode(encoding, errors="ignore").decode(encoding, errors="ignore")
    cleaned = "".join(
        character for character in cleaned if character == "\n" or character >= " "
    )
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[:limit] + "..."
    return cleaned


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
    return safe_for_console(snippet, max_characters)


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


def check_agent_dependencies(agent_names: list[str]) -> list[str]:
    """Whether the agents can be started at all.

    An agent that cannot import what it needs dies at once, and the
    reason is only in its log, where nobody looks. The command checks it
    before it starts, so the reason is shown right away.

    The check runs in a process of its own, because the agent runs in a
    process of its own. An import that was already done here says
    nothing about the agent.
    """
    import subprocess
    import sys

    problems: list[str] = []
    for agent_name in agent_names:
        module_name = f"lecturelens.agents.{agent_name}"
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module_name}"],
                capture_output=True,
                text=True,
                timeout=120
            )
        except Exception as error:
            problems.append(f"{agent_name} could not be checked: {error}")
            continue
        if result.returncode == 0:
            continue
        missing = _missing_module_name(result.stderr)
        if missing:
            problems.append(
                f"{agent_name} needs the module {missing}, which is not installed. "
                f"Run: pip install -e \".[full]\""
            )
        else:
            last_line = ""
            for line in reversed(result.stderr.splitlines()):
                if line.strip():
                    last_line = line.strip()
                    break
            problems.append(f"{agent_name} cannot be started: {last_line or 'unknown error'}")
    return problems


def _missing_module_name(stderr: str) -> Optional[str]:
    """The name of the module that is not there, if the log says so."""
    for line in stderr.splitlines():
        if "No module named" in line:
            marker = "No module named"
            start = line.find(marker)
            if start < 0:
                continue
            remainder = line[start + len(marker):].strip().strip("'\"")
            return remainder if remainder else None
    return None


def get_free_disk_bytes(directory: Path) -> int:
    import shutil
    probe = directory
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(str(probe)).free)


def format_low_disk_warning(free_bytes: int, threshold_bytes: Optional[int] = None) -> Optional[str]:
    limit = threshold_bytes if threshold_bytes is not None else LOW_DISK_BYTES
    if free_bytes >= limit:
        return None
    free_gigabytes = free_bytes / float(1024 ** 3)
    return (
        f"Only {free_gigabytes:.1f} GB free on the output drive. "
        "Raw audio and screenshots grow with every minute of lecture."
    )


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024.0:.1f} KB"
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MB"


def count_file_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    try:
        with open(file_path, "rb") as file_handle:
            return file_handle.read().count(b"\n")
    except Exception:
        return 0


ALLOWED_COUNT_TABLE_NAMES: frozenset[str] = frozenset({"concepts", "applied_segments"})


def count_sqlite_rows(db_path: Path, table_name: str) -> Optional[int]:
    if not db_path.exists():
        return None
    if table_name not in ALLOWED_COUNT_TABLE_NAMES:
        # The table name cannot be parameterised, so it is restricted to a
        # fixed set instead of being interpolated into the statement.
        return None
    try:
        import sqlite3
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cursor.fetchone()
            if row:
                return int(row[0])
    except Exception:
        return None
    return None


def count_images_in_directory(dir_path: Path) -> tuple[int, int]:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0, 0
    total_count = 0
    total_bytes = 0
    try:
        for file_path in dir_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                total_count += 1
                total_bytes += file_path.stat().st_size
    except Exception:
        return 0, 0
    return total_count, total_bytes


def build_session_output_table(session_dir: Path) -> Any:
    from rich.table import Table
    table = Table(title="Generated Session Files", show_header=True, header_style="bold cyan")
    table.add_column("File / Artifact", style="bold white", width=22)
    table.add_column("Description", style="white", width=34)
    table.add_column("Size / Details", style="green", width=24)
    table.add_column("Status", style="bold green", width=10)

    rows_added = 0

    notes_path = session_dir / "notes.md"
    if notes_path.exists():
        size_str = format_file_size(notes_path.stat().st_size)
        line_count = count_file_lines(notes_path)
        table.add_row("notes.md", "Structured Lecture Notes (Markdown)", f"{size_str} ({line_count} lines)", "Saved")
        rows_added += 1

    timeline_path = session_dir / "timeline.jsonl"
    if timeline_path.exists():
        size_str = format_file_size(timeline_path.stat().st_size)
        event_count = count_file_lines(timeline_path)
        table.add_row("timeline.jsonl", "Correlated Event Timeline", f"{size_str} ({event_count} events)", "Saved")
        rows_added += 1

    concepts_path = session_dir / "concepts.db"
    if concepts_path.exists():
        size_str = format_file_size(concepts_path.stat().st_size)
        concept_count = count_sqlite_rows(concepts_path, "concepts")
        count_str = f"{concept_count} concepts ({size_str})" if concept_count is not None else size_str
        table.add_row("concepts.db", "Knowledge Graph (SQLite)", count_str, "Saved")
        rows_added += 1

    slides_dir = session_dir / "raw" / "slides"
    if not slides_dir.exists():
        slides_dir = session_dir / "raw" / "screen"
    slides_count, slides_bytes = count_images_in_directory(slides_dir)
    if slides_count > 0:
        table.add_row("raw/slides/", "Slide / Screen Keyframes", f"{slides_count} images ({format_file_size(slides_bytes)})", "Saved")
        rows_added += 1

    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        size_str = format_file_size(meta_path.stat().st_size)
        table.add_row("session_meta.json", "Session Run Metadata", size_str, "Saved")
        rows_added += 1

    if rows_added == 0:
        table.add_row("No files", "No artifacts were written to this session", "-", "Empty")

    return table


def display_session_output_summary(session_dir: Path, console: Any) -> None:
    from rich.panel import Panel

    resolved_path = session_dir.resolve()
    session_id = session_dir.name

    header_text = (
        f"[bold white]Session Name[/bold white] : [cyan]{session_id}[/cyan]\n"
        f"[bold white]Output Folder[/bold white]: [green]{resolved_path}[/green]"
    )
    console.print("")
    console.print(Panel(header_text, title="[bold green]LectureLens Output Location[/bold green]", border_style="green"))
    console.print(build_session_output_table(session_dir))
    console.print(
        "[bold cyan]Quick Commands:[/bold cyan]\n"
        f"  [white]* View Notes  :[/white] [dim]type \"{resolved_path}\\notes.md\"[/dim]\n"
        f"  [white]* Ask Lecture :[/white] [dim]lecturelens ask --session \"{session_id}\" \"Summarize the key points\"[/dim]\n"
        f"  [white]* View Graph  :[/white] [dim]lecturelens review --session \"{session_id}\"[/dim]\n"
        f"  [white]* Export Notes:[/white] [dim]lecturelens export --session \"{session_id}\" --format md[/dim]\n"
    )
