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


def count_sqlite_rows(db_path: Path, table_name: str) -> Optional[int]:
    if not db_path.exists():
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

    anki_path = session_dir / "flashcards.apkg"
    if anki_path.exists():
        size_str = format_file_size(anki_path.stat().st_size)
        table.add_row("flashcards.apkg", "Anki Flashcards Deck", size_str, "Saved")
        rows_added += 1

    flashcards_json_path = session_dir / "flashcards.jsonl"
    if flashcards_json_path.exists() and not anki_path.exists():
        size_str = format_file_size(flashcards_json_path.stat().st_size)
        card_count = count_file_lines(flashcards_json_path)
        table.add_row("flashcards.jsonl", "Flashcards Dataset", f"{size_str} ({card_count} cards)", "Saved")
        rows_added += 1

    screen_dir = session_dir / "raw" / "screen"
    screen_count, screen_bytes = count_images_in_directory(screen_dir)
    if screen_count > 0:
        table.add_row("raw/screen/", "Slide / Screen Keyframes", f"{screen_count} images ({format_file_size(screen_bytes)})", "Saved")
        rows_added += 1

    board_dir = session_dir / "raw" / "board"
    board_count, board_bytes = count_images_in_directory(board_dir)
    if board_count > 0:
        table.add_row("raw/board/", "Whiteboard Keyframes", f"{board_count} images ({format_file_size(board_bytes)})", "Saved")
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
        f"  [white]* Export PDF  :[/white] [dim]lecturelens export --session \"{session_id}\" --format pdf[/dim]\n"
    )
