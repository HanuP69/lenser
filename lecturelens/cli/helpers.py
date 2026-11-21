import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from lecturelens.core.validation import (
    VALID_AUDIO_SOURCES,
    VALID_EXPORT_FORMATS,
    VALID_LLM_MODES,
    VALID_RUN_MODES,
)

HEARTBEAT_STALE_SECONDS = 10.0
LOW_DISK_BYTES = 2 * 1024 * 1024 * 1024
ERROR_LINE_PATTERN = re.compile(r"(^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\b)|(\bError:\s)|(\bERROR\b)")
ALLOWED_COUNT_TABLE_NAMES = frozenset({"concepts", "applied_segments"})


def validate_choice(value: str, allowed: tuple[str, ...], option_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{option_name} must be one of: {', '.join(allowed)} (got '{value}')")
    return value


def validate_start_options(audio_source: str, llm_mode: str, run_mode: str) -> None:
    validate_choice(audio_source, VALID_AUDIO_SOURCES, "--audio-source")
    validate_choice(llm_mode, VALID_LLM_MODES, "--llm-mode")
    validate_choice(run_mode, VALID_RUN_MODES, "--mode")


def validate_export_format(export_format: str) -> str:
    return validate_choice(export_format, VALID_EXPORT_FORMATS, "--format")


def copy_figure_assets(session_dir: Path, target_dir: Path) -> int:
    src, dst = session_dir / "assets", target_dir / "assets"
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pic in src.glob("figure_*.png"):
        try:
            shutil.copy2(pic, dst / pic.name)
            copied += 1
        except OSError:
            pass
    return copied


def format_elapsed(seconds_value: float) -> str:
    s = int(max(0.0, seconds_value))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def find_latest_session(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    candidates = [
        e for e in output_dir.iterdir()
        if e.is_dir() and not e.name.startswith("_") and (e / "session_meta.json").exists()
    ]
    return max(candidates, key=lambda e: (e / "session_meta.json").stat().st_mtime).name if candidates else None


def classify_heartbeat(heartbeat: Optional[dict[str, Any]], now_epoch: float) -> str:
    if not heartbeat:
        return "NO DATA"
    state = str(heartbeat.get("state", "unknown"))
    if state == "stopped":
        return "STOPPED"
    if state == "failed" or (now_epoch - float(heartbeat.get("time", 0.0))) > HEARTBEAT_STALE_SECONDS:
        return "ERROR"
    if state == "paused":
        return "PAUSED"
    return "LIVE" if state == "live" else "IDLE"


def tail_last_line(file_path: Path, max_bytes: int = 8192) -> Optional[str]:
    if not file_path.exists() or file_path.stat().st_size == 0:
        return None
    with open(file_path, "rb") as f:
        f.seek(max(0, file_path.stat().st_size - max_bytes))
        for line in reversed(f.read().split(b"\n")):
            if line.strip():
                return line.decode("utf-8", errors="replace")
    return None


def agent_failure_reason(log_path: Path, max_bytes: int = 16384) -> Optional[str]:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    with open(log_path, "rb") as f:
        f.seek(max(0, log_path.stat().st_size - max_bytes))
        lines = [r.decode("utf-8", errors="replace").strip() for r in f.read().split(b"\n")]
    for line in reversed(lines):
        if line and "Traceback" not in line and not line.startswith(("File ", "  ", "\t")) and ERROR_LINE_PATTERN.search(line):
            return line
    return None


def safe_for_console(text: str, limit: Optional[int] = None) -> str:
    if not text:
        return ""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        cleaned = text
    except (UnicodeEncodeError, LookupError):
        cleaned = text.encode(encoding, errors="ignore").decode(encoding, errors="ignore")
    cleaned = "".join(c for c in cleaned if c == "\n" or c >= " ")
    return f"{cleaned[:limit]}..." if limit is not None and len(cleaned) > limit else cleaned


def last_transcript_snippet(timeline_path: Path, max_characters: int = 120) -> str:
    if not timeline_path.exists():
        return ""
    with open(timeline_path, "rb") as f:
        f.seek(max(0, timeline_path.stat().st_size - 65536))
        raw = f.read()
    snip = ""
    for line in raw.split(b"\n"):
        if line.strip():
            try:
                e = json.loads(line.decode("utf-8"))
                if e.get("source_type") == "transcript":
                    snip = str(e.get("content", ""))
            except Exception:
                pass
    return safe_for_console(snip, max_characters)


class LineCounter:
    def __init__(self, file_path: Path) -> None:
        self.file_path, self.offset, self.count = file_path, 0, 0

    def refresh(self) -> int:
        if not self.file_path.exists():
            return self.count
        sz = self.file_path.stat().st_size
        if sz < self.offset:
            self.offset, self.count = 0, 0
        if sz == self.offset:
            return self.count
        with open(self.file_path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.count += data.count(b"\n")
        self.offset += len(data)
        return self.count


def build_agent_rows(health: dict[str, str], heartbeats: dict[str, dict[str, Any]], now_epoch: float) -> list[dict[str, str]]:
    names = list(dict.fromkeys(list(health.keys()) + list(heartbeats.keys())))
    rows = []
    for name in names:
        hb, p_status = heartbeats.get(name), health.get(name, "-")
        status = classify_heartbeat(hb, now_epoch)
        if p_status.startswith("FAILED") or p_status == "GAVE_UP":
            status = "ERROR"
        stats_text = " ".join(f"{k}={v}" for k, v in (hb.get("stats", {}) or {}).items()) if hb else ""
        rows.append({"agent": name, "status": status, "process": p_status, "stats": stats_text})
    return rows


def _read_requirement_lines(pyproject_text: str) -> list[str]:
    lines = []
    inside = False
    for raw in pyproject_text.splitlines():
        s = raw.strip()
        if not inside:
            if s.endswith("[") and not s.startswith("["):
                inside = True
            continue
        if s.startswith("]"):
            inside = False
            continue
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^"([^"]+)"', s)
        if m:
            lines.append(m.group(1))
    return lines


def check_dependency_versions() -> list[str]:
    from importlib.metadata import PackageNotFoundError, version
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return []
    pfile = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if not pfile.exists():
        return []
    try:
        txt = pfile.read_text(encoding="utf-8")
    except OSError:
        return []
    problems = []
    for req_str in _read_requirement_lines(txt):
        try:
            req = Requirement(req_str)
            inst = version(req.name)
            if req.specifier and not req.specifier.contains(inst, prereleases=True):
                problems.append(f"{req.name} {inst} is installed, but needs {req.specifier}. Run: {install_hint()}")
        except (Exception, PackageNotFoundError):
            continue
    return problems


def install_hint() -> str:
    import importlib.util
    if importlib.util.find_spec("pip") is not None:
        return 'pip install -e ".[full]"'
    if sys.prefix != sys.base_prefix:
        return 'uv pip install -e ".[full]"'
    return 'pip install -e ".[full]"'


def check_agent_dependencies(agent_names: list[str]) -> list[str]:
    problems = []
    for name in agent_names:
        mod = f"lecturelens.agents.{name}"
        try:
            res = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                missing = _missing_module_name(res.stderr) or mod
                problems.append(f"{name} needs the module {missing}, which does not work on this machine. Run: {install_hint()}")
        except Exception as e:
            problems.append(f"{name} dependency check failed: {e}")
    return problems


def _missing_module_name(stderr: str) -> Optional[str]:
    for line in stderr.splitlines():
        if "No module named" in line:
            m = "No module named"
            start = line.find(m)
            if start >= 0:
                rem = line[start + len(m):].strip().strip("'\"")
                if rem:
                    return rem
    return None


def get_free_disk_bytes(directory: Path) -> int:
    probe = directory
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(str(probe)).free)


def format_low_disk_warning(free_bytes: int, threshold_bytes: Optional[int] = None) -> Optional[str]:
    limit = threshold_bytes if threshold_bytes is not None else LOW_DISK_BYTES
    if free_bytes >= limit:
        return None
    return f"Only {free_bytes / float(1024 ** 3):.1f} GB free on output drive. Raw audio and screenshots grow quickly."


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
        with open(file_path, "rb") as f:
            return f.read().count(b"\n")
    except Exception:
        return 0


def count_sqlite_rows(db_path: Path, table_name: str) -> Optional[int]:
    if not db_path.exists() or table_name not in ALLOWED_COUNT_TABLE_NAMES:
        return None
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            r = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return int(r[0]) if r else None
    except Exception:
        return None


def count_images_in_directory(dir_path: Path) -> tuple[int, int]:
    if not dir_path.is_dir():
        return 0, 0
    cnt, total_b = 0, 0
    try:
        for f in dir_path.glob("**/*"):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                cnt += 1
                total_b += f.stat().st_size
    except Exception:
        pass
    return cnt, total_b


def build_session_output_table(session_dir: Path) -> Any:
    from rich.table import Table
    tbl = Table(title="Generated Session Files", show_header=True, header_style="bold cyan")
    tbl.add_column("File / Artifact", style="bold white", width=22)
    tbl.add_column("Description", style="white", width=34)
    tbl.add_column("Size / Details", style="green", width=24)
    tbl.add_column("Status", style="bold green", width=10)

    rows = 0
    for name, desc in [("notes.md", "Structured Lecture Notes (Markdown)"), ("timeline.jsonl", "Correlated Event Timeline")]:
        p = session_dir / name
        if p.exists():
            tbl.add_row(name, desc, f"{format_file_size(p.stat().st_size)} ({count_file_lines(p)} lines)", "Saved")
            rows += 1

    c_path = session_dir / "concepts.db"
    if c_path.exists():
        cnt = count_sqlite_rows(c_path, "concepts")
        cnt_s = f"{cnt} concepts ({format_file_size(c_path.stat().st_size)})" if cnt is not None else format_file_size(c_path.stat().st_size)
        tbl.add_row("concepts.db", "Knowledge Graph (SQLite)", cnt_s, "Saved")
        rows += 1

    s_dir = session_dir / "raw" / "slides"
    if not s_dir.exists():
        s_dir = session_dir / "raw" / "screen"
    scnt, sbytes = count_images_in_directory(s_dir)
    if scnt > 0:
        tbl.add_row("raw/slides/", "Slide / Screen Keyframes", f"{scnt} images ({format_file_size(sbytes)})", "Saved")
        rows += 1

    m_path = session_dir / "session_meta.json"
    if m_path.exists():
        tbl.add_row("session_meta.json", "Session Run Metadata", format_file_size(m_path.stat().st_size), "Saved")
        rows += 1

    if rows == 0:
        tbl.add_row("No files", "No artifacts were written to this session", "-", "Empty")
    return tbl


def display_session_output_summary(session_dir: Path, console: Any) -> None:
    from rich.panel import Panel
    res, sid = session_dir.resolve(), session_dir.name
    console.print("")
    console.print(Panel(f"[bold white]Session Name[/bold white] : [cyan]{sid}[/cyan]\n[bold white]Output Folder[/bold white]: [green]{res}[/green]", title="[bold green]LectureLens Output Location[/bold green]", border_style="green"))
    console.print(build_session_output_table(session_dir))
    console.print(f"[bold cyan]Quick Commands:[/bold cyan]\n  [white]* View Notes  :[/white] [dim]type \"{res}\\notes.md\"[/dim]\n  [white]* Ask Lecture :[/white] [dim]lecturelens ask --session \"{sid}\" \"Summarize the key points\"[/dim]\n  [white]* View Graph  :[/white] [dim]lecturelens review --session \"{sid}\"[/dim]\n  [white]* Export Notes:[/white] [dim]lecturelens export --session \"{sid}\" --format md[/dim]\n")
