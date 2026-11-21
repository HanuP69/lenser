"""The parts of the command line that every command needs."""

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from lecturelens.cli.helpers import (
    LineCounter,
    agent_failure_reason,
    build_agent_rows,
    copy_figure_assets,
    display_session_output_summary,
    find_latest_session,
    format_elapsed,
    format_low_disk_warning,
    get_free_disk_bytes,
    last_transcript_snippet,
    safe_for_console,
    validate_export_format,
    validate_start_options
)
from lecturelens.config.settings import Settings
from lecturelens.core import session_meta
from lecturelens.core.agent_runtime import (
    FINALIZE_FLAG_NAME,
    PERSISTER_STOP_FLAG_NAME,
    SESSION_FINISHED_FLAG_NAME,
    SHUTDOWN_REQUEST_FLAG_NAME,
    STOP_FLAG_NAME,
    SYNC_STOP_FLAG_NAME,
    create_postgres_storage,
    create_redis_bus,
    flag_exists,
    read_heartbeats,
    set_flag
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.process_manager import AgentProcessManager as AgentMonitor
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_session_id

app = typer.Typer(help="LectureLens - local-first multimodal lecture note taking")
console = Console()
logger = get_logger("lecturelens.cli")

CAPTURE_AGENT_NAMES: tuple[str, ...] = ("audio_agent", "screen_agent")
AGENTS_GONE_GRACE_SECONDS = 120.0

__all__ = [
    "AGENTS_GONE_GRACE_SECONDS",
    "CAPTURE_AGENT_NAMES",
    "LineCounter",
    "agent_failure_reason",
    "app",
    "build_agent_environment",
    "build_agent_rows",
    "build_router_table",
    "check_ollama",
    "check_postgres",
    "check_redis",
    "collect_snapshot",
    "console",
    "copy_figure_assets",
    "display_session_output_summary",
    "ensure_session_exists",
    "fail",
    "find_latest_session",
    "format_elapsed",
    "format_low_disk_warning",
    "get_free_disk_bytes",
    "live_dashboard_loop",
    "logger",
    "module_for_agent",
    "option_value",
    "plan_agents",
    "resolve_session",
    "run_shutdown",
    "safe_for_console",
    "validate_export_format",
    "validate_start_options",
]


def _use_utf_8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_use_utf_8_output()


def fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def option_value(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, typer.models.OptionInfo):
        default = value.default
        return fallback if default is None or isinstance(default, (list, dict)) else default
    if isinstance(value, typer.models.ArgumentInfo):
        return fallback
    return value


def resolve_session(settings: Settings, session: Optional[str]) -> str:
    if session is not None and len(session) > 0:
        if not is_valid_session_id(session):
            fail(f"Invalid session name: {session}")
        return session
    latest = find_latest_session(settings.output_dir)
    if latest is None:
        fail("No sessions found. Use --session to name one.")
    return str(latest)


def ensure_session_exists(settings: Settings, session_id: str) -> Path:
    session_dir = settings.get_session_output_dir(session_id)
    if not session_dir.exists():
        fail(f"Session directory not found: {session_dir}")
    return session_dir


def check_redis(settings: Settings) -> bool:
    async def ping() -> bool:
        bus = create_redis_bus(settings, component_name="ping")
        try:
            return await bus.ping()
        finally:
            await bus.close()
    try:
        return asyncio.run(ping())
    except Exception:
        return False


def check_ollama(settings: Settings, max_retries: int = 3, retry_delay_seconds: float = 0.8) -> bool:
    url = settings.ollama_url.rstrip("/") + "/api/tags"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
    return False


def check_postgres(settings: Settings) -> bool:
    if not settings.postgres_url:
        return False
    async def ping() -> bool:
        storage = create_postgres_storage(settings)
        try:
            conn = await storage.get_connection()
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1;")
                return True
        except Exception:
            return False
        finally:
            await storage.close()
    try:
        return asyncio.run(ping())
    except Exception:
        return False


def build_agent_environment(
    settings: Settings,
    session_id: str,
    audio_source: str,
    monitor_index: int,
    llm_mode: str,
    course: Optional[str],
    window_handle: Optional[int] = None,
    window_title: Optional[str] = None
) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "LECTURELENS_SESSION_NAME": session_id,
        "LECTURELENS_AUDIO_SOURCE": audio_source,
        "LECTURELENS_MONITOR_INDEX": str(monitor_index),
        "LECTURELENS_LLM_MODE": llm_mode,
        "LECTURELENS_OUTPUT_DIR": str(settings.output_dir.resolve()),
        "LECTURELENS_ROUTER_CONFIG_PATH": str(settings.router_config_path.resolve()),
        "PYTHONUNBUFFERED": "1",
        "LECTURELENS_ENABLE_DRIFT_FILTER": str(settings.enable_drift_filter).lower(),
        "LECTURELENS_DRIFT_SIMILARITY_THRESHOLD": str(settings.drift_similarity_threshold),
        "LECTURELENS_DRIFT_THRESHOLD": str(settings.drift_similarity_threshold),
    })
    if window_handle is not None:
        env["LECTURELENS_CAPTURE_TARGET"] = "window"
        env["LECTURELENS_WINDOW_HANDLE"] = str(window_handle)
        if window_title:
            env["LECTURELENS_WINDOW_TITLE"] = window_title
    if course:
        env["LECTURELENS_COURSE_NAME"] = course
    if settings.postgres_url:
        env["LECTURELENS_POSTGRES_URL"] = settings.postgres_url
    if settings.redis_url:
        env["LECTURELENS_REDIS_URL"] = str(settings.redis_url)
    if settings.ollama_url:
        env["LECTURELENS_OLLAMA_URL"] = str(settings.ollama_url)
    if settings.local_model:
        env["LECTURELENS_LOCAL_MODEL"] = str(settings.local_model)
    if settings.whisper_hallucination_phrases:
        env["LECTURELENS_WHISPER_HALLUCINATION_PHRASES"] = ",".join(settings.whisper_hallucination_phrases)
    if settings.ollama_model:
        env["LECTURELENS_OLLAMA_MODEL"] = str(settings.ollama_model)
    if settings.primary_provider:
        env["LECTURELENS_PRIMARY_PROVIDER"] = str(settings.primary_provider)

    for provider in ("unorouter", "google", "gemini", "openrouter"):
        for suffix in ("model", "api_key", "api_keys"):
            attr = f"{provider}_{suffix}"
            val = getattr(settings, attr, None)
            if val:
                env[f"LECTURELENS_{attr.upper()}"] = str(val)

    for key in (
        "GOOGLE_API_KEYS", "GEMINI_API_KEYS", "OPENROUTER_API_KEYS", "UNOROUTER_API_KEYS",
        "UNOROUTER_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"
    ):
        if key in os.environ and f"LECTURELENS_{key}" not in env:
            env[f"LECTURELENS_{key}"] = os.environ[key]
    return env


def plan_agents(
    mode: str,
    audio_only: bool,
    no_screen: bool = False,
    enable_persister: bool = False
) -> list[str]:
    if mode == "server":
        planned = ["sync_agent", "structuring_agent", "gateway"]
    else:
        planned = ["sync_agent", "structuring_agent", "audio_agent"]
        if not audio_only and not no_screen:
            planned.append("screen_agent")
    if enable_persister:
        planned.append("persister")
    return planned


def module_for_agent(agent_name: str) -> str:
    mapping = {
        "gateway": "lecturelens.gateway.server",
        "persister": "lecturelens.storage.persister",
    }
    return mapping.get(agent_name, f"lecturelens.agents.{agent_name}")


async def render_stream_backlogs(bus: RedisBus, session_id: str) -> dict[str, int]:
    consumer_group = Settings().sync_consumer_group
    return {
        st: await bus.get_group_backlog(f"lecturelens:{session_id}:stream:{st}", consumer_group)
        for st in ("transcript", "slide")
    }


def structuring_backlog_bytes(session_dir: Path) -> int:
    timeline, context = session_dir / "timeline.jsonl", session_dir / "context.json"
    if not timeline.exists():
        return 0
    processed = 0
    if context.exists():
        try:
            with open(context, "r", encoding="utf-8") as f:
                processed = int(json.load(f).get("timeline_byte_offset", 0))
        except (ValueError, OSError):
            processed = 0
    return max(0, timeline.stat().st_size - processed)


def build_agents_table(rows: list[dict[str, str]]) -> Table:
    table = Table(title="Agents")
    for col in ("Agent", "Status", "Process", "Stats"):
        table.add_column(col)
    for row in rows:
        table.add_row(row["agent"], row["status"], row["process"], row["stats"])
    return table


def build_router_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(title="LLM Router")
    for col in ("Provider", "State", "Calls/window", "Remaining"):
        table.add_column(col)
    for row in rows:
        table.add_row(row["provider"], str(row["state"]), str(row["calls"]), str(row["remaining"]))
    return table


async def collect_snapshot(
    settings: Settings,
    session_id: str,
    bus: RedisBus,
    health: dict[str, str],
    line_counter: LineCounter
) -> Group:
    from lecturelens.router.llm_router import get_router_status, load_router_config

    session_dir = settings.get_session_output_dir(session_id)
    heartbeats = read_heartbeats(session_dir)
    rows = build_agent_rows(health, heartbeats, time.time())
    for row in rows:
        if row.get("status") == "ERROR":
            reason = agent_failure_reason(session_dir / "logs" / f"{row['agent']}.log")
            if reason:
                row["stats"] = safe_for_console(reason, 90)

    backlogs = await render_stream_backlogs(bus, session_id)
    queue_table = Table(title="Queues")
    queue_table.add_column("Stream")
    queue_table.add_column("Unprocessed")
    for st, b in backlogs.items():
        queue_table.add_row(st, str(b))
    queue_table.add_row("structuring (bytes)", str(structuring_backlog_bytes(session_dir)))

    active_llm_mode = settings.llm_mode
    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        try:
            meta = session_meta.load_session_meta(session_dir)
            if meta.get("llm_mode"):
                active_llm_mode = meta["llm_mode"]
            elif meta.get("runs") and isinstance(meta["runs"], list) and meta["runs"]:
                last_run = meta["runs"][-1]
                if isinstance(last_run, dict):
                    cfg = last_run.get("configuration") or {}
                    if cfg.get("llm_mode"):
                        active_llm_mode = cfg["llm_mode"]
        except Exception:
            pass

    router_config = load_router_config(settings.router_config_path)
    router_rows = await get_router_status(bus.client, router_config, "structuring", active_llm_mode)

    elapsed_seconds = 0.0
    clock_path = session_dir / "clock.json"
    if clock_path.exists():
        try:
            elapsed_seconds = SessionClock.load_from_file(clock_path).get_elapsed_seconds()
        except Exception:
            pass
    entry_count = line_counter.refresh()
    snippet = last_transcript_snippet(session_dir / "timeline.jsonl")
    notes_lines = 0
    notes_path = session_dir / "notes.md"
    if notes_path.exists():
        try:
            with open(notes_path, "rb") as f:
                notes_lines = f.read().count(b"\n")
        except Exception:
            pass

    summary = Panel(
        f"Session: {session_id}   Elapsed: {format_elapsed(elapsed_seconds)}   "
        f"Timeline entries: {entry_count}   Notes lines: {notes_lines}\nLatest: {snippet or '-'}",
        title="LectureLens"
    )
    return Group(summary, build_agents_table(rows), queue_table, build_router_table(router_rows))


async def live_dashboard_loop(
    settings: Settings,
    session_id: str,
    monitor: AgentMonitor,
    session_dir: Path
) -> None:
    bus = create_redis_bus(settings, component_name="dashboard")
    line_counter = LineCounter(session_dir / "timeline.jsonl")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                if flag_exists(session_dir, SHUTDOWN_REQUEST_FLAG_NAME):
                    return
                try:
                    monitor.restart_failed_agents()
                except Exception as error:
                    live.update(Panel(f"agent restart error: {error}"))
                try:
                    snapshot = await collect_snapshot(settings, session_id, bus, monitor.check_health(), line_counter)
                    live.update(snapshot)
                except Exception as error:
                    live.update(Panel(f"dashboard error: {error}"))
                await asyncio.sleep(1.0)
    finally:
        await bus.close()


def run_shutdown(monitor: AgentMonitor, session_dir: Path, planned_agents: list[str]) -> None:
    capture_names = [name for name in planned_agents if name in CAPTURE_AGENT_NAMES or name == "gateway"]
    phases: list[tuple[list[str], str, Optional[float]]] = [
        (capture_names, STOP_FLAG_NAME, None),
        (["sync_agent"], SYNC_STOP_FLAG_NAME, None),
        (["structuring_agent"], FINALIZE_FLAG_NAME, None),
        (["persister"], PERSISTER_STOP_FLAG_NAME, None)
    ]
    unfinished: list[str] = []
    try:
        with console.status("[bold cyan]Stopping agents, flushing timeline, and generating final notes with LLM...[/bold cyan]", spinner="dots"):
            outcomes = monitor.run_shutdown_sequence(session_dir, phases)
        unfinished = [name for name, clean in outcomes.items() if not clean]
    except KeyboardInterrupt:
        logger.warning("the shutdown was interrupted, the agents finish on their own")
    except Exception as error:
        logger.error("the shutdown sequence failed: %s", error)
    if unfinished:
        try:
            console.print(f"[yellow]These agents did not finish in time: {', '.join(sorted(unfinished))}[/yellow]")
        except Exception:
            pass
    clock_path = session_dir / "clock.json"
    if clock_path.exists():
        try:
            clock = SessionClock.load_from_file(clock_path)
            clock.save_to_file(clock_path)
        except Exception:
            pass
    try:
        session_meta.finish_run(session_dir)
    except Exception as error:
        logger.error("the session metadata could not be closed: %s", error)
    set_flag(session_dir, SESSION_FINISHED_FLAG_NAME)
    notes_file = session_dir / "notes.md"
    try:
        if notes_file.exists() and notes_file.stat().st_size > 0:
            console.print("[green]Session stopped. Notes saved.[/green]")
        else:
            console.print("[yellow]Session stopped. Final notes not generated or empty.[/yellow]")
        display_session_output_summary(session_dir, console)
    except Exception as error:
        logger.error("the session summary could not be shown: %s", error)
