"""The parts of the command line that every command needs.

The commands themselves are in the modules beside this one.
"""

import asyncio
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

# How long the stop waits for the finished flag after the agents of the
# session stopped. The agents are stopped first and the flag is set
# afterwards, so the stop has to wait for it.
AGENTS_GONE_GRACE_SECONDS = 120.0

# The commands take what they need from here, so that they do not have to
# know where it lives.
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
    """Let the console print any script.

    The Windows console uses the code page of the machine, which is
    cp1252 on most of them. A lecture in Urdu or Hindi puts characters
    into the dashboard that cp1252 cannot encode, and the whole command
    died with an encoding error. The notes are written before the
    dashboard is shown, so the notes were there, but the session was
    never marked as finished and the stop waited for the full timeout.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Every command answered to the console, so the console has to be ready
# before the command runs. Calling it here runs it for every command of
# the package. It used to exist without ever being called, and the
# cp1252 crash it was written against still happened.
_use_utf_8_output()


def fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def option_value(value: Any, fallback: Any = None) -> Any:
    """The value of a command option.

    Typer hands the option object itself to the function when the command
    is called directly instead of through the command line, for example in
    a test. The value that was asked for is the default in that case.
    """
    if value is None:
        return fallback
    if isinstance(value, typer.models.OptionInfo):
        default = value.default
        if default is None or isinstance(default, (list, dict)):
            return fallback
        return default
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

    return asyncio.run(ping())


def check_ollama(settings: Settings, max_retries: int = 3, retry_delay_seconds: float = 0.8) -> bool:
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(settings.ollama_url.rstrip("/") + "/api/tags", timeout=3) as response:
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
    environment = os.environ.copy()
    environment["LECTURELENS_SESSION_NAME"] = session_id
    environment["LECTURELENS_AUDIO_SOURCE"] = audio_source
    environment["LECTURELENS_MONITOR_INDEX"] = str(monitor_index)
    environment["LECTURELENS_LLM_MODE"] = llm_mode
    environment["LECTURELENS_OUTPUT_DIR"] = str(settings.output_dir.resolve())
    environment["LECTURELENS_ROUTER_CONFIG_PATH"] = str(settings.router_config_path.resolve())
    environment["PYTHONUNBUFFERED"] = "1"
    if window_handle is not None:
        environment["LECTURELENS_CAPTURE_TARGET"] = "window"
        environment["LECTURELENS_WINDOW_HANDLE"] = str(window_handle)
        if window_title:
            environment["LECTURELENS_WINDOW_TITLE"] = window_title
    if course:
        environment["LECTURELENS_COURSE_NAME"] = course
    if settings.postgres_url:
        environment["LECTURELENS_POSTGRES_URL"] = settings.postgres_url
    if settings.redis_url:
        environment["LECTURELENS_REDIS_URL"] = str(settings.redis_url)
    if settings.ollama_url:
        environment["LECTURELENS_OLLAMA_URL"] = str(settings.ollama_url)
    environment["LECTURELENS_ENABLE_DRIFT_FILTER"] = str(settings.enable_drift_filter).lower()
    environment["LECTURELENS_DRIFT_SIMILARITY_THRESHOLD"] = str(settings.drift_similarity_threshold)
    environment["LECTURELENS_DRIFT_THRESHOLD"] = str(settings.drift_similarity_threshold)
    if settings.local_model:
        environment["LECTURELENS_LOCAL_MODEL"] = str(settings.local_model)
    if settings.whisper_hallucination_phrases:
        environment["LECTURELENS_WHISPER_HALLUCINATION_PHRASES"] = ",".join(
            settings.whisper_hallucination_phrases
        )
    if settings.ollama_model:
        environment["LECTURELENS_OLLAMA_MODEL"] = str(settings.ollama_model)
    if settings.primary_provider:
        environment["LECTURELENS_PRIMARY_PROVIDER"] = str(settings.primary_provider)
    if settings.unorouter_model:
        environment["LECTURELENS_UNOROUTER_MODEL"] = str(settings.unorouter_model)
    if settings.unorouter_api_key:
        environment["LECTURELENS_UNOROUTER_API_KEY"] = str(settings.unorouter_api_key)
    if settings.unorouter_api_keys:
        environment["LECTURELENS_UNOROUTER_API_KEYS"] = str(settings.unorouter_api_keys)
    if settings.google_api_key:
        environment["LECTURELENS_GOOGLE_API_KEY"] = str(settings.google_api_key)
    if settings.google_api_keys:
        environment["LECTURELENS_GOOGLE_API_KEYS"] = str(settings.google_api_keys)
    if settings.gemini_api_key:
        environment["LECTURELENS_GEMINI_API_KEY"] = str(settings.gemini_api_key)
    if settings.gemini_api_keys:
        environment["LECTURELENS_GEMINI_API_KEYS"] = str(settings.gemini_api_keys)
    if settings.openrouter_api_key:
        environment["LECTURELENS_OPENROUTER_API_KEY"] = str(settings.openrouter_api_key)
    if settings.openrouter_api_keys:
        environment["LECTURELENS_OPENROUTER_API_KEYS"] = str(settings.openrouter_api_keys)
    # The credentials may also come straight from the environment. The
    # Settings object only reads the LECTURELENS_ prefixed names, so the
    # unprefixed spellings documented in .env.example are copied here.
    for plural_key in ("GOOGLE_API_KEYS", "GEMINI_API_KEYS", "OPENROUTER_API_KEYS", "UNOROUTER_API_KEYS"):
        if plural_key in os.environ and f"LECTURELENS_{plural_key}" not in environment:
            environment[f"LECTURELENS_{plural_key}"] = os.environ[plural_key]
    for single_key in ("UNOROUTER_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if single_key in os.environ and f"LECTURELENS_{single_key}" not in environment:
            environment[f"LECTURELENS_{single_key}"] = os.environ[single_key]
    return environment


def plan_agents(
    mode: str,
    audio_only: bool,
    no_screen: bool = False,
    enable_persister: bool = False
) -> list[str]:
    if mode == "server":
        planned = ["sync_agent", "structuring_agent", "gateway"]
        if enable_persister:
            planned.append("persister")
        return planned
    planned = ["sync_agent", "structuring_agent", "audio_agent"]
    if not audio_only:
        if not no_screen:
            planned.append("screen_agent")
    if enable_persister:
        planned.append("persister")
    return planned


def module_for_agent(agent_name: str) -> str:
    if agent_name == "gateway":
        return "lecturelens.gateway.server"
    if agent_name == "persister":
        return "lecturelens.storage.persister"
    return f"lecturelens.agents.{agent_name}"


async def render_stream_backlogs(bus: RedisBus, session_id: str) -> dict[str, int]:
    backlogs: dict[str, int] = {}
    consumer_group = Settings().sync_consumer_group
    for source_type in ("transcript", "slide"):
        stream_name = f"lecturelens:{session_id}:stream:{source_type}"
        backlogs[source_type] = await bus.get_group_backlog(stream_name, consumer_group)
    return backlogs


def structuring_backlog_bytes(session_dir: Path) -> int:
    timeline_path = session_dir / "timeline.jsonl"
    context_path = session_dir / "context.json"
    if not timeline_path.exists():
        return 0
    processed = 0
    if context_path.exists():
        try:
            import json
            with open(context_path, "r", encoding="utf-8") as file_handle:
                state = json.load(file_handle)
            # The structuring agent writes "timeline_byte_offset"; reading
            # "processed_byte_offset" always returned 0, so the backlog never
            # shrank in the dashboard.
            processed = int(state.get("timeline_byte_offset", 0))
        except (ValueError, OSError):
            processed = 0
    return max(0, timeline_path.stat().st_size - processed)


def build_agents_table(rows: list[dict[str, str]]) -> Table:
    table = Table(title="Agents")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Process")
    table.add_column("Stats")
    for row in rows:
        table.add_row(row["agent"], row["status"], row["process"], row["stats"])
    return table


def build_router_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(title="LLM Router")
    table.add_column("Provider")
    table.add_column("State")
    table.add_column("Calls/window")
    table.add_column("Remaining")
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
    # An agent that died says why in its log and nowhere else, so the
    # reason is put next to it.
    for row in rows:
        if row.get("status") != "ERROR":
            continue
        reason = agent_failure_reason(session_dir / "logs" / f"{row['agent']}.log")
        if reason:
            row["stats"] = safe_for_console(reason, 90)
    backlogs = await render_stream_backlogs(bus, session_id)

    queue_table = Table(title="Queues")
    queue_table.add_column("Stream")
    queue_table.add_column("Unprocessed")
    for source_type, backlog in backlogs.items():
        queue_table.add_row(source_type, str(backlog))
    queue_table.add_row("structuring (bytes)", str(structuring_backlog_bytes(session_dir)))

    active_llm_mode = settings.llm_mode
    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        try:
            meta = session_meta.load_session_meta(session_dir)
            if meta.get("llm_mode"):
                active_llm_mode = meta["llm_mode"]
            elif meta.get("runs") and isinstance(meta["runs"], list) and len(meta["runs"]) > 0:
                last_run = meta["runs"][-1]
                if isinstance(last_run, dict):
                    config = last_run.get("configuration") or {}
                    if config.get("llm_mode"):
                        active_llm_mode = config["llm_mode"]
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
            elapsed_seconds = 0.0
    entry_count = line_counter.refresh()
    snippet = last_transcript_snippet(session_dir / "timeline.jsonl")
    notes_lines = 0
    notes_path = session_dir / "notes.md"
    if notes_path.exists():
        try:
            with open(notes_path, "rb") as file_handle:
                notes_lines = file_handle.read().count(b"\n")
        except Exception:
            notes_lines = 0

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
                    # A failure to respawn an agent must not take down the
                    # dashboard loop, otherwise the whole session is torn down.
                    live.update(Panel(f"agent restart error: {error}"))
                try:
                    snapshot = await collect_snapshot(
                        settings, session_id, bus, monitor.check_health(), line_counter
                    )
                    live.update(snapshot)
                except Exception as error:
                    live.update(Panel(f"dashboard error: {error}"))
                await asyncio.sleep(1.0)
    finally:
        await bus.close()


def run_shutdown(monitor: AgentMonitor, session_dir: Path, planned_agents: list[str]) -> None:
    capture_names: list[str] = []
    for name in planned_agents:
        if name in CAPTURE_AGENT_NAMES or name == "gateway":
            capture_names.append(name)
    phases: list[tuple[list[str], str, Optional[float]]] = [
        (capture_names, STOP_FLAG_NAME, 30.0),
        (["sync_agent"], SYNC_STOP_FLAG_NAME, 60.0),
        (["structuring_agent"], FINALIZE_FLAG_NAME, 300.0),
        (["persister"], PERSISTER_STOP_FLAG_NAME, 30.0)
    ]
    unfinished: list[str] = []
    try:
        with console.status("[bold cyan]Stopping agents, flushing timeline, and generating final notes with LLM...[/bold cyan]", spinner="dots"):
            outcomes = monitor.run_shutdown_sequence(session_dir, phases)
        unfinished = [name for name, clean in outcomes.items() if not clean]
    except KeyboardInterrupt:
        # The user pressed the key again. The agents have their stop
        # flags, so they finish on their own and the queue of the session
        # is still drained. The session is marked as finished below, so
        # the stop command does not wait for a timeout either.
        logger.warning("the shutdown was interrupted, the agents finish on their own")
    except Exception as error:
        # The shutdown is the cleanup path. When it fails the agents are
        # left running and the session is never marked as finished, so the
        # stop command waits for the whole timeout for nothing.
        logger.error("the shutdown sequence failed: %s", error)
    try:
        if unfinished:
            console.print(
                "[yellow]These agents did not finish in time and their remaining work was "
                f"discarded: {', '.join(sorted(unfinished))}[/yellow]"
            )
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
    # The flag is set before anything is printed, because a console that
    # cannot show a character must not keep the session from finishing.
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
