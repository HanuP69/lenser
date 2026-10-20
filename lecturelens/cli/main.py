import asyncio
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from lecturelens.core.event_loop import ensure_event_loop_policy

ensure_event_loop_policy()

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from lecturelens.cli.helpers import (
    LineCounter,
    build_agent_rows,
    display_session_output_summary,
    find_latest_session,
    format_elapsed,
    format_low_disk_warning,
    get_free_disk_bytes,
    last_transcript_snippet,
    validate_export_format,
    validate_start_options
)
from lecturelens.config.settings import Settings
from lecturelens.core import session_meta
from lecturelens.core.agent_runtime import (
    PAUSE_FLAG_NAME,
    PERSISTER_STOP_FLAG_NAME,
    SESSION_FINISHED_FLAG_NAME,
    SHUTDOWN_REQUEST_FLAG_NAME,
    FINALIZE_FLAG_NAME,
    STOP_FLAG_NAME,
    SYNC_STOP_FLAG_NAME,
    clear_all_flags,
    clear_flag,
    flag_exists,
    read_heartbeats,
    set_flag
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.process_manager import AgentProcessManager as AgentMonitor
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_course_name, is_valid_session_id

app = typer.Typer(help="LectureLens - local-first multimodal lecture note taking")
console = Console()

CAPTURE_AGENT_NAMES: tuple[str, ...] = ("audio_agent", "screen_agent")


def fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


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
        bus = RedisBus(settings.redis_url)
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
        from lecturelens.storage.postgres import PostgresStorage
        storage = PostgresStorage(settings.postgres_url)
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
    course: Optional[str]
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["LECTURELENS_SESSION_NAME"] = session_id
    environment["LECTURELENS_AUDIO_SOURCE"] = audio_source
    environment["LECTURELENS_MONITOR_INDEX"] = str(monitor_index)
    environment["LECTURELENS_LLM_MODE"] = llm_mode
    environment["LECTURELENS_OUTPUT_DIR"] = str(settings.output_dir.resolve())
    environment["LECTURELENS_ROUTER_CONFIG_PATH"] = str(settings.router_config_path.resolve())
    environment["PYTHONUNBUFFERED"] = "1"
    if course:
        environment["LECTURELENS_COURSE_NAME"] = course
    if settings.postgres_url:
        environment["LECTURELENS_POSTGRES_URL"] = settings.postgres_url
    if "UNOROUTER_API_KEY" in os.environ and "LECTURELENS_UNOROUTER_API_KEY" not in environment:
        environment["LECTURELENS_UNOROUTER_API_KEY"] = os.environ["UNOROUTER_API_KEY"]
    if "UNOROUTER_API_KEYS" in os.environ and "LECTURELENS_UNOROUTER_API_KEYS" not in environment:
        environment["LECTURELENS_UNOROUTER_API_KEYS"] = os.environ["UNOROUTER_API_KEYS"]
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
    for source_type in ("transcript", "slide"):
        stream_name = f"lecturelens:{session_id}:stream:{source_type}"
        backlogs[source_type] = await bus.get_group_backlog(stream_name, "group:kuudere")
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
                processed = int(json.load(file_handle).get("processed_byte_offset", 0))
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
    bus = RedisBus(settings.redis_url)
    line_counter = LineCounter(session_dir / "timeline.jsonl")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                if flag_exists(session_dir, SHUTDOWN_REQUEST_FLAG_NAME):
                    return
                monitor.restart_failed_agents()
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
        (["sync_agent"], SYNC_STOP_FLAG_NAME, 30.0),
        (["structuring_agent"], FINALIZE_FLAG_NAME, 120.0),
        (["persister"], PERSISTER_STOP_FLAG_NAME, 10.0)
    ]
    with console.status("[bold cyan]Stopping agents, flushing timeline, and generating final notes with LLM...[/bold cyan]", spinner="dots"):
        monitor.run_shutdown_sequence(session_dir, phases)
    clock_path = session_dir / "clock.json"
    if clock_path.exists():
        try:
            clock = SessionClock.load_from_file(clock_path)
            clock.save_to_file(clock_path)
        except Exception:
            pass
    session_meta.finish_run(session_dir)
    set_flag(session_dir, SESSION_FINISHED_FLAG_NAME)
    notes_file = session_dir / "notes.md"
    if notes_file.exists() and notes_file.stat().st_size > 0:
        console.print("[green]Session stopped. Notes saved.[/green]")
    else:
        console.print("[yellow]Session stopped. Final notes not generated or empty.[/yellow]")
    display_session_output_summary(session_dir, console)


@app.command()
def start(
    mode: str = typer.Option("local", help="local or server"),
    audio_source: str = typer.Option("mic", "--audio-source", help="mic, system or both"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    persist: bool = typer.Option(False, "--persist", help="Enable durable Postgres persister in local mode"),
    session: Optional[str] = typer.Option(None, "--session"),
    course: Optional[str] = typer.Option(None, "--course"),
    monitor_index: int = typer.Option(1, "--monitor"),
    llm_mode: str = typer.Option("local", "--llm-mode", help="local, hybrid or cloud"),
    resume: bool = typer.Option(False, "--resume", help="continue an existing session"),
) -> None:
    settings = Settings()

    try:
        validate_start_options(audio_source, llm_mode, mode)
    except ValueError as error:
        fail(str(error))
    if mode != "server" and audio_source == "system":
        from lecturelens.agents.audio_agent import find_loopback_microphone
        _, loopback_err = find_loopback_microphone()
        if loopback_err:
            fail(f"Cannot use '--audio-source system': {loopback_err}")
    if course is not None and not is_valid_course_name(course):
        fail(f"Invalid course name: {course}")

    session_id = session or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if not is_valid_session_id(session_id):
        fail(f"Invalid session name: {session_id}")

    session_dir = settings.get_session_output_dir(session_id)
    if (session_dir / "session_meta.json").exists() and not resume:
        fail(f"Session '{session_id}' already exists. Use --resume to continue it or pick another name.")

    if not check_redis(settings):
        fail(f"Redis is not reachable at {settings.redis_url}")
    if llm_mode != "cloud" and not check_ollama(settings):
        console.print(
            f"[yellow]Ollama not reachable at {settings.ollama_url}. "
            "Capture continues; structuring will retry when it returns.[/yellow]"
        )

    settings.create_session_directories(session_id)
    disk_warning = format_low_disk_warning(get_free_disk_bytes(settings.output_dir), settings.low_disk_bytes)
    if disk_warning is not None:
        console.print(f"[yellow]{disk_warning}[/yellow]")
    clear_all_flags(session_dir)
    offset_seconds = session_meta.get_previous_duration_seconds(session_dir) if resume else 0.0
    clock = SessionClock(elapsed_offset_seconds=offset_seconds)
    clock.save_to_file(session_dir / "clock.json")

    should_persist = persist
    if should_persist and not check_postgres(settings):
        console.print(
            f"[yellow]Postgres is not reachable at {settings.postgres_url}. "
            "Persister disabled; session will run without Postgres storage.[/yellow]"
        )
        should_persist = False

    planned_agents = plan_agents(mode, audio_only, no_screen, enable_persister=should_persist)
    session_meta.begin_run(
        session_dir,
        session_id,
        course,
        {
            "mode": mode,
            "audio_source": audio_source,
            "monitor_index": monitor_index,
            "llm_mode": llm_mode,
            "audio_only": audio_only,
            "no_screen": no_screen,
            "persist": persist
        },
        planned_agents
    )

    environment = build_agent_environment(
        settings, session_id, audio_source, monitor_index, llm_mode, course
    )
    monitor = AgentMonitor()
    for agent_name in planned_agents:
        command = [sys.executable, "-m", module_for_agent(agent_name)]
        if agent_name != "gateway":
            command.extend(["--session", session_id, "--audio-source", audio_source])
        monitor.register_agent(
            agent_name,
            command,
            environment,
            session_dir / "logs" / f"{agent_name}.log"
        )
        monitor.start_agent(agent_name)

    settings.llm_mode = llm_mode
    try:
        asyncio.run(live_dashboard_loop(settings, session_id, monitor, session_dir))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    run_shutdown(monitor, session_dir, planned_agents)


LIVE_HEARTBEAT_MAX_AGE_SECONDS: float = 15.0


def session_has_live_agents(session_dir: Path) -> bool:
    now = time.time()
    for heartbeat in read_heartbeats(session_dir).values():
        state = str(heartbeat.get("state", ""))
        if state in ("stopped", "finished", "failed"):
            continue
        if now - float(heartbeat.get("time", 0.0)) <= LIVE_HEARTBEAT_MAX_AGE_SECONDS:
            return True
    return False


@app.command()
def stop(
    session: Optional[str] = typer.Option(None, "--session"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    timeout_seconds: float = typer.Option(300.0, "--timeout"),
) -> None:
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    clear_flag(session_dir, SESSION_FINISHED_FLAG_NAME)
    set_flag(session_dir, SHUTDOWN_REQUEST_FLAG_NAME)
    console.print(f"Shutdown requested for session {session_id}.")
    if not wait:
        return
    deadline = (
        time.monotonic() + timeout_seconds
        if (timeout_seconds is not None and timeout_seconds > 0)
        else None
    )
    while deadline is None or time.monotonic() < deadline:
        if flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME):
            console.print("[green]Session stopped.[/green]")
            display_session_output_summary(session_dir, console)
            return
        time.sleep(1.0)
    fail("Timed out waiting for the session to finish.")


@app.command()
def pause(session: Optional[str] = typer.Option(None, "--session")) -> None:
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    set_flag(session_dir, PAUSE_FLAG_NAME)
    console.print(f"Capture paused for {session_id}.")


@app.command()
def resume(session: Optional[str] = typer.Option(None, "--session")) -> None:
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    clear_flag(session_dir, PAUSE_FLAG_NAME)
    console.print(f"Capture resumed for {session_id}.")


@app.command()
def status(session: Optional[str] = typer.Option(None, "--session")) -> None:
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)

    async def snapshot() -> Group:
        bus = RedisBus(settings.redis_url)
        try:
            counter = LineCounter(session_dir / "timeline.jsonl")
            return await collect_snapshot(settings, session_id, bus, {}, counter)
        finally:
            await bus.close()

    console.print(asyncio.run(snapshot()))


@app.command()
def export(
    export_format: str = typer.Option("md", "--format", help="Export format (md)"),
    session: Optional[str] = typer.Option(None, "--session"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    try:
        validate_export_format(export_format)
    except ValueError as error:
        fail(str(error))
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    target_dir = output if output is not None else session_dir / "export"
    target_dir.mkdir(parents=True, exist_ok=True)

    notes_path = session_dir / "notes.md"
    if notes_path.exists():
        destination = target_dir / f"{session_id}_notes.md"
        destination.write_bytes(notes_path.read_bytes())
        console.print(f"Markdown: {destination}")
    else:
        console.print("[yellow]No notes.md yet.[/yellow]")


@app.command()
def reprocess(
    session: str = typer.Option(..., "--session"),
    llm_mode: str = typer.Option("local", "--llm-mode"),
) -> None:
    settings = Settings()
    settings.llm_mode = llm_mode
    if not is_valid_session_id(session):
        fail(f"Invalid session name: {session}")
    session_dir = ensure_session_exists(settings, session)
    if not (session_dir / "timeline.jsonl").exists():
        fail("No timeline.jsonl in this session.")
    if session_has_live_agents(session_dir):
        fail("This session still has live agents. Run 'lecturelens stop' first, then reprocess.")
    meta = session_meta.load_session_meta(session_dir)
    if meta.get("course_name"):
        settings.course_name = meta["course_name"]

    backup_files: dict[str, tuple[Path, Path]] = {}
    for filename in ("notes.md", "context.json", "timeline.jsonl", "notes_marker.json"):
        p = session_dir / filename
        if p.exists():
            backup_p = session_dir / f"{filename}.bak"
            shutil.copy2(p, backup_p)
            backup_files[filename] = (p, backup_p)

    async def run() -> int:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus = None
        redis_client = None
        if check_redis(settings):
            bus = RedisBus(settings.redis_url)
            redis_client = bus.client
        try:
            router = build_router_for_task(
                "structuring", redis_client, load_router_config(settings.router_config_path),
                settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds
            )
            agent = StructuringAgent(settings=settings, session_id=session, llm_router=router, redis_bus=None)
            agent.reset_outputs()
            return await agent.process_all_pending()
        finally:
            if bus is not None:
                await bus.close()

    try:
        batches = asyncio.run(run())
        for _, (_, backup_p) in backup_files.items():
            if backup_p.exists():
                backup_p.unlink()
    except Exception as error:
        for _, (p, backup_p) in backup_files.items():
            if backup_p.exists():
                shutil.copy2(backup_p, p)
                backup_p.unlink()
        fail(f"Reprocess failed: {error}")
    console.print(f"[green]Reprocessed {batches} batch(es).[/green]")


@app.command()
def ask(
    question: str = typer.Argument(...),
    session: Optional[str] = typer.Option(None, "--session"),
    llm_mode: str = typer.Option("local", "--llm-mode"),
    student_id: Optional[str] = typer.Option(None, "--student-id", help="Student ID for personalized tutoring and misconception tracking"),
) -> None:
    settings = Settings()
    settings.llm_mode = llm_mode
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)

    async def run() -> tuple[str, list[str]]:
        from lecturelens.knowledge.qa_engine import QAEngine
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus = RedisBus(settings.redis_url)
        try:
            router = None
            try:
                router = build_router_for_task(
                    "qa", bus.client, load_router_config(settings.router_config_path),
                    settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds
                )
            except Exception as error:
                console.print(f"[yellow]Router unavailable, using Ollama directly: {error}[/yellow]")
            storage = None
            if settings.postgres_url:
                try:
                    from lecturelens.storage.postgres import PostgresStorage
                    storage = PostgresStorage(settings.postgres_url)
                except Exception:
                    storage = None

            if student_id and isinstance(student_id, str):
                import json
                from lecturelens.knowledge.student_chat import StudentChatAgent
                from lecturelens.knowledge.student_graph import StudentKnowledgeGraph

                graph_db = settings.output_dir / student_id / "student_graph.db"
                graph = StudentKnowledgeGraph(graph_db)
                chat_agent = StudentChatAgent(
                    llm_router=router,
                    student_graph=graph,
                    ollama_url=settings.ollama_url,
                    model_name=settings.local_model
                )

                timeline_events: list[dict[str, Any]] = []
                notes_segments: list[dict[str, Any]] = []
                if storage is not None:
                    try:
                        timeline_events = await storage.get_timeline_events(session_id, limit=40)
                        notes_segments = await storage.get_general_notes_since(session_id, after_sequence_number=0)
                    except Exception:
                        pass

                if not timeline_events and (session_dir / "timeline.jsonl").exists():
                    try:
                        with open(session_dir / "timeline.jsonl", "r", encoding="utf-8") as file_handle:
                            for line in file_handle:
                                if line.strip():
                                    timeline_events.append(json.loads(line))
                        timeline_events = timeline_events[-40:]
                    except Exception:
                        pass

                try:
                    return await chat_agent.ask(question, timeline_events=timeline_events, notes_segments=notes_segments)
                finally:
                    if storage is not None:
                        await storage.close()

            engine = QAEngine(
                session_dir,
                settings.ollama_url,
                router,
                embedding_model=settings.embedding_model,
                fallback_model=settings.local_model,
                storage=storage
            )
            try:
                return await engine.ask_with_sources(question)
            finally:
                if storage is not None:
                    await storage.close()
        finally:
            await bus.close()

    try:
        answer, references = asyncio.run(run())
    except Exception as error:
        fail(f"Could not answer: {error}")
    console.print(answer)
    if len(references) > 0:
        console.print("\n[bold]Sources[/bold]")
        for reference in references:
            console.print(f"- {reference}")


@app.command()
def review(
    course: Optional[str] = typer.Option(None, "--course"),
    session: Optional[str] = typer.Option(None, "--session"),
    student_id: Optional[str] = typer.Option(None, "--student-id"),
) -> None:
    settings = Settings()
    if student_id and isinstance(student_id, str):
        from lecturelens.knowledge.student_graph import StudentKnowledgeGraph
        graph_db = settings.output_dir / student_id / "student_graph.db"
        if graph_db.exists():
            graph = StudentKnowledgeGraph(graph_db)
            summary = graph.get_student_summary()
            console.print(f"[bold]Student {student_id} Mastery Summary[/bold]")
            console.print(f"Concepts tracked: {summary['concepts_tracked']}, Misconceptions: {summary['total_misconceptions']}")
            for c in summary["concepts"][:10]:
                decayed = c.get("decayed_mastery", c["mastery_level"])
                console.print(f"  - {c['concept_name']}: mastery {c['mastery_level']:.2f} (decayed: {decayed:.2f})")
            return
    if settings.postgres_url:
        from lecturelens.storage.postgres import PostgresStorage

        course_val = course if isinstance(course, str) else None
        session_val = session if isinstance(session, str) else None

        async def fetch():
            storage = PostgresStorage(settings.postgres_url)
            try:
                if course_val:
                    return await storage.get_concepts_for_course(course_val)
                elif session_val:
                    return await storage.get_all_concepts_for_session(session_val)
                return []
            finally:
                await storage.close()

        try:
            concepts = asyncio.run(fetch())
            if concepts:
                console.print(f"[bold]Concepts ({len(concepts)})[/bold]")
                for c in concepts:
                    console.print(f"  - {c['concept_name']}: {c.get('definition') or 'No definition'} (Related: {', '.join(c.get('related_concepts') or [])})")
                return
        except Exception as err:
            console.print(f"[yellow]Could not query Postgres concepts: {err}[/yellow]")
    console.print("No review records found.")


@app.command()
def student(
    server: Optional[str] = typer.Option(None, "--server", help="WebSocket URL of lecture gateway"),
    session: str = typer.Option(..., "--session", help="Session ID to subscribe to"),
    student_id: str = typer.Option("student_1", "--student-id", help="Unique student ID"),
    language: str = typer.Option("English", "--language", help="Target language/style (e.g. English, Hindi, Hinglish)"),
    detail: str = typer.Option("comprehensive", "--detail", help="Detail level (comprehensive, concise, beginner)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LECTURELENS_GATEWAY_API_KEY"),
) -> None:
    settings = Settings()
    resolved_server = server or f"ws://{settings.gateway_host}:{settings.gateway_port}"
    from lecturelens.client.student_client import StudentClient, StudentProfile
    from lecturelens.knowledge.student_graph import StudentKnowledgeGraph

    profile = StudentProfile(
        student_id=student_id,
        preferred_language=language,
        detail_level=detail
    )
    graph_db = settings.output_dir / student_id / "student_graph.db"
    graph = StudentKnowledgeGraph(graph_db)

    router = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        config = load_router_config(settings.router_config_path)
        redis_client = None
        if check_redis(settings):
            bus = RedisBus(settings.redis_url)
            redis_client = bus.client
        router = build_router_for_task(
            "structuring",
            redis_client,
            config,
            settings.ollama_url,
            settings.llm_mode,
            settings.llm_timeout_seconds
        )
    except Exception:
        try:
            from langchain_ollama import ChatOllama
            router = ChatOllama(
                base_url=settings.ollama_url,
                model=settings.local_model,
                num_ctx=8192,
                temperature=0.2
            )
        except Exception:
            router = None

    client = StudentClient(
        server_url=resolved_server,
        session_id=session,
        profile=profile,
        output_dir=settings.output_dir / student_id,
        api_key=api_key,
        llm_router=router,
        student_graph=graph
    )

    console.print(f"[green]Connecting student client '{student_id}' to {resolved_server} (session: {session}, language: {language})...[/green]")
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        console.print("[yellow]Student client stopped.[/yellow]")


@app.command()
def persister(
    session: Optional[str] = typer.Option(None, "--session", help="Session ID to persist"),
) -> None:
    settings = Settings()
    if not settings.postgres_url:
        fail("LECTURELENS_POSTGRES_URL is not configured.")
    session_id = resolve_session(settings, session)
    from lecturelens.storage.persister import PostgresPersister
    from lecturelens.storage.postgres import PostgresStorage

    async def run() -> None:
        redis_bus = RedisBus(settings.redis_url)
        storage = PostgresStorage(settings.postgres_url)
        await storage.init_schema()
        persister_worker = PostgresPersister(settings, redis_bus, storage)
        console.print(f"[green]Persister listening on streams for session {session_id}...[/green]")
        try:
            await persister_worker.run_for_session(session_id)
        finally:
            await storage.close()
            await redis_bus.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("[yellow]Persister stopped.[/yellow]")


@app.command()
def client(
    session: str = typer.Option(..., "--session"),
    server: Optional[str] = typer.Option(None, "--server", help="WebSocket URL of lecture gateway"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LECTURELENS_GATEWAY_API_KEY"),
    audio_source: str = typer.Option("mic", "--audio-source"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    monitor_index: int = typer.Option(1, "--monitor"),
) -> None:
    try:
        validate_start_options(audio_source, "local", "local")
    except ValueError as error:
        fail(str(error))
    if audio_source == "system":
        from lecturelens.agents.audio_agent import find_loopback_microphone
        _, loopback_err = find_loopback_microphone()
        if loopback_err:
            fail(f"Cannot use '--audio-source system': {loopback_err}")
    if not is_valid_session_id(session):
        fail(f"Invalid session name: {session}")
    from lecturelens.client.remote_client import run_remote_client

    settings = Settings()
    settings.audio_source = audio_source
    settings.monitor_index = monitor_index
    resolved_server = server or f"ws://{settings.gateway_host}:{settings.gateway_port}"

    async def run() -> None:
        stop_event = asyncio.Event()
        await run_remote_client(
            settings, resolved_server, api_key, session,
            True, (not audio_only and not no_screen),
            stop_event
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("Client stopped.")


@app.command()
def replay(
    session: str = typer.Option(..., "--session"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier (0 for instant)"),
    target_session: Optional[str] = typer.Option(None, "--target-session", help="Session ID to emit events into"),
) -> None:
    from lecturelens.core.replay import ReplayEngine

    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    timeline_path = session_dir / "timeline.jsonl"
    if not timeline_path.exists():
        fail(f"No timeline.jsonl found in {session_dir}")

    async def run() -> int:
        bus = RedisBus(settings.redis_url)
        try:
            engine = ReplayEngine(timeline_path, redis_bus=bus, speed=speed)
            return await engine.replay(target_session_id=target_session or session_id)
        finally:
            await bus.close()

    try:
        count = asyncio.run(run())
        console.print(f"[green]Replayed {count} event(s) at {speed}x speed.[/green]")
    except Exception as error:
        fail(f"Replay failed: {error}")


if __name__ == "__main__":
    app()
