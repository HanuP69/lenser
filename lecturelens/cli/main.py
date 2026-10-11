import asyncio
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
from lecturelens.cli.interactive import run_interactive_wizard
from lecturelens.config.settings import Settings
from lecturelens.core import session_meta
from lecturelens.core.agent_runtime import (
    PAUSE_FLAG_NAME,
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
from lecturelens.core.recovery import AgentMonitor
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_course_name, is_valid_session_id

app = typer.Typer(help="LectureLens - local-first multimodal lecture note taking")
console = Console()

CAPTURE_AGENT_NAMES: tuple[str, ...] = ("audio_agent", "screen_agent", "camera_agent")


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


def build_agent_environment(
    settings: Settings,
    session_id: str,
    audio_source: str,
    monitor_index: int,
    llm_mode: str,
    course: Optional[str],
    primary_provider: Optional[str] = None,
    model_name: Optional[str] = None
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
    if primary_provider:
        environment["LECTURELENS_PRIMARY_PROVIDER"] = primary_provider
    if model_name:
        if primary_provider in ("gemma", "ollama"):
            environment["LECTURELENS_OLLAMA_MODEL"] = model_name
        elif primary_provider == "unorouter":
            environment["LECTURELENS_UNOROUTER_MODEL"] = model_name
    unorouter_key = os.environ.get("UNOROUTER_API_KEY") or os.environ.get("LECTURELENS_UNOROUTER_API_KEY")
    if unorouter_key:
        environment["UNOROUTER_API_KEY"] = unorouter_key
        environment["LECTURELENS_UNOROUTER_API_KEY"] = unorouter_key
    return environment


def plan_agents(
    mode: str,
    audio_only: bool,
    no_camera: bool,
    no_screen: bool
) -> list[str]:
    planned = ["sync_agent", "structuring_agent", "audio_agent"]
    if not audio_only:
        if not no_screen:
            planned.append("screen_agent")
        if not no_camera:
            planned.append("camera_agent")
    if mode == "server":
        planned.append("gateway")
    return planned


def module_for_agent(agent_name: str) -> str:
    if agent_name == "gateway":
        return "lecturelens.gateway.server"
    return f"lecturelens.agents.{agent_name}"


async def render_stream_backlogs(bus: RedisBus, session_id: str) -> dict[str, int]:
    backlogs: dict[str, int] = {}
    for source_type in ("transcript", "slide", "board"):
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

    router_config = load_router_config(settings.router_config_path)
    router_rows = await get_router_status(bus.client, router_config, "structuring", settings.llm_mode)

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
        (["structuring_agent"], FINALIZE_FLAG_NAME, 120.0)
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
    no_camera: bool = typer.Option(False, "--no-camera"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    session: Optional[str] = typer.Option(None, "--session"),
    course: Optional[str] = typer.Option(None, "--course"),
    monitor_index: int = typer.Option(1, "--monitor"),
    llm_mode: str = typer.Option("local", "--llm-mode", help="local, hybrid or cloud"),
    resume: bool = typer.Option(False, "--resume", help="continue an existing session"),
    interactive: Optional[bool] = typer.Option(None, "--interactive/--non-interactive", "-i", help="Run interactive setup wizard"),
) -> None:
    settings = Settings()
    primary_provider: Optional[str] = None
    model_name: Optional[str] = None

    should_interactive = False
    if interactive is True:
        should_interactive = True
    elif interactive is None:
        if sys.stdin.isatty() and os.environ.get("LECTURELENS_NON_INTERACTIVE") != "1":
            should_interactive = True

    if should_interactive:
        session_id_initial = session or datetime.now().strftime("%Y-%m-%d_%H%M%S")
        wizard_config = {
            "session_id": session_id_initial,
            "audio_source": audio_source,
            "no_camera": no_camera,
            "no_screen": no_screen,
            "audio_only": audio_only,
            "course": course,
            "monitor_index": monitor_index,
            "llm_mode": llm_mode,
            "provider": "gemma",
            "model": "gemma4:e4b"
        }
        resolved = run_interactive_wizard(wizard_config, settings, console)
        if resolved is None:
            raise typer.Exit(code=0)
        session = resolved["session_id"]
        audio_source = resolved["audio_source"]
        no_camera = resolved["no_camera"]
        no_screen = resolved["no_screen"]
        course = resolved["course"]
        llm_mode = resolved["llm_mode"]
        primary_provider = resolved.get("primary_provider")
        model_name = resolved.get("model")

    try:
        validate_start_options(audio_source, llm_mode, mode)
    except ValueError as error:
        fail(str(error))
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

    if primary_provider == "unorouter" or (llm_mode in ("hybrid", "cloud") and primary_provider != "gemma"):
        from lecturelens.cli.interactive import check_unorouter_health, load_env_api_key
        unorouter_key = os.environ.get("UNOROUTER_API_KEY") or os.environ.get("LECTURELENS_UNOROUTER_API_KEY")
        if not unorouter_key:
            unorouter_key = load_env_api_key()
            if unorouter_key:
                os.environ["UNOROUTER_API_KEY"] = unorouter_key
                os.environ["LECTURELENS_UNOROUTER_API_KEY"] = unorouter_key
        if unorouter_key:
            healthy, health_msg = check_unorouter_health(unorouter_key)
            if healthy:
                console.print(f"[green]UnoRouter API health check: {health_msg}[/green]")
            else:
                if primary_provider == "unorouter" or llm_mode == "cloud":
                    fail(f"UnoRouter health check failed: {health_msg}. Please check your API key.")
                else:
                    console.print(f"[yellow]UnoRouter health check warning: {health_msg}. Local Ollama will be used as fallback.[/yellow]")
        elif primary_provider == "unorouter":
            fail("UnoRouter is selected as primary provider, but UNOROUTER_API_KEY is not set.")

    settings.create_session_directories(session_id)
    disk_warning = format_low_disk_warning(get_free_disk_bytes(settings.output_dir))
    if disk_warning is not None:
        console.print(f"[yellow]{disk_warning}[/yellow]")
    clear_all_flags(session_dir)
    offset_seconds = session_meta.get_previous_duration_seconds(session_dir) if resume else 0.0
    clock = SessionClock(elapsed_offset_seconds=offset_seconds)
    clock.save_to_file(session_dir / "clock.json")

    planned_agents = plan_agents(mode, audio_only, no_camera, no_screen)
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
            "no_camera": no_camera,
            "no_screen": no_screen
        },
        planned_agents
    )

    environment = build_agent_environment(
        settings, session_id, audio_source, monitor_index, llm_mode, course,
        primary_provider=primary_provider, model_name=model_name
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


@app.command()
def stop(
    session: Optional[str] = typer.Option(None, "--session"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    timeout_seconds: float = typer.Option(60.0, "--timeout"),
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
    export_format: str = typer.Option("all", "--format", help="md, anki, pdf or all"),
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
    if export_format in ("md", "all"):
        if notes_path.exists():
            destination = target_dir / f"{session_id}_notes.md"
            destination.write_bytes(notes_path.read_bytes())
            console.print(f"Markdown: {destination}")
        else:
            console.print("[yellow]No notes.md yet.[/yellow]")

    if export_format in ("anki", "all"):
        apkg_path = session_dir / "flashcards.apkg"
        if apkg_path.exists():
            destination = target_dir / f"{session_id}_flashcards.apkg"
            destination.write_bytes(apkg_path.read_bytes())
            console.print(f"Anki: {destination}")
        else:
            console.print("[yellow]No flashcards.apkg yet.[/yellow]")

    if export_format in ("pdf", "all"):
        if not notes_path.exists():
            console.print("[yellow]No notes.md to convert to PDF.[/yellow]")
        else:
            try:
                from lecturelens.exports.pdf_exporter import create_pdf_from_markdown
            except ImportError:
                fail("PDF export needs reportlab: pip install 'lecturelens[pdf]'")
            destination = target_dir / f"{session_id}_notes.pdf"
            create_pdf_from_markdown(
                notes_path.read_text(encoding="utf-8"), destination, f"Lecture notes: {session_id}"
            )
            console.print(f"PDF: {destination}")


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
    meta = session_meta.load_session_meta(session_dir)
    if meta.get("course_name"):
        settings.course_name = meta["course_name"]

    course_db_path = settings.get_course_database_path(settings.course_name)
    backup_files: dict[str, tuple[Path, Path]] = {}
    for filename in ("notes.md", "context.json", "flashcards.jsonl", f"{session}.apkg", "flashcards.apkg", "concepts.db"):
        p = session_dir / filename
        if p.exists():
            backup_p = session_dir / f"{filename}.bak"
            shutil.copy2(p, backup_p)
            backup_files[filename] = (p, backup_p)
    if course_db_path.exists():
        course_backup_p = course_db_path.with_suffix(".db.bak")
        shutil.copy2(course_db_path, course_backup_p)
        backup_files["course_db"] = (course_db_path, course_backup_p)

    async def run() -> int:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus = RedisBus(settings.redis_url)
        try:
            router = build_router_for_task(
                "structuring", bus.client, load_router_config(settings.router_config_path),
                settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds
            )
            agent = StructuringAgent(settings=settings, session_id=session, llm_router=router)
            agent.reset_outputs()
            return await agent.process_all_pending()
        finally:
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
            engine = QAEngine(session_dir, settings.ollama_url, router)
            return await engine.ask_with_sources(question)
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
) -> None:
    from lecturelens.knowledge.concept_graph import ConceptGraph

    settings = Settings()
    if session is not None:
        session_dir = ensure_session_exists(settings, session)
        database_path = session_dir / "concepts.db"
    else:
        database_path = settings.get_course_database_path(course)
    if not database_path.exists():
        fail(f"No concept database found at {database_path}")

    overview = ConceptGraph(database_path).get_course_overview()
    if len(overview) == 0:
        console.print("No concepts recorded yet.")
        return
    for session_key, concepts in overview.items():
        table = Table(title=f"Session: {session_key}")
        table.add_column("Concept")
        table.add_column("Definition")
        table.add_column("Related")
        for concept in concepts:
            table.add_row(
                concept["concept_name"],
                str(concept["definition"] or ""),
                ", ".join(concept["related_concepts"])
            )
        console.print(table)


@app.command()
def client(
    server: str = typer.Option(..., "--server", help="ws://host:8765"),
    session: str = typer.Option(..., "--session"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LECTURELENS_GATEWAY_API_KEY"),
    audio_source: str = typer.Option("mic", "--audio-source"),
    no_camera: bool = typer.Option(False, "--no-camera"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    monitor_index: int = typer.Option(1, "--monitor"),
) -> None:
    try:
        validate_start_options(audio_source, "local", "local")
    except ValueError as error:
        fail(str(error))
    if not is_valid_session_id(session):
        fail(f"Invalid session name: {session}")
    from lecturelens.client.remote_client import run_remote_client

    settings = Settings()
    settings.audio_source = audio_source
    settings.monitor_index = monitor_index

    async def run() -> None:
        stop_event = asyncio.Event()
        await run_remote_client(
            settings, server, api_key, session,
            True, (not audio_only and not no_screen), (not audio_only and not no_camera),
            stop_event
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("Client stopped.")


if __name__ == "__main__":
    app()
