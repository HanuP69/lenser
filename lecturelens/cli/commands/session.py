"""The commands that run a lecture and close it."""

import asyncio
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Group

from lecturelens.cli.app import (
    AGENTS_GONE_GRACE_SECONDS,
    LineCounter,
    app,
    build_agent_environment,
    check_ollama,
    check_postgres,
    check_redis,
    collect_snapshot,
    console,
    copy_figure_assets,
    display_session_output_summary,
    ensure_session_exists,
    fail,
    format_low_disk_warning,
    get_free_disk_bytes,
    live_dashboard_loop,
    logger,
    module_for_agent,
    option_value,
    plan_agents,
    resolve_session,
    run_shutdown,
    validate_export_format,
    validate_start_options
)
from lecturelens.config.settings import Settings
from lecturelens.core import session_meta
from lecturelens.core.agent_runtime import (
    PAUSE_FLAG_NAME,
    SESSION_FINISHED_FLAG_NAME,
    SHUTDOWN_REQUEST_FLAG_NAME,
    clear_all_flags,
    clear_flag,
    create_redis_bus,
    flag_exists,
    read_heartbeats,
    set_flag
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.process_manager import AgentProcessManager as AgentMonitor
from lecturelens.core.validation import is_valid_course_name, is_valid_session_id

LIVE_HEARTBEAT_MAX_AGE_SECONDS: float = 15.0


def has_cloud_provider(settings: Settings) -> bool:
    for provider in ("unorouter", "openrouter", "google", "gemini"):
        for suffix in ("api_key", "api_keys"):
            val = getattr(settings, f"{provider}_{suffix}", None)
            if isinstance(val, str) and val.strip():
                return True
    return False


def _process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def agents_are_gone(session_dir: Path) -> bool:
    heartbeats = read_heartbeats(session_dir)
    if not heartbeats:
        return False
    pids = [int(p.get("pid", 0)) for p in heartbeats.values() if str(p.get("pid", "")).isdigit()]
    pids = [pid for pid in pids if pid > 0]
    return bool(pids and not any(_process_is_alive(pid) for pid in pids))


def session_has_live_agents(session_dir: Path) -> bool:
    now = time.time()
    for heartbeat in read_heartbeats(session_dir).values():
        if heartbeat.get("state") in ("stopped", "finished", "failed"):
            continue
        if now - float(heartbeat.get("time", 0.0)) <= LIVE_HEARTBEAT_MAX_AGE_SECONDS:
            return True
    return False


def _remember_window(window_handle: int, window_title: Optional[str]) -> None:
    try:
        from lecturelens.core.env_file import set_env_value
        set_env_value("LECTURELENS_CAPTURE_TARGET", "window")
        set_env_value("LECTURELENS_WINDOW_HANDLE", str(window_handle))
        if window_title:
            set_env_value("LECTURELENS_WINDOW_TITLE", window_title)
    except Exception as error:
        logger.warning("the chosen window could not be remembered: %s", error)


@app.command()
def start(
    mode: str = typer.Option("local", help="local or server"),
    audio_source: Optional[str] = typer.Option(None, "--audio-source", help="mic, system or both"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    persist: bool = typer.Option(False, "--persist", help="Enable durable Postgres persister in local mode"),
    session: Optional[str] = typer.Option(None, "--session"),
    course: Optional[str] = typer.Option(None, "--course"),
    monitor_index: Optional[int] = typer.Option(None, "--monitor"),
    window: Optional[str] = typer.Option(None, "--window", help="The window to record, by number or title."),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode", help="local, hybrid or cloud"),
    resume: bool = typer.Option(False, "--resume", help="continue an existing session"),
    video: Optional[Path] = typer.Option(None, "--video", help="Lecture video file to process directly"),
) -> None:
    settings = Settings()
    audio_source = option_value(audio_source)
    monitor_index = option_value(monitor_index)
    llm_mode = option_value(llm_mode)
    window = option_value(window)
    resolved_audio_source = audio_source if audio_source is not None else settings.audio_source
    resolved_monitor_index = monitor_index if monitor_index is not None else settings.monitor_index
    resolved_llm_mode = llm_mode if llm_mode is not None else settings.llm_mode
    video_opt = option_value(video)

    if video_opt is not None:
        from lecturelens.cli.commands.offline import offline
        offline(
            video=Path(video_opt),
            session=session or datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            course=course,
            fps=None,
            llm_mode=resolved_llm_mode
        )
        return

    try:
        validate_start_options(resolved_audio_source, resolved_llm_mode, mode)
    except ValueError as error:
        fail(str(error))
    if mode != "server" and resolved_audio_source == "system":
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

    capture_target = "screen"
    window_handle: Optional[int] = None
    window_title: Optional[str] = None
    if not no_screen and not audio_only:
        from lecturelens.cli.window_picker import resolve_capture_target
        capture_target, window_handle, window_title = resolve_capture_target(
            console,
            window_query=window,
            configured_handle=settings.window_handle,
            configured_title=settings.window_title,
        )
        if capture_target == "window" and window_handle is not None:
            settings.capture_target = "window"
            settings.window_handle = window_handle
            settings.window_title = window_title
            if window is None:
                _remember_window(window_handle, window_title)

    if not check_redis(settings):
        fail(f"Redis is not reachable at {settings.redis_url}")
    if resolved_llm_mode != "cloud" and not check_ollama(settings):
        if resolved_llm_mode == "hybrid" and has_cloud_provider(settings):
            console.print(f"[yellow]Ollama is not reachable at {settings.ollama_url}. The cloud providers carry the work.[/yellow]")
        else:
            console.print(f"[yellow]Ollama not reachable at {settings.ollama_url}. Capture continues; structuring will retry.[/yellow]")

    settings.create_session_directories(session_id)
    disk_warning = format_low_disk_warning(get_free_disk_bytes(settings.output_dir), settings.low_disk_bytes)
    if disk_warning is not None:
        console.print(f"[yellow]{disk_warning}[/yellow]")
    clear_all_flags(session_dir)
    existing_meta = session_meta.load_session_meta(session_dir)
    if resume and course is None:
        course = existing_meta.get("course_name")
    offset_seconds = session_meta.get_previous_duration_seconds(session_dir) if resume else 0.0
    clock = SessionClock(elapsed_offset_seconds=offset_seconds)
    clock.save_to_file(session_dir / "clock.json")

    should_persist = persist
    if should_persist and not check_postgres(settings):
        console.print(f"[yellow]Postgres is not reachable at {settings.postgres_url}. Persister disabled.[/yellow]")
        should_persist = False

    planned_agents = plan_agents(mode, audio_only, no_screen, enable_persister=should_persist)

    from lecturelens.cli.helpers import check_agent_dependencies
    dependency_problems = check_agent_dependencies(planned_agents)
    if dependency_problems:
        for problem in dependency_problems:
            console.print(f"[red]{problem}[/red]")
        fail("The agents cannot be started. Install what is missing and try again.")

    session_meta.begin_run(
        session_dir,
        session_id,
        course,
        {
            "mode": mode,
            "audio_source": resolved_audio_source,
            "monitor_index": resolved_monitor_index,
            "capture_target": capture_target,
            "window_title": window_title,
            "llm_mode": resolved_llm_mode,
            "audio_only": audio_only,
            "no_screen": no_screen,
            "persist": should_persist
        },
        planned_agents
    )

    environment = build_agent_environment(
        settings,
        session_id,
        resolved_audio_source,
        resolved_monitor_index,
        resolved_llm_mode,
        course,
        window_handle=window_handle,
        window_title=window_title
    )
    monitor = AgentMonitor()
    for agent_name in planned_agents:
        command = [sys.executable, "-m", module_for_agent(agent_name)]
        if agent_name != "gateway":
            command.extend(["--session", session_id, "--audio-source", resolved_audio_source])
        monitor.register_agent(agent_name, command, environment, session_dir / "logs" / f"{agent_name}.log")
        monitor.start_agent(agent_name)

    settings.llm_mode = resolved_llm_mode
    try:
        try:
            asyncio.run(live_dashboard_loop(settings, session_id, monitor, session_dir))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        run_shutdown(monitor, session_dir, planned_agents)


@app.command()
def stop(
    session: Optional[str] = typer.Option(None, "--session"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    timeout_seconds: Optional[float] = typer.Option(None, "--timeout"),
) -> None:
    settings = Settings()
    if timeout_seconds is not None and timeout_seconds <= 0:
        fail("--timeout must be greater than 0 seconds (use --no-wait to return immediately).")
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    if agents_are_gone(session_dir):
        if flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME):
            console.print(f"[green]Session {session_id} already stopped.[/green]")
            display_session_output_summary(session_dir, console)
        else:
            console.print(f"[yellow]Session {session_id} is not running.[/yellow]")
        return
    clear_flag(session_dir, SESSION_FINISHED_FLAG_NAME)
    set_flag(session_dir, SHUTDOWN_REQUEST_FLAG_NAME)
    console.print(f"Shutdown requested for session {session_id}.")
    if not wait:
        return
    deadline = (time.monotonic() + timeout_seconds) if (timeout_seconds is not None and timeout_seconds > 0) else None
    agents_gone_since: Optional[float] = None
    while deadline is None or time.monotonic() < deadline:
        if flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME):
            console.print("[green]Session stopped.[/green]")
            display_session_output_summary(session_dir, console)
            return
        if agents_are_gone(session_dir):
            if agents_gone_since is None:
                agents_gone_since = time.monotonic()
            elif time.monotonic() - agents_gone_since >= AGENTS_GONE_GRACE_SECONDS:
                console.print("[yellow]The agents of the session are not running any more.[/yellow]")
                break
        else:
            agents_gone_since = None
        time.sleep(1.0)
    console.print("[yellow]The session is still finishing. The notes are written when the final structuring pass is done.[/yellow]")
    console.print(f"[yellow]Check it with: lecturelens status --session {session_id}[/yellow]")
    raise typer.Exit(code=1)


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
        bus = create_redis_bus(settings, component_name="status")
        try:
            counter = LineCounter(session_dir / "timeline.jsonl")
            return await collect_snapshot(settings, session_id, bus, {}, counter)
        finally:
            await bus.close()

    console.print(asyncio.run(snapshot()))


@app.command()
def export(
    export_format: str = typer.Option("md", "--format", help="Export format (md, html)"),
    session: Optional[str] = typer.Option(None, "--session"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    try:
        export_format = validate_export_format(export_format)
    except ValueError as error:
        fail(str(error))
    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    target_dir = output if output is not None else session_dir / "export"
    target_dir.mkdir(parents=True, exist_ok=True)

    notes_path = session_dir / "notes.md"
    if not notes_path.exists():
        console.print("[yellow]No notes.md yet.[/yellow]")
        return

    if export_format == "html":
        from lecturelens.core.notes_html import render_notes_html
        destination = target_dir / f"{session_id}_notes.html"
        destination.write_text(
            render_notes_html(notes_path.read_text(encoding="utf-8"), title=f"Lecture Notes - {session_id}", base_dir=session_dir),
            encoding="utf-8"
        )
        console.print(f"HTML: {destination}")
        console.print("[yellow]Open it in a browser to see the formulas.[/yellow]")
        return

    destination = target_dir / f"{session_id}_notes.md"
    destination.write_bytes(notes_path.read_bytes())
    console.print(f"Markdown: {destination}")
    copied = copy_figure_assets(session_dir, target_dir)
    if copied > 0:
        console.print(f"Pictures: {copied} copied to {target_dir / 'assets'}")


@app.command()
def reprocess(
    session: str = typer.Option(..., "--session"),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode"),
) -> None:
    settings = Settings()
    settings.llm_mode = llm_mode if llm_mode is not None else settings.llm_mode
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

    redis_reachable = check_redis(settings)

    async def run() -> int:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus, redis_client = None, None
        if redis_reachable:
            bus = create_redis_bus(settings, component_name="reprocess")
            redis_client = bus.client
        try:
            router = build_router_for_task(
                "structuring", redis_client, load_router_config(settings.router_config_path),
                settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds,
                settings.local_model
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
