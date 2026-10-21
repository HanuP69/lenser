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

# How old a heartbeat may be before the agent is taken to be gone.
LIVE_HEARTBEAT_MAX_AGE_SECONDS: float = 15.0


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
    window: Optional[str] = typer.Option(
        None,
        "--window",
        help="The window to record, by number or by title. Without it the whole screen is recorded."
    ),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode", help="local, hybrid or cloud"),
    resume: bool = typer.Option(False, "--resume", help="continue an existing session"),
) -> None:
    settings = Settings()

    # The option defaults used to be exported to the agents unconditionally,
    # so a value configured in .env (for example LECTURELENS_AUDIO_SOURCE)
    # was always overridden by the CLI default. The options now default to
    # None and fall back to the configured settings.
    audio_source = option_value(audio_source)
    monitor_index = option_value(monitor_index)
    llm_mode = option_value(llm_mode)
    window = option_value(window)
    resolved_audio_source = audio_source if audio_source is not None else settings.audio_source
    resolved_monitor_index = monitor_index if monitor_index is not None else settings.monitor_index
    resolved_llm_mode = llm_mode if llm_mode is not None else settings.llm_mode

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
        # The check is done before the window is chosen, so a name that is
        # already taken does not throw the choice of the window away.
        fail(f"Session '{session_id}' already exists. Use --resume to continue it or pick another name.")

    # The window that is recorded is chosen before the session starts, so
    # the screen agent records the right thing from the first frame. The
    # question is only asked when nothing has been chosen yet.
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
                # The window was chosen by answering the question, so it is
                # remembered. The question is only asked once.
                _remember_window(window_handle, window_title)

    if not check_redis(settings):
        fail(f"Redis is not reachable at {settings.redis_url}")
    if resolved_llm_mode != "cloud" and not check_ollama(settings):
        console.print(
            f"[yellow]Ollama not reachable at {settings.ollama_url}. "
            "Capture continues; structuring will retry when it returns.[/yellow]"
        )

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
        console.print(
            f"[yellow]Postgres is not reachable at {settings.postgres_url}. "
            "Persister disabled; session will run without Postgres storage.[/yellow]"
        )
        should_persist = False

    planned_agents = plan_agents(mode, audio_only, no_screen, enable_persister=should_persist)

    # An agent that cannot import what it needs dies at once and the
    # reason is only in its log. The capture goes on and everything is
    # recorded, but the notes are never written, so the check is done
    # before the session starts.
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
            "persist": persist
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
        monitor.register_agent(
            agent_name,
            command,
            environment,
            session_dir / "logs" / f"{agent_name}.log"
        )
        monitor.start_agent(agent_name)

    settings.llm_mode = resolved_llm_mode
    try:
        try:
            asyncio.run(live_dashboard_loop(settings, session_id, monitor, session_dir))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        # The shutdown sequence is the only path that writes the stop flags
        # and terminates the child processes. It must run for *every* exit,
        # otherwise a failure in the dashboard loop leaves the capture agents
        # running with no stop flag.
        run_shutdown(monitor, session_dir, planned_agents)


def session_has_live_agents(session_dir: Path) -> bool:
    now = time.time()
    for heartbeat in read_heartbeats(session_dir).values():
        state = str(heartbeat.get("state", ""))
        if state in ("stopped", "finished", "failed"):
            continue
        if now - float(heartbeat.get("time", 0.0)) <= LIVE_HEARTBEAT_MAX_AGE_SECONDS:
            return True
    return False


def _remember_window(window_handle: int, window_title: Optional[str]) -> None:
    """Keep the window that was chosen, so the question is asked once."""
    from lecturelens.core.env_file import set_env_value

    try:
        set_env_value("LECTURELENS_CAPTURE_TARGET", "window")
        set_env_value("LECTURELENS_WINDOW_HANDLE", str(window_handle))
        if window_title:
            set_env_value("LECTURELENS_WINDOW_TITLE", window_title)
    except Exception as error:
        logger.warning("the chosen window could not be remembered: %s", error)


def agents_are_gone(session_dir: Path) -> bool:
    """Whether every agent of the session has stopped.

    The heartbeat files hold the process id of each agent. When none of
    them is running any more the session cannot make any more progress,
    so waiting for it is pointless. This happens when the process that
    owns the session dies.
    """
    heartbeats = read_heartbeats(session_dir)
    if not heartbeats:
        return False
    pids: list[int] = []
    for payload in heartbeats.values():
        try:
            pids.append(int(payload.get("pid", 0)))
        except (TypeError, ValueError):
            continue
    pids = [pid for pid in pids if pid > 0]
    if not pids:
        return False
    return not any(_process_is_alive(pid) for pid in pids)


def _process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == 259
                return True
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


@app.command()
def stop(
    session: Optional[str] = typer.Option(None, "--session"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    timeout_seconds: float = typer.Option(300.0, "--timeout"),
) -> None:
    settings = Settings()
    if timeout_seconds <= 0:
        fail("--timeout must be greater than 0 seconds (use --no-wait to return immediately).")
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    clear_flag(session_dir, SESSION_FINISHED_FLAG_NAME)
    set_flag(session_dir, SHUTDOWN_REQUEST_FLAG_NAME)
    console.print(f"Shutdown requested for session {session_id}.")
    if not wait:
        return
    deadline = time.monotonic() + timeout_seconds
    # The agents are stopped by the process that owns the session, and the
    # flag is set after that. So the agents being gone is the normal way
    # a session ends, and the stop waits a little for the flag instead of
    # giving up at once.
    agents_gone_since: Optional[float] = None
    while time.monotonic() < deadline:
        if flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME):
            console.print("[green]Session stopped.[/green]")
            display_session_output_summary(session_dir, console)
            return
        if agents_are_gone(session_dir):
            if agents_gone_since is None:
                agents_gone_since = time.monotonic()
            elif time.monotonic() - agents_gone_since >= AGENTS_GONE_GRACE_SECONDS:
                console.print(
                    "[yellow]The agents of the session are not running any more, "
                    "so the session cannot be finished.[/yellow]"
                )
                break
        else:
            agents_gone_since = None
        time.sleep(1.0)
    # The session is not killed by the timeout. The agents keep running and
    # the notes are written when the final structuring pass is done, which
    # takes a while when the provider is on a free tier.
    console.print(
        "[yellow]The session is still finishing. The notes are written when the "
        "final structuring pass is done.[/yellow]"
    )
    console.print(
        f"[yellow]Check it with: lecturelens status --session {session_id}[/yellow]"
    )
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
        notes_text = notes_path.read_text(encoding="utf-8")
        destination.write_text(
            render_notes_html(notes_text, title=f"Lecture Notes - {session_id}"),
            encoding="utf-8"
        )
        console.print(f"HTML: {destination}")
        console.print("[yellow]Open it in a browser to see the formulas.[/yellow]")
        return

    destination = target_dir / f"{session_id}_notes.md"
    destination.write_bytes(notes_path.read_bytes())
    console.print(f"Markdown: {destination}")


@app.command()
def reprocess(
    session: str = typer.Option(..., "--session"),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode"),
) -> None:
    settings = Settings()
    # The option default used to be exported unconditionally, so the value
    # configured in .env was always overridden.
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

    # Redis reachability must be probed *outside* the event loop: check_redis()
    # spins up its own loop via asyncio.run(), which cannot be called from
    # inside a running loop.
    redis_reachable = check_redis(settings)

    async def run() -> int:
        from lecturelens.agents.structuring_agent import StructuringAgent
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus = None
        redis_client = None
        if redis_reachable:
            bus = create_redis_bus(settings, component_name="reprocess")
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
