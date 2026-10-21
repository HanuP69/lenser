"""The commands that choose what is recorded."""

from typing import Optional

import typer

from lecturelens.cli.app import app, check_redis, console, fail
from lecturelens.config.settings import Settings


@app.command("doctor")
def doctor_command() -> None:
    """Check whether this machine can run a lecture."""
    from lecturelens.cli.helpers import check_agent_dependencies, check_dependency_versions

    settings = Settings()
    console.print("[bold]LectureLens doctor[/bold]\n")

    console.print(f"output folder : {settings.output_dir}")
    console.print(f"router file   : {settings.router_config_path}")

    console.print("\n[bold]The packages[/bold]")
    version_problems = check_dependency_versions()
    if version_problems:
        for problem in version_problems:
            console.print(f"  [red]{problem}[/red]")
    else:
        console.print("  [green]The installed versions fit each other.[/green]")

    console.print("\n[bold]The agents[/bold]")
    agent_problems = check_agent_dependencies(
        ["audio_agent", "screen_agent", "sync_agent", "structuring_agent"]
    )
    if agent_problems:
        for problem in agent_problems:
            console.print(f"  [red]{problem}[/red]")
    else:
        console.print("  [green]Every agent can be started.[/green]")

    console.print("\n[bold]The services[/bold]")
    if check_redis(settings):
        console.print(f"  [green]Redis is running[/green] at {settings.redis_url}")
    else:
        console.print(f"  [red]Redis is not reachable[/red] at {settings.redis_url}")

    console.print("\n[bold]The sound[/bold]")
    from lecturelens.agents.audio_agent import find_loopback_microphone

    device, error = find_loopback_microphone()
    if device is not None:
        console.print(f"  [green]The system audio can be recorded[/green] through {device}")
    else:
        console.print(f"  [yellow]The system audio cannot be recorded[/yellow]: {error}")

    console.print("\n[bold]The windows[/bold]")
    from lecturelens.core.window_capture import WindowCaptureUnavailable, list_windows

    try:
        windows = list_windows()
        console.print(f"  [green]{len(windows)} windows[/green] can be recorded")
    except WindowCaptureUnavailable as error:
        console.print(f"  [yellow]The windows cannot be listed[/yellow]: {error}")


@app.command("windows")
def list_windows_command(
    set_window: Optional[str] = typer.Option(
        None,
        "--set",
        help="Remember this window (by number or title) as the one to record."
    ),
    clear: bool = typer.Option(False, "--clear", help="Forget the window and record the whole screen again."),
) -> None:
    """Show the windows that can be recorded."""
    from lecturelens.cli.window_picker import format_window_table
    from lecturelens.core.env_file import env_file_location, remove_env_value, set_env_value
    from lecturelens.core.window_capture import (
        WindowCaptureUnavailable,
        find_window,
        list_windows,
    )

    try:
        windows = list_windows()
    except WindowCaptureUnavailable as error:
        fail(str(error))
        return
    except Exception as error:
        fail(f"The windows of the desktop could not be listed: {error}")
        return

    if not windows:
        console.print("[yellow]No window is open to record.[/yellow]")
        return

    format_window_table(windows, console)

    if clear:
        remove_env_value("LECTURELENS_WINDOW_HANDLE")
        remove_env_value("LECTURELENS_WINDOW_TITLE")
        remove_env_value("LECTURELENS_CAPTURE_TARGET")
        console.print(f"[green]The whole screen is recorded again.[/green] ({env_file_location()})")
        return

    if set_window:
        window = find_window(set_window, windows)
        if window is None:
            fail(f"No window matches {set_window!r}.")
            return
        set_env_value("LECTURELENS_CAPTURE_TARGET", "window")
        set_env_value("LECTURELENS_WINDOW_HANDLE", str(window.handle))
        set_env_value("LECTURELENS_WINDOW_TITLE", window.title)
        console.print(
            f"[green]The window[/green] {window.title!r} [green]is recorded from now on.[/green]"
            f" ({env_file_location()})"
        )
