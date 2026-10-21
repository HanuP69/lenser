"""Choosing which window is recorded.

The whole desktop is rarely what a lecture is about. A video call, a chat
window and the editor are usually open at the same time, and recording
all of them puts the wrong content into the notes. A single window is
recorded instead, and the user is asked which one on the first run.
"""

from typing import Any, Optional

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.window_picker")


def describe_window(window: Any) -> str:
    """A line that says which window it is, without any doubt.

    The number of a window is only right for the list that was shown, so
    the title, the size and the place of the window are shown as well.
    """
    return (
        f"{window.title!r} ({window.width}x{window.height} at "
        f"({window.left}, {window.top}))"
    )


def _can_ask() -> bool:
    """Whether there is somebody to answer the question.

    A script, a service or a test has nobody, and the prompt would block
    forever.
    """
    import sys

    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def format_window_table(windows: list[Any], console: Any) -> None:
    """Show the windows that can be recorded."""
    from rich.table import Table

    table = Table(title="Windows that can be recorded")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Size", justify="right")
    table.add_column("State")
    for index, window in enumerate(windows, start=1):
        state = "minimized" if window.is_minimized else "visible"
        table.add_row(str(index), window.title, f"{window.width}x{window.height}", state)
    console.print(table)


def prompt_for_window(
    console: Any,
    query: Optional[str] = None,
    allow_screen: bool = True,
) -> Optional[Any]:
    """Ask the user which window is recorded.

    Returns the window, or None when the whole screen is recorded.
    """
    from lecturelens.core.window_capture import (
        WindowCaptureUnavailable,
        find_window,
        list_windows,
    )

    try:
        windows = list_windows()
    except WindowCaptureUnavailable as error:
        console.print(f"[yellow]The windows of the desktop are not available: {error}[/yellow]")
        return None
    except Exception as error:
        console.print(f"[yellow]The windows of the desktop could not be listed: {error}[/yellow]")
        return None

    if not windows:
        console.print("[yellow]No window is open to record.[/yellow]")
        return None

    if query:
        window = find_window(query, windows)
        if window is None:
            console.print(f"[red]No window matches {query!r}.[/red]")
        return window

    format_window_table(windows, console)
    if not allow_screen:
        console.print("\nEnter the number of the window to record.")
    else:
        console.print(
            "\nEnter the number of the window to record, "
            "or press enter to record the whole screen."
        )
    try:
        answer = console.input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not answer:
        return None
    if answer.isdigit():
        index = int(answer) - 1
        if 0 <= index < len(windows):
            return windows[index]
        console.print(f"[red]There is no window number {answer}.[/red]")
        return None
    window = find_window(answer, windows)
    if window is None:
        console.print(f"[red]No window matches {answer!r}.[/red]")
    return window


def resolve_capture_target(
    console: Any,
    window_query: Optional[str] = None,
    configured_handle: Optional[int] = None,
    configured_title: Optional[str] = None,
    ask: bool = True,
) -> tuple[str, Optional[int], Optional[str]]:
    """Work out what is recorded.

    Returns the capture target, the window handle and the window title.
    """
    from lecturelens.core.window_capture import (
        WindowCaptureUnavailable,
        find_window,
        is_window_alive,
        list_windows,
    )

    if window_query:
        try:
            window = find_window(window_query)
        except WindowCaptureUnavailable as error:
            console.print(f"[red]{error}[/red]")
            return "screen", None, None
        if window is None:
            console.print(f"[red]No window matches {window_query!r}.[/red]")
            return "screen", None, None
        console.print(f"[green]Recording the window[/green] {describe_window(window)}")
        return "window", window.handle, window.title

    if configured_handle and is_window_alive(configured_handle):
        title = configured_title
        if not title:
            try:
                for window in list_windows(include_minimized=True):
                    if window.handle == configured_handle:
                        title = window.title
                        break
            except Exception:
                title = None
        return "window", configured_handle, title

    if not ask:
        return "screen", None, None

    # The question is only asked when there is somebody to answer it. In a
    # script or a service the prompt would block forever.
    if not _can_ask():
        return "screen", None, None

    window = prompt_for_window(console)
    if window is None:
        console.print("[yellow]Recording the whole screen.[/yellow]")
        return "screen", None, None
    console.print(f"[green]Recording the window[/green] {describe_window(window)}")
    return "window", window.handle, window.title
