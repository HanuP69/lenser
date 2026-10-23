"""Finding a single window on the desktop and capturing it.

The screen agent can capture the whole display or one window. A lecture
recording needs one window, because the rest of the desktop belongs to
other work. A video call that happens to be on the same display would
put the faces of the participants into the notes instead of the slides.

The window is captured through ``PrintWindow``, which asks the window to
draw itself. That works when the window is behind other windows, which a
plain screenshot of the screen area cannot do. When a window refuses to
draw itself, the screen area is used instead.
"""

import ctypes
import sys
from dataclasses import dataclass
from typing import Optional

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.window_capture")

IS_WINDOWS = sys.platform == "win32"

# A window smaller than this is a tool tip, a task bar button or some other
# piece of chrome that is never the thing the user wants to record.
MINIMUM_WINDOW_WIDTH = 240
MINIMUM_WINDOW_HEIGHT = 180

# The shell windows. They are always open and are never what a lecture is
# about, so they are not offered as a choice.
IGNORED_WINDOW_TITLES = {
    "program manager",
    "windows input experience",
    "nvidia geforce overlay",
    "windows shell experience host",
    "microsoft text input application",
    "default IME",
}

# PW_RENDERFULLCONTENT, so a window that is drawn by the desktop window
# manager is rendered and not left black.
PRINT_WINDOW_RENDER_FULL_CONTENT = 0x00000002


@dataclass(frozen=True)
class WindowInfo:
    """One window that is on the desktop."""

    handle: int
    title: str
    left: int
    top: int
    width: int
    height: int
    process_id: int
    is_minimized: bool

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class WindowCaptureUnavailable(RuntimeError):
    """The windows of the desktop cannot be listed or captured."""


def _user32():
    if not IS_WINDOWS:
        raise WindowCaptureUnavailable("window capture only works on Windows")
    return ctypes.windll.user32


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


def list_windows(include_minimized: bool = False) -> list[WindowInfo]:
    """Every window on the desktop that could be recorded.

    The list is sorted by title, so the numbering does not change when a
    window is closed.
    """
    if not IS_WINDOWS:
        raise WindowCaptureUnavailable("window capture only works on Windows")

    user32 = _user32()
    windows: list[WindowInfo] = []

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
    )

    def collect(handle: int, _lparam: int) -> bool:
        if not user32.IsWindow(handle):
            return True
        if not include_minimized and not user32.IsWindowVisible(handle):
            return True
        title_buffer = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(handle, title_buffer, 512)
        title = title_buffer.value.strip()
        if not title:
            return True
        if title.lower() in IGNORED_WINDOW_TITLES:
            return True
        rect = _Rect()
        if not user32.GetWindowRect(handle, ctypes.byref(rect)):
            return True
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < MINIMUM_WINDOW_WIDTH or height < MINIMUM_WINDOW_HEIGHT:
            return True
        process_id = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
        is_minimized = bool(user32.IsIconic(handle))
        windows.append(
            WindowInfo(
                handle=int(handle),
                title=title,
                left=int(rect.left),
                top=int(rect.top),
                width=int(width),
                height=int(height),
                process_id=int(process_id.value),
                is_minimized=is_minimized,
            )
        )
        return True

    user32.EnumWindows(callback_type(collect), 0)
    windows.sort(key=lambda item: (item.title.lower(), item.handle))
    return windows


def is_window_alive(handle: int) -> bool:
    """Whether the window with this handle is still on the desktop."""
    if not IS_WINDOWS:
        return False
    try:
        return bool(_user32().IsWindow(int(handle)))
    except Exception:
        return False


def find_window(
    query: str,
    windows: Optional[list[WindowInfo]] = None,
    include_minimized: bool = False
) -> Optional[WindowInfo]:
    """Find the window the user meant.

    The query is either the number from the listing or part of the title.

    The number is looked for in the same list that the listing shows, so
    the number the user saw is the window that is found. The list used
    to have the minimized windows in it as well, so the numbers were off
    by however many minimized windows there were, and the window that
    was recorded was not the one on the screen.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return None
    available = windows if windows is not None else list_windows()

    if cleaned.isdigit():
        index = int(cleaned) - 1
        if 0 <= index < len(available):
            return available[index]
        # The window may have been closed between the listing and the
        # start. A minimized window is still a window the user can mean.
        if include_minimized:
            with_minimized = list_windows(include_minimized=True)
            if 0 <= index < len(with_minimized):
                return with_minimized[index]
        return None

    lowered = cleaned.lower()
    exact = [item for item in available if item.title.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    contains = [item for item in available if lowered in item.title.lower()]
    if len(contains) == 1:
        return contains[0]
    if contains:
        # The first one is the shortest title, which is usually the most
        # specific one.
        contains.sort(key=lambda item: (len(item.title), item.title.lower()))
        return contains[0]
    return None


def capture_window_bgra(handle: int) -> tuple[bytes, int, int]:
    """Capture one window and return its pixels as BGRA bytes.

    The window is asked to draw itself, so it is captured even when it is
    behind other windows. Windows that refuse to draw itself fall back to
    the screen area that the window covers.
    """
    if not IS_WINDOWS:
        raise WindowCaptureUnavailable("window capture only works on Windows")

    user32 = _user32()
    gdi32 = ctypes.windll.gdi32
    handle = int(handle)

    rect = _Rect()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise WindowCaptureUnavailable(f"the window {handle} is gone")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        raise WindowCaptureUnavailable(f"the window {handle} has no size")

    window_dc = user32.GetWindowDC(handle)
    if not window_dc:
        raise WindowCaptureUnavailable(f"the window {handle} has no device context")
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    if not memory_dc:
        user32.ReleaseDC(handle, window_dc)
        raise WindowCaptureUnavailable(f"the window {handle} could not be prepared")

    header = _BitmapInfoHeader()
    header.biSize = ctypes.sizeof(_BitmapInfoHeader)
    header.biWidth = width
    # A negative height puts the first row at the top of the buffer.
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0

    bits = ctypes.c_void_p()
    bitmap = gdi32.CreateDIBSection(
        memory_dc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0
    )
    if not bitmap:
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(handle, window_dc)
        raise WindowCaptureUnavailable(f"the window {handle} could not be prepared")

    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        drawn = user32.PrintWindow(handle, memory_dc, PRINT_WINDOW_RENDER_FULL_CONTENT)
        if not drawn:
            drawn = user32.PrintWindow(handle, memory_dc, 0)
        if not drawn:
            raise WindowCaptureUnavailable(f"the window {handle} refused to draw itself")
        raw = ctypes.string_at(bits, width * height * 4)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(handle, window_dc)

    if not _has_visible_pixels(raw):
        # Some windows draw themselves only when they are on screen. The
        # screen area is the next best thing.
        logger.warning(
            "the window %s drew an empty frame, using the screen area instead", handle
        )
        return _capture_screen_region(rect.left, rect.top, width, height)
    return raw, width, height


def _has_visible_pixels(raw: bytes) -> bool:
    """Whether the frame has anything in it.

    A window that refuses to draw itself leaves a black frame, which is
    not useful as a slide.
    """
    step = 4 * 64
    for offset in range(0, len(raw), step):
        pixel = raw[offset:offset + 3]
        if len(pixel) < 3:
            continue
        if pixel != b"\x00\x00\x00":
            return True
    return False


def _capture_screen_region(
    left: int, top: int, width: int, height: int
) -> tuple[bytes, int, int]:
    """Capture the part of the screen that the window covers."""
    import mss

    with mss.mss() as screen_capturer:
        region = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        screenshot = screen_capturer.grab(region)
        return bytes(screenshot.bgra), int(screenshot.width), int(screenshot.height)
