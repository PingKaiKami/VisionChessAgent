from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from .geometry import Region


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    region: Region


def get_foreground_window() -> WindowInfo:
    if sys.platform != "win32":
        raise RuntimeError("Window inspection is only supported on Windows")

    enable_dpi_awareness()
    user32 = ctypes.windll.user32
    handle = user32.GetForegroundWindow()
    if not handle:
        raise RuntimeError("No foreground window")

    length = user32.GetWindowTextLengthW(handle)
    title_buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, title_buffer, length + 1)

    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise RuntimeError("Unable to read foreground window bounds")

    return WindowInfo(
        handle=handle,
        title=title_buffer.value,
        region=Region(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
    )
