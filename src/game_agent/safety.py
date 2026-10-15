from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .geometry import Region
from .window import WindowInfo, get_foreground_window


class SafetyError(RuntimeError):
    pass


@dataclass
class SafetyGuard:
    expected_title: str
    allowed_region: Region
    window_provider: Callable[[], WindowInfo] = get_foreground_window
    expected_handle: int | None = None

    def verify(self, point: tuple[int, int] | None = None) -> WindowInfo:
        window = self.window_provider()
        if self.expected_title.casefold() not in window.title.casefold():
            raise SafetyError(
                f"Foreground window {window.title!r} does not match {self.expected_title!r}"
            )
        if self.expected_handle is None:
            self.expected_handle = window.handle
        elif window.handle != self.expected_handle:
            raise SafetyError(
                f"Foreground window handle {window.handle} does not match "
                f"the original Chess window {self.expected_handle}"
            )
        if not window.region.contains_region(self.allowed_region):
            raise SafetyError("Allowed action region is outside the foreground window")
        if point is not None and not self.allowed_region.contains_point(*point):
            raise SafetyError(f"Action point {point} is outside the allowed region")
        return window
