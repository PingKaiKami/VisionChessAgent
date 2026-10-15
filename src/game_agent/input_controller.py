from __future__ import annotations

import atexit
import time
from typing import Any

from .safety import SafetyGuard


class PointerInterferenceError(RuntimeError):
    pass


class PyAutoGUIBackend:
    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True
        self._api = pyautogui

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)


class InputController:
    """Safe mouse/keyboard controller. Real input is disabled by default."""

    def __init__(
        self,
        safety: SafetyGuard,
        *,
        enable_live: bool = False,
        backend: Any | None = None,
        sleeper=time.sleep,
    ) -> None:
        self.safety = safety
        self.enable_live = enable_live
        self.backend = backend if backend is not None else (
            PyAutoGUIBackend() if enable_live else None
        )
        self.sleeper = sleeper
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        self.actions: list[tuple[Any, ...]] = []
        atexit.register(self.release_all)

    @property
    def dry_run(self) -> bool:
        return not self.enable_live

    def pointer_position(self) -> tuple[int, int] | None:
        if not self.enable_live:
            return None
        point = self.backend.position()
        return int(point[0]), int(point[1])

    def pointer_near(self, x: int, y: int, tolerance: int = 4) -> bool:
        position = self.pointer_position()
        return position is None or (
            abs(position[0] - x) <= tolerance and abs(position[1] - y) <= tolerance
        )

    def _record_or_call(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.actions.append((name, *args, kwargs))
        if self.enable_live:
            getattr(self.backend, name)(*args, **kwargs)

    def move_to(self, x: int, y: int, duration: float = 0.2) -> None:
        self.safety.verify((x, y))
        try:
            self._record_or_call("moveTo", x, y, duration=max(0.0, duration))
            if self.enable_live and not self.pointer_near(x, y):
                raise PointerInterferenceError(
                    f"Pointer did not reach expected position {(x, y)}"
                )
        except BaseException:
            self.release_all()
            raise

    def click(self, x: int, y: int, *, button: str = "left", duration: float = 0.2) -> None:
        self.move_to(x, y, duration)
        try:
            self.safety.verify((x, y))
            self._record_or_call("click", button=button)
        except BaseException:
            self.release_all()
            raise

    def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        self.move_to(*start, duration=0.0)
        self.safety.verify(end)
        self.mouse_down(button)
        try:
            self.move_to(*end, duration=duration)
        finally:
            self.mouse_up(button)

    def mouse_down(self, button: str = "left") -> None:
        self.safety.verify()
        self.held_buttons.add(button)
        try:
            self._record_or_call("mouseDown", button=button)
        except BaseException:
            self.release_all()
            raise

    def mouse_up(self, button: str = "left") -> None:
        if button in self.held_buttons:
            self._record_or_call("mouseUp", button=button)
            self.held_buttons.discard(button)

    def key_down(self, key: str) -> None:
        self.safety.verify()
        self.held_keys.add(key)
        try:
            self._record_or_call("keyDown", key)
        except BaseException:
            self.release_all()
            raise

    def key_up(self, key: str) -> None:
        if key in self.held_keys:
            self._record_or_call("keyUp", key)
            self.held_keys.discard(key)

    def press(self, key: str, duration: float = 0.05) -> None:
        self.key_down(key)
        try:
            self.sleeper(max(0.0, duration))
        finally:
            self.key_up(key)

    def release_all(self) -> None:
        for key in tuple(self.held_keys):
            try:
                self._record_or_call("keyUp", key)
            except Exception:
                pass
            finally:
                self.held_keys.discard(key)
        for button in tuple(self.held_buttons):
            try:
                self._record_or_call("mouseUp", button=button)
            except Exception:
                pass
            finally:
                self.held_buttons.discard(button)

    def __enter__(self) -> "InputController":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release_all()
