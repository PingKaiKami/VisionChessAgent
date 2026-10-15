from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .geometry import Region
from .window import enable_dpi_awareness


@dataclass(frozen=True)
class ScreenFrame:
    image: np.ndarray
    region: Region


class ScreenCapture:
    """Capture pixels from the desktop only; no game internals are accessed."""

    def capture(self, region: Region | None = None):
        return self.capture_frame(region).image

    def capture_frame(self, region: Region | None = None) -> ScreenFrame:
        import mss

        enable_dpi_awareness()
        with mss.mss() as grabber:
            monitor = region.as_mss_monitor() if region else grabber.monitors[0]
            shot = grabber.grab(monitor)
            captured_region = Region(
                int(monitor["left"]), int(monitor["top"]),
                int(monitor["width"]), int(monitor["height"]),
            )
            image = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
            return ScreenFrame(image, captured_region)

    def save(self, path: str | Path, region: Region | None = None) -> Path:
        import mss
        from mss.tools import to_png

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        enable_dpi_awareness()
        with mss.mss() as grabber:
            monitor = region.as_mss_monitor() if region else grabber.monitors[0]
            shot = grabber.grab(monitor)
            to_png(shot.rgb, shot.size, output=str(output))
        return output
