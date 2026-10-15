import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from game_agent.capture import ScreenCapture
from game_agent.geometry import Region


class FakeShot:
    rgb = b"\x00\x00\x00" * 4
    size = (2, 2)


class FakeMSS:
    monitors = [{"left": 0, "top": 0, "width": 10, "height": 10}]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def grab(self, monitor):
        self.monitor = monitor
        return FakeShot()


class CaptureTests(unittest.TestCase):
    def test_save_capture_uses_requested_region(self):
        fake = FakeMSS()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shot.png"
            def fake_png(_rgb, _size, output):
                Path(output).write_bytes(b"png")
            with patch("mss.mss", return_value=fake), patch("mss.tools.to_png", fake_png):
                result = ScreenCapture().save(output, Region(1, 2, 3, 4))
            self.assertEqual(output, result)
            self.assertEqual({"left": 1, "top": 2, "width": 3, "height": 4}, fake.monitor)
            self.assertTrue(output.exists())

    def test_save_capture_preserves_negative_screen_coordinates(self):
        fake = FakeMSS()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "left-monitor-board.png"
            with patch("mss.mss", return_value=fake), patch("mss.tools.to_png"):
                ScreenCapture().save(output, Region(-1200, 150, 640, 640))
        self.assertEqual(
            {"left": -1200, "top": 150, "width": 640, "height": 640},
            fake.monitor,
        )


if __name__ == "__main__":
    unittest.main()
