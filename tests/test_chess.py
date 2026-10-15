import tempfile
import unittest
from pathlib import Path

from game_agent.chess import ChessCalibration, Orientation
from game_agent.geometry import Region


class ChessCalibrationTests(unittest.TestCase):
    def test_white_bottom_coordinates(self):
        calibration = ChessCalibration(Region(100, 200, 800, 800))
        self.assertEqual((150, 950), calibration.square_center("a1"))
        self.assertEqual((550, 850), calibration.square_center("e2"))
        self.assertEqual((550, 650), calibration.square_center("e4"))
        self.assertEqual((850, 250), calibration.square_center("h8"))

    def test_black_bottom_coordinates(self):
        calibration = ChessCalibration(
            Region(100, 200, 800, 800), Orientation.BLACK_BOTTOM
        )
        self.assertEqual((850, 250), calibration.square_center("a1"))
        self.assertEqual((150, 950), calibration.square_center("h8"))

    def test_rejects_invalid_square(self):
        with self.assertRaises(ValueError):
            ChessCalibration(Region(0, 0, 80, 80)).square_center("e9")

    def test_round_trip_config(self):
        expected = ChessCalibration(Region(1, 2, 80, 88), Orientation.BLACK_BOTTOM)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            expected.save(path)
            self.assertEqual(expected, ChessCalibration.load(path))


if __name__ == "__main__":
    unittest.main()
