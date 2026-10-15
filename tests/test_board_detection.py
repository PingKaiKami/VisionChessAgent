import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from game_agent.board_detection import BoardDetectionError, BoardDetector
from game_agent.capture import ScreenFrame
from game_agent.chess import ChessCalibration, Orientation
from game_agent.chess_perception import INITIAL_POSITION, TemplateLibrary, screen_cell_square
from game_agent.geometry import Region


SIZE = 32


def board_image(orientation=Orientation.WHITE_BOTTOM, side=SIZE * 8):
    image = np.zeros((side, side, 3), dtype=np.uint8)
    for row in range(8):
        y0, y1 = round(row * side / 8), round((row + 1) * side / 8)
        for column in range(8):
            x0, x1 = round(column * side / 8), round((column + 1) * side / 8)
            square = screen_cell_square(row, column, orientation)
            shade = 190 if (ord(square[0]) - ord("a") + int(square[1])) % 2 else 90
            tile = image[y0:y1, x0:x1]
            tile[:] = shade
            symbol = INITIAL_POSITION.get(square, ".")
            if symbol != ".":
                color = (245, 245, 245) if symbol.isupper() else (20, 20, 20)
                cell = (tile.shape[0] + tile.shape[1]) / 2
                scale = 0.75 * cell / SIZE
                thickness = max(1, round(2 * cell / SIZE))
                (width, height), _ = cv2.getTextSize(
                    symbol, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
                )
                origin = ((tile.shape[1] - width) // 2, (tile.shape[0] + height) // 2)
                outline = (220, 220, 220) if not symbol.isupper() else (45, 45, 45)
                cv2.putText(
                    tile, symbol, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv2.LINE_AA,
                )
                cv2.putText(
                    tile, symbol,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
                )
    return image


def templates():
    calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
    return TemplateLibrary.learn(board_image(), calibration)


class BoardDetectionTests(unittest.TestCase):
    def test_locates_board_and_converts_to_negative_absolute_coordinates(self):
        desktop = np.full((500, 900, 3), 35, dtype=np.uint8)
        desktop[80:336, 140:396] = board_image()
        frame = ScreenFrame(desktop, Region(-1920, 0, 900, 500))

        result = BoardDetector().detect(
            frame,
            templates(),
            expected_placement="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        )

        self.assertEqual(Region(-1780, 80, 256, 256), result.calibration.board)
        self.assertEqual(Orientation.WHITE_BOTTOM, result.calibration.orientation)
        self.assertGreater(result.confidence, 0.9)

    def test_relocates_when_board_position_changes(self):
        detector = BoardDetector()
        library = templates()
        expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        first = np.full((500, 900, 3), 35, dtype=np.uint8)
        second = first.copy()
        first[40:296, 50:306] = board_image()
        second[180:436, 500:756] = board_image()

        initial = detector.detect(ScreenFrame(first, Region(0, 0, 900, 500)), library, expected_placement=expected)
        moved = detector.detect(ScreenFrame(second, Region(0, 0, 900, 500)), library, expected_placement=expected)

        self.assertNotEqual(initial.calibration.board, moved.calibration.board)
        self.assertEqual(Region(500, 180, 256, 256), moved.calibration.board)

    def test_refines_offset_contour_without_fixed_board_size(self):
        desktop = np.full((500, 900, 3), 35, dtype=np.uint8)
        desktop[80:336, 140:396] = board_image()
        frame = ScreenFrame(desktop, Region(-1920, 0, 900, 500))
        with tempfile.TemporaryDirectory() as directory:
            detector = BoardDetector(debug_directory=directory)
            with patch.object(
                detector,
                "_candidate_regions",
                return_value=[Region(139, 75, 259, 259)],
            ):
                result = detector.detect(
                    frame,
                    templates(),
                    expected_placement="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                )
            self.assertTrue((Path(directory) / "detected-board.png").is_file())
            self.assertTrue((Path(directory) / "detected-grid.png").is_file())
            self.assertIn("region=Region", (Path(directory) / "detection.txt").read_text())

        self.assertEqual(Region(-1780, 80, 256, 256), result.calibration.board)
        self.assertFalse(result.state.unknown_squares)

    def test_detects_independently_rendered_windowed_board(self):
        fullscreen = board_image(side=1135)
        library = TemplateLibrary.learn(
            fullscreen, ChessCalibration(Region(0, 0, 1135, 1135))
        )
        desktop = np.full((850, 1000, 3), 35, dtype=np.uint8)
        desktop[80:786, 140:846] = board_image(side=706)
        frame = ScreenFrame(desktop, Region(-1920, 0, 1000, 850))
        detector = BoardDetector()

        with patch.object(
            detector,
            "_candidate_regions",
            return_value=[Region(139, 75, 710, 710)],
        ):
            result = detector.detect(
                frame,
                library,
                expected_placement="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                orientation=Orientation.WHITE_BOTTOM,
            )

        self.assertEqual(Region(-1780, 80, 706, 706), result.calibration.board)
        self.assertFalse(result.state.unknown_squares)

    def test_detection_failure_stops_safely(self):
        blank = ScreenFrame(np.full((400, 600, 3), 40, dtype=np.uint8), Region(0, 0, 600, 400))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BoardDetectionError):
                BoardDetector(debug_directory=directory).detect(
                    blank,
                    templates(),
                    expected_placement="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                )
            report = (Path(directory) / "attempt-0001.txt").read_text()
            self.assertIn("reason=No reliable Chess board candidate found", report)
            self.assertTrue((Path(directory) / "attempt-0001-frame.png").is_file())
            self.assertTrue((Path(directory) / "attempt-0001-candidates.png").is_file())

    def test_diagnostics_compare_first_and_subsequent_detection(self):
        desktop = np.full((500, 900, 3), 35, dtype=np.uint8)
        desktop[80:336, 140:396] = board_image()
        frame = ScreenFrame(desktop, Region(0, 0, 900, 500))
        with tempfile.TemporaryDirectory() as directory:
            detector = BoardDetector(debug_directory=directory)
            for _ in range(2):
                detector.detect(
                    frame,
                    templates(),
                    expected_placement="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                )
            report = (Path(directory) / "attempt-0002.txt").read_text()
            self.assertIn("confidence_delta=", report)
            self.assertIn("same_fen=True", report)
            self.assertIn("region_changed=False", report)


if __name__ == "__main__":
    unittest.main()
