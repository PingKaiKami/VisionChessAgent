import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from game_agent.chess import ChessCalibration, Orientation
from game_agent.chess_perception import (
    BoardExtractor,
    INITIAL_POSITION,
    TemplateLibrary,
    screen_cell_square,
)
from game_agent.geometry import Region


SIZE = 32


def synthetic_board(orientation=Orientation.WHITE_BOTTOM, position=None):
    position = INITIAL_POSITION if position is None else position
    image = np.zeros((SIZE * 8, SIZE * 8, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            square = screen_cell_square(row, column, orientation)
            shade = 190 if (ord(square[0]) - ord("a") + int(square[1])) % 2 else 90
            tile = image[row * SIZE:(row + 1) * SIZE, column * SIZE:(column + 1) * SIZE]
            tile[:] = shade
            symbol = position.get(square, ".")
            if symbol != ".":
                color = (245, 245, 245) if symbol.isupper() else (20, 20, 20)
                cv2.putText(tile, symbol, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return image


def rendered_board(side, orientation=Orientation.WHITE_BOTTOM, position=None):
    position = INITIAL_POSITION if position is None else position
    image = np.zeros((side, side, 3), dtype=np.uint8)
    for row in range(8):
        y0, y1 = round(row * side / 8), round((row + 1) * side / 8)
        for column in range(8):
            x0, x1 = round(column * side / 8), round((column + 1) * side / 8)
            square = screen_cell_square(row, column, orientation)
            light = (ord(square[0]) - ord("a") + int(square[1])) % 2
            image[y0:y1, x0:x1] = (208, 236, 235) if light else (82, 149, 115)
            symbol = position.get(square, ".")
            if symbol == ".":
                continue
            cell = (x1 - x0 + y1 - y0) / 2
            scale = cell / 42.0
            thickness = max(1, round(cell / 16.0))
            (width, height), _ = cv2.getTextSize(
                symbol, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
            origin = (
                x0 + (x1 - x0 - width) // 2,
                y0 + (y1 - y0 + height) // 2,
            )
            outline = (210, 210, 210) if symbol.islower() else (45, 45, 45)
            cv2.putText(
                image, symbol, origin,
                cv2.FONT_HERSHEY_SIMPLEX, scale, outline,
                thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                image, symbol,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (238, 238, 238) if symbol.isupper() else (25, 25, 25),
                thickness,
                cv2.LINE_AA,
            )
    return image


class ChessPerceptionTests(unittest.TestCase):
    def test_board_split_respects_both_orientations(self):
        for orientation, top_left in (
            (Orientation.WHITE_BOTTOM, "a8"),
            (Orientation.BLACK_BOTTOM, "h1"),
        ):
            calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8), orientation)
            tiles = BoardExtractor(calibration).squares(synthetic_board(orientation))
            self.assertEqual(64, len(tiles))
            expected = 190 if (ord(top_left[0]) - ord("a") + int(top_left[1])) % 2 else 90
            self.assertEqual(expected, int(tiles[top_left][0, 0, 0]))

    def test_initial_position_and_fen(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        image = synthetic_board()
        state = TemplateLibrary.learn(image, calibration).recognize(image, calibration)
        self.assertFalse(state.unknown_squares)
        self.assertEqual("white", state.squares["e1"].color)
        self.assertEqual("king", state.squares["e1"].piece_type)
        self.assertEqual("empty", state.squares["e4"].piece_type)
        self.assertEqual("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", state.piece_placement())

    def test_e2_to_e4(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        moved = dict(INITIAL_POSITION)
        moved.pop("e2")
        moved["e4"] = "P"
        state = templates.recognize(synthetic_board(position=moved), calibration)
        self.assertEqual(".", state.squares["e2"].symbol)
        self.assertEqual("P", state.squares["e4"].symbol)

    def test_black_king_recognizes_on_opposite_color_square(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        moved = dict(INITIAL_POSITION)
        moved.pop("e8")
        moved["d8"] = "k"

        state = templates.recognize(synthetic_board(position=moved), calibration)

        self.assertEqual("k", state.squares["d8"].symbol)

    def test_white_king_recognizes_on_opposite_color_square(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        moved = dict(INITIAL_POSITION)
        moved.pop("e1")
        moved["d1"] = "K"

        state = templates.recognize(synthetic_board(position=moved), calibration)

        self.assertEqual("K", state.squares["d1"].symbol)

    def test_white_queen_recognizes_on_opposite_color_square(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        moved = dict(INITIAL_POSITION)
        moved.pop("d1")
        moved["d8"] = "Q"

        state = templates.recognize(synthetic_board(position=moved), calibration)

        self.assertEqual("Q", state.squares["d8"].symbol)

    def test_black_queen_recognizes_on_opposite_color_square(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        moved = dict(INITIAL_POSITION)
        moved.pop("d8")
        moved["e8"] = "q"

        state = templates.recognize(synthetic_board(position=moved), calibration)

        self.assertEqual("q", state.squares["e8"].symbol)

    def test_same_templates_recognize_independently_rendered_sizes(self):
        fullscreen = rendered_board(1135)
        templates = TemplateLibrary.learn(
            fullscreen, ChessCalibration(Region(0, 0, 1135, 1135))
        )
        moved = dict(INITIAL_POSITION)
        moved.pop("e2")
        moved["e4"] = "P"
        windowed = rendered_board(706, position=moved)
        state = templates.recognize(
            windowed, ChessCalibration(Region(0, 0, 706, 706))
        )

        self.assertFalse(state.unknown_squares)
        self.assertEqual(".", state.squares["e2"].symbol)
        self.assertEqual("P", state.squares["e4"].symbol)
        for square in ("a7", "b7"):
            self.assertEqual("p", state.squares[square].symbol)
            self.assertGreaterEqual(state.squares[square].best_score, 0.72)

    def test_non_divisible_board_normalizes_every_square_to_96_pixels(self):
        calibration = ChessCalibration(Region(0, 0, 707, 707))
        extractor = BoardExtractor(calibration)
        image = rendered_board(707)

        x_edges, y_edges = extractor.grid_edges(image)
        tiles = extractor.squares(image)

        self.assertAlmostEqual(707 / 8, x_edges[1])
        self.assertAlmostEqual(707 / 8, y_edges[1])
        self.assertTrue(all(tile.shape[:2] == (96, 96) for tile in tiles.values()))

    def test_black_bottom_recognition(self):
        calibration = ChessCalibration(
            Region(0, 0, SIZE * 8, SIZE * 8), Orientation.BLACK_BOTTOM
        )
        image = synthetic_board(Orientation.BLACK_BOTTOM)
        state = TemplateLibrary.learn(image, calibration).recognize(image, calibration)
        self.assertEqual("K", state.squares["e1"].symbol)
        self.assertEqual("k", state.squares["e8"].symbol)

    def test_board_only_image_ignores_negative_absolute_calibration_offset(self):
        calibration = ChessCalibration(Region(-1200, 150, SIZE * 8, SIZE * 8))
        image = synthetic_board()

        state = TemplateLibrary.learn(image, calibration).recognize(image, calibration)

        self.assertFalse(state.unknown_squares)
        self.assertEqual("K", state.squares["e1"].symbol)
        self.assertEqual("p", state.squares["e7"].symbol)

    def test_unrecognized_tile_is_unknown_without_high_confidence(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        templates = TemplateLibrary.learn(synthetic_board(), calibration)
        changed = synthetic_board()
        changed[4 * SIZE:5 * SIZE, 4 * SIZE:5 * SIZE] = np.random.default_rng(1).integers(
            0, 256, (SIZE, SIZE, 3), dtype=np.uint8
        )
        state = templates.recognize(changed, calibration)
        result = state.squares["e4"]
        self.assertEqual("unknown", result.symbol)
        self.assertLess(result.confidence, 0.9)
        self.assertIsNotNone(result.best_label)
        self.assertGreaterEqual(result.best_score, 0.0)
        self.assertGreaterEqual(result.margin, 0.0)
        self.assertIsNotNone(result.second_label)
        self.assertGreaterEqual(result.second_score, 0.0)

    def test_template_save_load(self):
        calibration = ChessCalibration(Region(0, 0, SIZE * 8, SIZE * 8))
        expected = TemplateLibrary.learn(synthetic_board(), calibration)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.npz"
            expected.save(path)
            actual = TemplateLibrary.load(path)
        np.testing.assert_array_equal(expected.features, actual.features)

    def test_obsolete_template_requires_relearning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-templates.npz"
            np.savez_compressed(
                path,
                labels=np.asarray(["."]),
                colors=np.asarray([0], dtype=np.uint8),
                features=np.zeros((1, 48, 48), dtype=np.uint8),
            )
            with self.assertRaisesRegex(ValueError, "run chess-learn again"):
                TemplateLibrary.load(path)


if __name__ == "__main__":
    unittest.main()
