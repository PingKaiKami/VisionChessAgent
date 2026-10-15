import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import cv2
import numpy as np

from game_agent.chess import ChessCalibration
from game_agent.chess_dataset import (
    LABELS, _DLL_HANDLES, _load_cairosvg, _prepare_windows_cairo,
    _scale_svg_pieces, generate_dataset, grouped_split, import_verified_board,
    random_legal_positions,
)
from game_agent.geometry import Region


def fake_renderer(board, size, light, dark, piece_scale):
    image = np.zeros((size, size, 3), np.uint8)
    for row in range(8):
        for column in range(8):
            shade = 210 if (row + column) % 2 else 90
            image[round(row * size / 8):round((row + 1) * size / 8),
                  round(column * size / 8):round((column + 1) * size / 8)] = shade
    return image


class ChessDatasetTests(unittest.TestCase):
    def test_cairosvg_import_is_tried_before_windows_dll_setup(self):
        module = object()
        with patch("game_agent.chess_dataset._prepare_windows_cairo") as prepare:
            with patch("game_agent.chess_dataset.importlib.import_module", return_value=module):
                self.assertIs(module, _load_cairosvg())
        prepare.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows DLL search test")
    def test_windows_cairo_uses_explicit_dll_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CAIRO_DLL_DIR": directory}):
                with patch("game_agent.chess_dataset.os.add_dll_directory") as add_directory:
                    handle = object()
                    add_directory.return_value = handle
                    _DLL_HANDLES.clear()
                    _prepare_windows_cairo()

        add_directory.assert_called_once_with(directory)
        self.assertIn(handle, _DLL_HANDLES)

    @unittest.skipUnless(os.name == "nt", "Windows DLL search test")
    def test_windows_cairo_continues_after_dll_candidate_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CAIRO_DLL_DIR": directory}):
                with patch("game_agent.chess_dataset.Path.is_dir", return_value=True):
                    with patch(
                        "game_agent.chess_dataset.os.add_dll_directory",
                        side_effect=(FileNotFoundError(206, "too long"), object()),
                    ):
                        diagnostics = _prepare_windows_cairo()
        self.assertTrue(diagnostics[0].startswith("failed:"))
        self.assertTrue(diagnostics[1].startswith("loaded:"))

    def test_cairosvg_failure_includes_diagnostics_and_repair_hint(self):
        with patch(
            "game_agent.chess_dataset._prepare_windows_cairo",
            return_value=["failed: test-path"],
        ):
            with patch(
                "game_agent.chess_dataset.importlib.import_module",
                side_effect=OSError("missing cairo"),
            ):
                with self.assertRaisesRegex(RuntimeError, "CAIRO_DLL_DIR"):
                    _load_cairosvg()

    def test_piece_scale_changes_only_svg_piece_nodes(self):
        svg = '<rect transform="translate(0, 0)"/><use href="#king" transform="translate(45, 90)" />'
        scaled = _scale_svg_pieces(svg, 0.9)
        self.assertIn('<rect transform="translate(0, 0)"', scaled)
        self.assertIn('scale(0.9)', scaled)

    def test_random_positions_are_legal_and_cover_stages(self):
        positions = random_legal_positions(6, 7)
        self.assertEqual({"opening", "middlegame", "endgame"}, {item.stage for item in positions})
        for item in positions:
            self.assertTrue(chess.Board(item.fen).is_valid())

    def test_grouped_split_never_leaks_a_game(self):
        mapping = grouped_split(["a", "a", "b", "c", "d", "e"], 3)
        self.assertEqual(5, len(mapping))
        self.assertEqual({"train", "validation", "test"}, set(mapping.values()))

    def test_generate_crops_rgb_sized_tiles_and_keeps_variants_together(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_dataset(directory, 3, 9, variants=2, renderer=fake_renderer)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(3 * 2 * 64, len(rows))
            by_game = {}
            for row in rows:
                by_game.setdefault(row["game_id"], set()).add(row["split"])
                image = cv2.imread(str(Path(directory) / row["path"]))
                self.assertEqual((96, 96, 3), image.shape)
                self.assertIn(row["label"], LABELS)
                self.assertGreater(row["balanced_weight"], 0)
            self.assertTrue(all(len(splits) == 1 for splits in by_game.values()))

    def test_real_import_uses_only_user_confirmed_fen(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "board.png"
            cv2.imwrite(str(image_path), fake_renderer(chess.Board(), 256, "", "", 1.0))
            manifest = import_verified_board(
                Path(directory) / "data", image_path, chess.Board().board_fen(),
                ChessCalibration(Region(-900, 100, 256, 256)), source_id="manual-1",
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            e1 = next(row for row in rows if row["square"] == "e1")
            e4 = next(row for row in rows if row["square"] == "e4")
            self.assertEqual("K", e1["label"])
            self.assertEqual("empty", e4["label"])
            self.assertEqual("verified-real", e1["source"])


if __name__ == "__main__":
    unittest.main()
