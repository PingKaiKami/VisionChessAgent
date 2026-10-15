import json
import random
import tempfile
import unittest
from pathlib import Path

import chess
import cv2
import numpy as np

from game_agent.chess import ChessCalibration
from game_agent.chess_assets import (
    AssetLibrary, _align_empty_board, _alpha_composite, _paired_foreground, extract_piece_assets,
    _square_color, extract_real_asset_library, generate_real_style_dataset,
)
from game_agent.chess_dataset import LABELS
from game_agent.chess_perception import BoardExtractor
from game_agent.geometry import Region


def fixture_board(side=384, board=None):
    cell = side // 8
    image = np.empty((side, side, 3), np.uint8)
    board = board or chess.Board()
    for row in range(8):
        for column in range(8):
            square = chess.square(column, 7 - row)
            light = (chess.square_rank(square) + chess.square_file(square)) % 2
            image[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = (185, 215, 235) if light else (80, 120, 150)
            piece = board.piece_at(square)
            if piece:
                center = (column * cell + cell // 2, row * cell + cell // 2)
                color = (245, 245, 245) if piece.color else (15, 15, 15)
                if piece.piece_type == chess.ROOK:
                    cv2.rectangle(image, (center[0] - 13, center[1] - 14), (center[0] + 13, center[1] + 14), color, -1)
                elif piece.piece_type == chess.BISHOP:
                    cv2.ellipse(image, center, (9, 17), 0, 0, 360, color, -1, cv2.LINE_AA)
                elif piece.piece_type == chess.KNIGHT:
                    points = np.asarray([(center[0] - 14, center[1] + 14), (center[0] + 14, center[1] + 14), (center[0] + 5, center[1] - 16)])
                    cv2.fillPoly(image, [points], color, cv2.LINE_AA)
                else:
                    cv2.circle(image, center, cell // 4, color, -1, cv2.LINE_AA)
                cv2.putText(image, piece.symbol().upper(), (center[0] - 9, center[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 40, 200), 2)
    return image


def fixture_blank(side=384):
    cell = side // 8
    image = np.empty((side, side, 3), np.uint8)
    for row in range(8):
        for column in range(8):
            square = chess.square(column, 7 - row)
            light = (chess.square_rank(square) + chess.square_file(square)) % 2
            image[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = (185, 215, 235) if light else (80, 120, 150)
    return image


class ChessAssetTests(unittest.TestCase):
    def _assets(self, directory):
        image = Path(directory) / "start.png"
        cv2.imwrite(str(image), fixture_board())
        root = Path(directory) / "assets"
        manifest = extract_piece_assets(
            image, ChessCalibration(Region(0, 0, 384, 384)),
            chess.Board().board_fen(), "fixture-style", root,
        )
        return root, json.loads(manifest.read_text(encoding="utf-8"))

    def _real_assets(self, directory, *, include_opposite=True):
        root = Path(directory)
        calibration = ChessCalibration(Region(0, 0, 384, 384))
        calibration.save(root / "calibration.json")
        cv2.imwrite(str(root / "blank.png"), fixture_blank())
        cv2.imwrite(str(root / "start.png"), fixture_board())
        sources = [{
            "source_id": "start", "image": "start.png", "fen": chess.Board().board_fen(),
        }]
        if include_opposite:
            opposite_fen = "rnb2bnr/pppqkppp/3pp3/8/8/3PP3/PPPQKPPP/RNB2BNR"
            opposite = chess.Board(f"{opposite_fen} w - - 0 1")
            cv2.imwrite(str(root / "opposite.png"), fixture_board(board=opposite))
            sources.append({
                "source_id": "opposite-kq", "image": "opposite.png", "fen": opposite_fen,
            })
        manifest = root / "sources.json"
        manifest.write_text(json.dumps({
            "piece_style": "fixture-style", "config": "calibration.json",
            "empty_board": "blank.png", "sources": sources,
        }), encoding="utf-8")
        output = root / "real-assets"
        result = extract_real_asset_library(manifest, output)
        return output, json.loads(result.read_text(encoding="utf-8"))

    def test_initial_board_extracts_complete_labelled_rgba_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root, metadata = self._assets(directory)
            self.assertEqual(set(LABELS), set(metadata["labels"]))
            self.assertEqual({"light", "dark"}, set(metadata["empty_assets"]))
            self.assertEqual(set(LABELS[:-1]), {item["label"] for item in metadata["assets"]})
            self.assertTrue((root / "asset-preview.png").exists())
            rooks = [item for item in metadata["assets"] if item["label"] == "r"]
            bishops = [item for item in metadata["assets"] if item["label"] == "b"]
            self.assertTrue(rooks and bishops)
            self.assertTrue(all(item["path"] != bishops[0]["path"] for item in rooks))
            rgba = cv2.imread(str(root / rooks[0]["path"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(4, rgba.shape[2])
            self.assertGreater(np.count_nonzero(rgba[:, :, 3]), 0)

    def test_alpha_compositing_changes_foreground_but_preserves_corners(self):
        background = np.full((32, 32, 3), 100, np.uint8)
        foreground = np.zeros((32, 32, 4), np.uint8)
        foreground[8:24, 8:24, :3] = 240
        foreground[8:24, 8:24, 3] = 255
        result = _alpha_composite(background, foreground)
        np.testing.assert_array_equal(result[0, 0], background[0, 0])
        self.assertTrue(np.all(result[16, 16] == 240))

    def test_soft_alpha_and_decontamination_recover_known_composite(self):
        side = 64
        background = np.full((side, side, 3), (65, 125, 155), np.uint8)
        foreground_color = np.asarray((225, 238, 248), np.float32)
        alpha = np.zeros((side, side), np.float32)
        cv2.circle(alpha, (32, 32), 20, 1.0, -1, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        occupied = np.clip(
            foreground_color * alpha[:, :, None]
            + background.astype(np.float32) * (1.0 - alpha[:, :, None]), 0, 255,
        ).astype(np.uint8)

        rgba, reconstruction, quality, diagnostics = _paired_foreground(
            occupied, background, 2.0
        )
        estimated = rgba[:, :, 3].astype(np.float32) / 255.0
        self.assertTrue(np.isfinite(estimated).all())
        self.assertGreaterEqual(float(estimated.min()), 0.0)
        self.assertLessEqual(float(estimated.max()), 1.0)
        self.assertGreater(quality["soft_alpha_fraction"], 0.0)
        self.assertLess(float(np.mean(np.abs(estimated - alpha))), 0.12)
        core = estimated > 0.8
        self.assertLess(float(np.mean(np.abs(rgba[core, :3].astype(float) - foreground_color))), 12.0)
        self.assertLess(float(np.mean(cv2.absdiff(occupied, reconstruction))), 4.0)

        opposite = np.full_like(background, (210, 190, 80))
        composed = _alpha_composite(opposite, rgba)
        expected = np.clip(
            foreground_color * alpha[:, :, None]
            + opposite.astype(np.float32) * (1.0 - alpha[:, :, None]), 0, 255,
        ).astype(np.uint8)
        edge = (alpha > 0.02) & (alpha < 0.98)
        self.assertLess(float(np.mean(cv2.absdiff(composed, expected)[edge])), 18.0)
        self.assertEqual({0, 128, 255}, set(np.unique(diagnostics["trimap"])))

    def test_soft_alpha_keeps_internal_hole_and_png_channel_order(self):
        background = np.full((64, 64, 3), (80, 145, 175), np.uint8)
        alpha = np.zeros((64, 64), np.float32)
        cv2.circle(alpha, (32, 32), 21, 1.0, -1, cv2.LINE_AA)
        cv2.circle(alpha, (32, 32), 8, 0.0, -1, cv2.LINE_AA)
        foreground = np.full((64, 64, 3), (20, 30, 220), np.float32)
        occupied = np.clip(
            foreground * alpha[:, :, None] + background * (1.0 - alpha[:, :, None]), 0, 255
        ).astype(np.uint8)
        rgba, _, _, _ = _paired_foreground(occupied, background, 2.0)
        self.assertLess(int(rgba[32, 32, 3]), 8)
        self.assertGreater(int(rgba[32, 15, 3]), 220)
        self.assertTrue(np.isfinite(rgba).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            self.assertTrue(cv2.imwrite(str(path), rgba))
            loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(loaded, rgba)

    def test_paired_empty_board_aligns_and_creates_clean_masks_and_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            occupied = fixture_board()
            blank = fixture_blank()
            shifted = cv2.warpAffine(
                blank, np.float32([[1, 0, 2], [0, 1, -1]]), (384, 384),
                borderMode=cv2.BORDER_REFLECT,
            )
            occupied_path = Path(directory) / "start.png"
            blank_path = Path(directory) / "blank.png"
            cv2.imwrite(str(occupied_path), occupied)
            cv2.imwrite(str(blank_path), shifted)
            root = Path(directory) / "paired"
            manifest = extract_piece_assets(
                occupied_path, ChessCalibration(Region(0, 0, 384, 384)),
                chess.Board().board_fen(), "paired-style", root,
                empty_board_path=blank_path,
            )
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual("ecc-translation", metadata["alignment"]["method"])
            self.assertTrue(all(item["extraction"] == "paired-empty-board" for item in metadata["assets"]))
            self.assertTrue(all(item["accepted"] for item in metadata["assets"]))
            for name in ("asset-preview.png", "compositing-preview.png", "reconstruction-preview.png", "metadata.json"):
                self.assertTrue((root / name).exists(), name)
            white_bishop = next(item for item in metadata["assets"] if item["label"] == "B")
            mask = cv2.imread(str(root / white_bishop["mask_path"]), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(np.mean(mask > 16), 0.05)
            self.assertLess(white_bishop["missed_change_fraction"], 0.015)
            self.assertLess(white_bishop["reconstruction_mae"], 12.0)
            self.assertIn("soft_alpha_fraction", white_bishop)
            self.assertTrue((root / white_bishop["trimap_path"]).exists())
            self.assertTrue((root / white_bishop["difference_path"]).exists())

    def test_paired_board_rejects_scale_theme_and_alignment_mismatch(self):
        board = chess.Board()
        occupied = fixture_board()
        with self.assertRaisesRegex(ValueError, "scale differs too much"):
            _align_empty_board(occupied, fixture_blank(320), board)
        wrong_theme = np.full_like(occupied, 25)
        with self.assertRaisesRegex(ValueError, "does not match|Unable to align"):
            _align_empty_board(occupied, wrong_theme, board)

    def test_paired_black_rook_and_bishop_keep_distinct_silhouettes(self):
        with tempfile.TemporaryDirectory() as directory:
            occupied_path = Path(directory) / "start.png"
            blank_path = Path(directory) / "blank.png"
            cv2.imwrite(str(occupied_path), fixture_board())
            cv2.imwrite(str(blank_path), fixture_blank())
            root = Path(directory) / "paired"
            metadata_path = extract_piece_assets(
                occupied_path, ChessCalibration(Region(0, 0, 384, 384)),
                chess.Board().board_fen(), "paired-style", root,
                empty_board_path=blank_path,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rook = next(item for item in metadata["assets"] if item["label"] == "r")
            bishop = next(item for item in metadata["assets"] if item["label"] == "b")
            rook_mask = cv2.imread(str(root / rook["mask_path"]), cv2.IMREAD_GRAYSCALE)
            bishop_mask = cv2.imread(str(root / bishop["mask_path"]), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(float(np.mean(cv2.absdiff(rook_mask, bishop_mask))), 5.0)

    def test_real_style_generation_has_grouped_splits_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._assets(directory)
            output = Path(directory) / "dataset"
            manifest = generate_real_style_dataset(root, output, 3, 7, variants=2)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(3 * 2 * 64, len(rows))
            self.assertEqual({"real-style-composite"}, {row["source"] for row in rows})
            self.assertEqual({"fixture-style"}, {row["piece_style"] for row in rows})
            by_game = {}
            for row in rows:
                by_game.setdefault(row["game_id"], set()).add(row["split"])
                self.assertIn(row["label"], LABELS)
            self.assertTrue(all(len(value) == 1 for value in by_game.values()))
            self.assertTrue(any((output / "previews").glob("*.png")))

    def test_multi_source_real_assets_cover_every_label_and_square_color(self):
        with tempfile.TemporaryDirectory() as directory:
            root, metadata = self._real_assets(directory)
            self.assertEqual("real-only", metadata["mode"])
            self.assertEqual([], metadata["statistics"]["missing"])
            for label in LABELS[:-1]:
                self.assertGreater(metadata["statistics"]["counts"][label]["light"], 0)
                self.assertGreater(metadata["statistics"]["counts"][label]["dark"], 0)
            for label in ("K", "Q", "k", "q"):
                sources = {
                    item["source_id"] for item in metadata["assets"] if item["label"] == label
                }
                self.assertEqual({"start", "opposite-kq"}, sources)
            paths = [item["path"] for item in metadata["assets"]]
            self.assertEqual(len(paths), len(set(paths)))
            source_fens = {item["source_id"]: item["source_fen"] for item in metadata["assets"]}
            for item in metadata["assets"]:
                labelled = chess.Board(f"{source_fens[item['source_id']]} w - - 0 1")
                self.assertEqual(
                    item["label"], labelled.piece_at(chess.parse_square(item["square"])).symbol(),
                )
            self.assertTrue((root / "real-asset-preview.png").exists())
            self.assertTrue((root / "asset-statistics.json").exists())

    def test_real_only_render_uses_same_color_full_tiles_without_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._real_assets(directory)
            library = AssetLibrary.load(root)
            with unittest.mock.patch(
                "game_agent.chess_assets._alpha_composite",
                side_effect=AssertionError("real-only must not alpha composite"),
            ):
                image, usage = library.render(
                    chess.Board(), 768, random.Random(9), mode="real-only", return_usage=True,
                )
            self.assertEqual((768, 768, 3), image.shape)
            rendered = BoardExtractor(ChessCalibration(Region(0, 0, 768, 768))).squares(image)
            for square, item in usage.items():
                self.assertEqual(_square_color(square), item["square_color"])
                original = cv2.imread(str(root / item["path"]), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(rendered[square], original)
            self.assertTrue(all(item["source_id"] for item in usage.values()))

    def test_real_only_missing_color_is_reported_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing true tiles.*Q/dark.*K/light"):
                self._real_assets(directory, include_opposite=False)

    def test_real_only_dataset_matches_fen_and_records_actual_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._real_assets(directory)
            output = Path(directory) / "real-dataset"
            manifest = generate_real_style_dataset(
                root, output, 3, 17, variants=1, mode="real-only", augmentation=False,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(3 * 64, len(rows))
            self.assertEqual({"real-only"}, {row["render_mode"] for row in rows})
            self.assertTrue(all(row["augmentation"] is None for row in rows))
            for row in rows:
                board = chess.Board(row["fen"])
                piece = board.piece_at(chess.parse_square(row["square"]))
                self.assertEqual(piece.symbol() if piece else "empty", row["label"])
                expected_color = "light" if row["square_color"] else "dark"
                self.assertEqual(expected_color, row["asset_used"]["square_color"])
            info = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertFalse(info["independent_real_test"])
            self.assertEqual("real-only", info["render_mode"])

    def test_low_quality_assets_are_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, metadata = self._assets(directory)
            for item in metadata["assets"]:
                if item["label"] == "r":
                    item["accepted"] = False
            (root / "assets.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No accepted assets for: r"):
                AssetLibrary.load(root)


if __name__ == "__main__":
    unittest.main()
