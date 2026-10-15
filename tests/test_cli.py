import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

from game_agent.cli import build_parser, main
from game_agent.geometry import Region


class CliTests(unittest.TestCase):
    def test_cnn_backend_requires_model_argument_at_runtime(self):
        args = build_parser().parse_args([
            "chess-scan", "--image", "board.png", "--backend", "cnn", "--model", "model.pt"
        ])
        self.assertEqual("cnn", args.backend)
        self.assertEqual("model.pt", args.model)

    def test_training_cli_arguments(self):
        args = build_parser().parse_args([
            "chess-train", "--dataset", "data", "--checkpoint", "model.pt", "--device", "cpu"
        ])
        self.assertEqual("cpu", args.device)
        self.assertEqual(30, args.epochs)
        self.assertEqual(0, args.num_workers)
        self.assertEqual(0, args.save_epochs)

    def test_m4_cli_arguments(self):
        assets = build_parser().parse_args([
            "chess-assets", "--image", "start.png", "--fen", "8/8/8/8/8/8/8/8",
            "--empty-board", "blank.png", "--piece-style", "classic", "--output", "assets",
        ])
        self.assertEqual("classic", assets.piece_style)
        self.assertEqual("blank.png", assets.empty_board)
        manifest_assets = build_parser().parse_args([
            "chess-assets", "--manifest", "sources.json", "--output", "assets",
            "--mode", "real-only",
        ])
        self.assertEqual("sources.json", manifest_assets.manifest)
        self.assertEqual("real-only", manifest_assets.mode)
        data = build_parser().parse_args([
            "chess-real-dataset", "--assets", "assets", "--output", "data", "--mode", "real-only",
            "--no-augmentation",
        ])
        self.assertEqual(2, data.variants)
        self.assertEqual("real-only", data.mode)
        self.assertTrue(data.no_augmentation)
        train = build_parser().parse_args([
            "chess-train", "--synthetic-dataset", "synthetic", "--real-dataset", "real",
            "--real-ratio", "0.7", "--init-checkpoint", "v2.pt", "--checkpoint", "v3.pt",
        ])
        self.assertEqual(0.7, train.real_ratio)
        self.assertEqual("v2.pt", train.init_checkpoint)
        evaluate = build_parser().parse_args([
            "chess-evaluate", "--dataset", "real-test", "--model", "v2.pt",
            "--compare-model", "v3.pt",
        ])
        self.assertEqual("v3.pt", evaluate.compare_model)

    def test_training_cli_accepts_save_interval_and_rejects_negative(self):
        args = build_parser().parse_args([
            "chess-train", "--dataset", "data", "--checkpoint", "model.pt",
            "--save-epochs", "5", "--patience", "0",
        ])
        self.assertEqual(5, args.save_epochs)
        self.assertEqual(0, args.patience)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "chess-train", "--dataset", "data", "--checkpoint", "model.pt",
                "--save-epochs", "-1",
            ])

    @patch("game_agent.chess_training.train_model", side_effect=RuntimeError("CUDA unavailable"))
    def test_training_runtime_error_is_reported_cleanly(self, _train):
        output = StringIO()
        with redirect_stdout(output):
            result = main([
                "chess-train", "--dataset", "data", "--checkpoint", "model.pt",
                "--device", "cuda",
            ])
        self.assertEqual(2, result)
        self.assertIn("Training failed: CUDA unavailable", output.getvalue())

    def test_chess_play_accepts_profile(self):
        args = build_parser().parse_args([
            "chess-play",
            "--engine-path", "stockfish.exe",
            "--window-title", "Chess",
            "--profile",
        ])
        self.assertTrue(args.profile)

    def test_chess_play_accepts_unlimited_pause_and_promotion_options(self):
        args = build_parser().parse_args([
            "chess-play", "--engine-path", "stockfish.exe", "--window-title", "Chess",
            "--max-moves", "0", "--pause-timeout", "300",
            "--pointer-idle-seconds", "2", "--promotion-mode", "auto",
        ])
        self.assertEqual(0, args.max_moves)
        self.assertEqual(300.0, args.pause_timeout)
        self.assertEqual(2.0, args.pointer_idle_seconds)
        self.assertEqual("auto", args.promotion_mode)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "chess-play", "--engine-path", "stockfish.exe",
                "--window-title", "Chess", "--max-moves", "-1",
            ])

    @patch("game_agent.cli.ScreenCapture")
    @patch("game_agent.cli.ChessInputAdapter")
    @patch("game_agent.cli.InputController")
    @patch("game_agent.cli.SafetyGuard")
    @patch("game_agent.cli.ChessCalibration.load")
    @patch("game_agent.cli.time.sleep")
    def test_after_delay_waits_before_screenshot(
        self, sleep, load, _guard, controller_type, _adapter, capture_type
    ):
        events = []
        calibration = MagicMock(board=Region(-900, 20, 800, 800))
        load.return_value = calibration
        controller_type.return_value.__enter__.return_value.actions = []
        sleep.side_effect = lambda seconds: events.append(("sleep", seconds))
        capture_type.return_value.save.side_effect = (
            lambda path, region: events.append(("save", path, region)) or path
        )

        result = main([
            "move", "e2", "e4", "--window-title", "Chess",
            "--after", "after.png", "--after-delay", "1.0",
        ])

        self.assertEqual(0, result)
        self.assertEqual([
            ("sleep", 1.0),
            ("save", "after.png", calibration.board),
        ], events)

    @patch("game_agent.cli.ScreenCapture")
    @patch("game_agent.cli.ChessCalibration.load")
    def test_screenshot_config_captures_only_absolute_board_region(self, load, capture_type):
        board = Region(-1200, 100, 640, 640)
        load.return_value = MagicMock(board=board)
        capture_type.return_value.save.return_value = "board.png"

        result = main([
            "screenshot", "--config", "chess-calibration.json", "--output", "board.png"
        ])

        self.assertEqual(0, result)
        capture_type.return_value.save.assert_called_once_with("board.png", board)


if __name__ == "__main__":
    unittest.main()
