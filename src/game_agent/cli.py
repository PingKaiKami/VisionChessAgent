from __future__ import annotations

import argparse
import time

import chess

from .board_detection import BoardDetectionError, BoardDetector
from .capture import ScreenCapture
from .chess import ChessCalibration, ChessInputAdapter, Orientation
from .chess_agent import (
    AutonomousChessAgent,
    ChessAgentError,
    ScreenBoardObserver,
    StockfishEngine,
)
from .chess_perception import TemplateLibrary, load_board_image, save_debug_image
from .geometry import Region
from .input_controller import InputController
from .safety import SafetyGuard


PROFILE_STAGES = (
    "Full Screenshot",
    "Candidate Detection",
    "Candidate Refinement",
    "Piece Recognition",
    "Precheck",
    "Stockfish",
    "Move Verification",
    "Sleep / Polling",
)


def _region(values: list[int]) -> Region:
    return Region(*values)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="game-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Save manual board calibration")
    calibrate.add_argument("--board", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    calibrate.add_argument("--orientation", choices=[item.value for item in Orientation], default=Orientation.WHITE_BOTTOM.value)
    calibrate.add_argument("--output", default="chess-calibration.json")

    coordinate = commands.add_parser("coordinate", help="Print a square center")
    coordinate.add_argument("square")
    coordinate.add_argument("--config", default="chess-calibration.json")

    screenshot = commands.add_parser("screenshot", help="Save a desktop region as PNG")
    screenshot_source = screenshot.add_mutually_exclusive_group()
    screenshot_source.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    screenshot_source.add_argument("--config", help="Capture only the calibrated Chess board")
    screenshot.add_argument("--output", required=True)

    move = commands.add_parser("move", help="Perform or preview a chess move")
    move.add_argument("source")
    move.add_argument("target")
    move.add_argument("--config", default="chess-calibration.json")
    move.add_argument("--window-title", required=True, help="Required foreground window title substring")
    move.add_argument("--duration", type=float, default=0.4)
    move.add_argument("--start-delay", type=float, default=0.0, help="Seconds available to focus the game window")
    move.add_argument("--drag", action="store_true")
    move.add_argument("--enable-live", action="store_true", help="Explicitly enable real mouse input")
    move.add_argument("--after", help="Save the board region after the move")
    move.add_argument("--after-delay", type=float, default=0.0, help="Seconds to wait before the after screenshot")

    learn = commands.add_parser("chess-learn", help="Learn templates from a standard initial board")
    learn.add_argument("--image", required=True)
    learn.add_argument("--config", default="chess-calibration.json")
    learn.add_argument("--templates", default="chess-templates.npz")

    scan = commands.add_parser("chess-scan", help="Recognize a board image")
    scan.add_argument("--image", required=True)
    scan.add_argument("--config", default="chess-calibration.json")
    scan.add_argument("--templates", default="chess-templates.npz")
    scan.add_argument("--backend", choices=("template", "cnn"), default="template")
    scan.add_argument("--model", help="CNN checkpoint path (required for --backend cnn)")
    scan.add_argument("--debug")

    play = commands.add_parser("chess-play", help="Play an offline Chess game as White")
    play.add_argument("--engine-path", required=True)
    play.add_argument("--config", default="chess-calibration.json")
    play.add_argument("--templates", default="chess-templates.npz")
    play.add_argument("--backend", choices=("template", "cnn"), default="template")
    play.add_argument("--model", help="CNN checkpoint path (required for --backend cnn)")
    play.add_argument("--window-title", required=True)
    play.add_argument("--start-delay", type=float, default=3.0)
    play.add_argument("--think-time", type=float, default=0.2)
    play.add_argument("--poll-interval", type=float, default=1.0)
    play.add_argument("--wait-timeout", "--verify-timeout", dest="verify_timeout", type=float, default=60.0)
    play.add_argument("--settle-delay", type=float, default=1.0)
    play.add_argument(
        "--max-moves", type=_nonnegative_int, default=100,
        help="Maximum agent moves; 0 continues until a confirmed board-state game over",
    )
    play.add_argument("--max-retries", type=int, default=20)
    play.add_argument("--stable-observations", type=int, default=2)
    play.add_argument(
        "--pause-timeout", type=_nonnegative_float, default=300.0,
        help="Maximum seconds to wait in a recoverable paused state",
    )
    play.add_argument(
        "--pointer-idle-seconds", type=_nonnegative_float, default=2.0,
        help="Required pointer stability before recovery",
    )
    play.add_argument(
        "--promotion-mode", choices=("auto", "manual"), default="auto",
        help="auto supports only the verified white-bottom Chess.com promotion menu",
    )
    play.add_argument("--auto-detect", action="store_true")
    play.add_argument("--board-orientation", choices=[item.value for item in Orientation])
    play.add_argument("--enable-live", action="store_true")
    play.add_argument("--debug", action="store_true")
    play.add_argument("--profile", action="store_true")

    dataset = commands.add_parser("chess-dataset", help="Generate a synthetic CNN dataset")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--count", type=int, default=1000, help="Number of legal positions")
    dataset.add_argument("--variants", type=int, default=2)
    dataset.add_argument("--seed", type=int, default=42)

    import_board = commands.add_parser("chess-import", help="Import a user-labelled real board")
    import_board.add_argument("--dataset", required=True)
    import_board.add_argument("--image", required=True)
    import_board.add_argument("--fen", required=True, help="User-confirmed FEN or piece placement")
    import_board.add_argument("--config", default="chess-calibration.json")
    import_board.add_argument("--split", choices=("train", "validation", "test"), default="test")
    import_board.add_argument("--source-id", required=True)

    assets = commands.add_parser("chess-assets", help="Extract reusable pieces from a verified real board")
    assets.add_argument("--image")
    assets.add_argument("--manifest", help="Multi-source real-only asset manifest")
    assets.add_argument("--empty-board", help="Matching empty board image for paired background subtraction")
    assets.add_argument("--config", default="chess-calibration.json")
    assets.add_argument("--fen", help="User-confirmed FEN or piece placement")
    assets.add_argument("--piece-style")
    assets.add_argument("--mode", choices=("real-only", "composite-allowed"))
    assets.add_argument("--output", required=True)

    real_dataset = commands.add_parser("chess-real-dataset", help="Generate legal boards from real piece assets")
    real_dataset.add_argument("--assets", required=True)
    real_dataset.add_argument("--output", required=True)
    real_dataset.add_argument("--count", type=int, default=1000)
    real_dataset.add_argument("--variants", type=int, default=2)
    real_dataset.add_argument("--seed", type=int, default=42)
    real_dataset.add_argument("--mode", choices=("real-only", "composite-allowed"))
    real_dataset.add_argument("--no-augmentation", action="store_true")

    train = commands.add_parser("chess-train", help="Train the small Chess CNN")
    train.add_argument("--dataset")
    train.add_argument("--synthetic-dataset")
    train.add_argument("--real-dataset")
    train.add_argument("--real-ratio", type=float, default=0.5)
    train.add_argument("--init-checkpoint")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers; keep 0 for maximum Windows compatibility",
    )
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--patience", type=_nonnegative_int, default=5)
    train.add_argument(
        "--save-epochs", type=_nonnegative_int, default=0,
        help="Save an additional checkpoint every N epochs; 0 disables it",
    )

    evaluate = commands.add_parser("chess-evaluate", help="Evaluate CNN and optional templates")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--compare-model", help="Optional second CNN checkpoint for a fair comparison")
    evaluate.add_argument("--templates")
    evaluate.add_argument("--output", default="cnn-evaluation.json")
    return parser


def _load_perception(backend: str, templates: str, model: str | None):
    if backend == "template":
        return TemplateLibrary.load(templates)
    if not model:
        raise ValueError("--model is required for --backend cnn")
    from .chess_cnn import CNNPerception
    return CNNPerception(model)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        calibration = ChessCalibration(_region(args.board), Orientation(args.orientation))
        calibration.save(args.output)
        print(f"Saved calibration to {args.output}")
    elif args.command == "coordinate":
        print(ChessCalibration.load(args.config).square_center(args.square))
    elif args.command == "screenshot":
        region = (
            ChessCalibration.load(args.config).board
            if args.config
            else _region(args.region) if args.region else None
        )
        print(ScreenCapture().save(args.output, region))
    elif args.command == "move":
        calibration = ChessCalibration.load(args.config)
        guard = SafetyGuard(args.window_title, calibration.board)
        if args.start_delay > 0:
            print(f"Focus the target window; starting in {args.start_delay:g} seconds...")
            time.sleep(args.start_delay)
        with InputController(guard, enable_live=args.enable_live) as controller:
            ChessInputAdapter(calibration, controller).move(
                args.source, args.target, duration=args.duration, drag=args.drag
            )
            mode = "LIVE" if args.enable_live else "DRY-RUN"
            print(f"{mode}: {args.source}->{args.target}; actions={controller.actions}")
        if args.after:
            if args.after_delay > 0:
                time.sleep(args.after_delay)
            print(ScreenCapture().save(args.after, calibration.board))
    elif args.command == "chess-learn":
        calibration = ChessCalibration.load(args.config)
        templates = TemplateLibrary.learn(load_board_image(args.image), calibration)
        print(templates.save(args.templates))
    elif args.command == "chess-scan":
        calibration = ChessCalibration.load(args.config)
        image = load_board_image(args.image)
        state = _load_perception(args.backend, args.templates, args.model).recognize(image, calibration)
        for rank in range(8, 0, -1):
            cells = [state.squares[f"{file_}{rank}"] for file_ in "abcdefgh"]
            print(f"{rank} {' '.join(f'{cell.symbol}:{cell.confidence:.2f}' for cell in cells)}")
        print("  a b c d e f g h")
        print(f"FEN: {state.piece_placement()}")
        print(f"Unknown squares: {', '.join(state.unknown_squares) or 'none'}")
        for square in state.unknown_squares:
            result = state.squares[square]
            print(
                f"  {square}: best={result.best_label}:{result.best_score:.4f}; "
                f"second={result.second_label}:{result.second_score:.4f}; "
                f"margin={result.margin:.4f}"
            )
        if args.debug:
            print(save_debug_image(args.debug, image, calibration, state))
    elif args.command == "chess-play":
        timings: dict[str, list[float]] = {}

        def profile(stage: str, elapsed: float) -> None:
            if args.profile:
                timings.setdefault(stage, []).append(elapsed)

        if args.start_delay > 0:
            print(f"Focus the target window; starting in {args.start_delay:g} seconds...")
            time.sleep(args.start_delay)
        try:
            calibration = ChessCalibration.load(args.config)
            perception = _load_perception(args.backend, args.templates, args.model)
            detector = BoardDetector(
                debug_directory="chess-debug/detection" if args.debug else None,
                profiler=profile if args.profile else None,
            ) if args.auto_detect else None
            orientation = Orientation(args.board_orientation) if args.board_orientation else None
            if detector:
                started = time.perf_counter()
                frame = ScreenCapture().capture_frame()
                profile("Full Screenshot", time.perf_counter() - started)
                detection = detector.detect(
                    frame,
                    perception,
                    expected_placement=chess.Board().board_fen(),
                    orientation=orientation,
                )
                calibration = detection.calibration
                print(
                    f"Detected board: {calibration.board}; "
                    f"orientation={calibration.orientation.value}; "
                    f"confidence={detection.confidence:.2f}"
                )
                print(f"Detected FEN: {detection.state.piece_placement()}")
            guard = SafetyGuard(args.window_title, calibration.board)
            with StockfishEngine(args.engine_path, args.think_time) as engine:
                with InputController(guard, enable_live=args.enable_live) as controller:
                    adapter = ChessInputAdapter(calibration, controller)
                    observer = ScreenBoardObserver(
                        calibration,
                        perception,
                        debug_directory="chess-debug" if args.debug else None,
                        detector=detector,
                        orientation=orientation,
                        profiler=profile if args.profile else None,
                    )

                    from .chess_agent import PromotionController
                    promotion = PromotionController(
                        calibration,
                        observer.capture_board_image,
                        controller.click,
                        poll_interval=min(0.25, max(0.05, args.poll_interval)),
                        timeout=args.verify_timeout,
                        stable_observations=args.stable_observations,
                        logger=print if args.debug else None,
                    )

                    def apply_relocation(updated: ChessCalibration) -> None:
                        adapter.calibration = updated
                        guard.allowed_region = updated.board
                        promotion.calibration = updated

                    observer.on_relocated = apply_relocation

                    def pointer_check(move) -> bool:
                        target = chess.square_name(move.to_square)
                        point = adapter.calibration.square_center(target)
                        guard.verify(point)
                        return controller.pointer_near(*point)

                    agent = AutonomousChessAgent(
                        engine,
                        observer,
                        adapter,
                        enable_live=args.enable_live,
                        poll_interval=args.poll_interval,
                        wait_timeout=args.verify_timeout,
                        settle_delay=args.settle_delay,
                        max_retries=args.max_retries,
                        stable_observations=args.stable_observations,
                        pre_action_check=guard.verify,
                        post_action_check=pointer_check,
                        pause_timeout=args.pause_timeout,
                        pointer_idle_seconds=args.pointer_idle_seconds,
                        pointer_position=controller.pointer_position,
                        release_inputs=controller.release_all,
                        promotion_controller=promotion,
                        promotion_mode=args.promotion_mode,
                        profiler=profile if args.profile else None,
                        logger=print if args.debug else None,
                    )
                    print(agent.run(args.max_moves))
        except KeyboardInterrupt:
            print("Stopped by user")
            return 130
        except (ChessAgentError, BoardDetectionError) as error:
            print(f"Stopped safely: {error}")
            return 2
        finally:
            if args.profile:
                print("Profile:")
                for stage in PROFILE_STAGES:
                    values = timings.get(stage, [])
                    total = sum(values)
                    average = total / len(values) if values else 0.0
                    print(
                        f"  {stage}: total={total:.6f}s; "
                        f"count={len(values)}; avg={average:.6f}s"
                    )
    elif args.command == "chess-dataset":
        from .chess_dataset import generate_dataset
        print(generate_dataset(args.output, args.count, args.seed, variants=args.variants))
    elif args.command == "chess-import":
        from .chess_dataset import import_verified_board
        print(import_verified_board(
            args.dataset, args.image, args.fen, ChessCalibration.load(args.config),
            split=args.split, source_id=args.source_id,
        ))
    elif args.command == "chess-assets":
        from .chess_assets import extract_piece_assets, extract_real_asset_library
        if args.manifest:
            if args.image or args.fen or args.piece_style or args.empty_board:
                raise ValueError("--manifest cannot be combined with single-image asset options")
            print(extract_real_asset_library(
                args.manifest, args.output, mode=args.mode or "real-only",
            ))
        else:
            if not args.image or not args.fen or not args.piece_style:
                raise ValueError("Single-image mode requires --image, --fen and --piece-style")
            print(extract_piece_assets(
                args.image, ChessCalibration.load(args.config), args.fen,
                args.piece_style, args.output, empty_board_path=args.empty_board,
            ))
    elif args.command == "chess-real-dataset":
        from .chess_assets import generate_real_style_dataset
        print(generate_real_style_dataset(
            args.assets, args.output, args.count, args.seed, variants=args.variants,
            mode=args.mode, augmentation=not args.no_augmentation,
        ))
    elif args.command == "chess-train":
        from .chess_training import train_model
        try:
            report = train_model(
                args.dataset, args.checkpoint, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                device_name=args.device, seed=args.seed, patience=args.patience,
                num_workers=args.num_workers, save_epochs=args.save_epochs,
                init_checkpoint=args.init_checkpoint,
                synthetic_dataset=args.synthetic_dataset,
                real_dataset=args.real_dataset, real_ratio=args.real_ratio,
            )
        except (RuntimeError, ValueError) as error:
            print(f"Training failed: {error}")
            return 2
        print(f"Test accuracy: {report['test']['accuracy']:.4f}")
        print(f"Report: {args.checkpoint.rsplit('.', 1)[0]}.report.json")
    elif args.command == "chess-evaluate":
        import json
        from .chess_training import evaluate_backends
        report = evaluate_backends(
            args.dataset, args.model, templates_path=args.templates,
            compare_model_path=args.compare_model, output=args.output
        )
        print(json.dumps(report, indent=2))
    return 0


def chess_learn_main() -> int:
    import sys
    return main(["chess-learn", *sys.argv[1:]])


def chess_scan_main() -> int:
    import sys
    return main(["chess-scan", *sys.argv[1:]])


def chess_play_main() -> int:
    import sys
    return main(["chess-play", *sys.argv[1:]])


def chess_dataset_main() -> int:
    import sys
    return main(["chess-dataset", *sys.argv[1:]])


def chess_assets_main() -> int:
    import sys
    return main(["chess-assets", *sys.argv[1:]])


def chess_real_dataset_main() -> int:
    import sys
    return main(["chess-real-dataset", *sys.argv[1:]])


def chess_train_main() -> int:
    import sys
    return main(["chess-train", *sys.argv[1:]])


def chess_evaluate_main() -> int:
    import sys
    return main(["chess-evaluate", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
