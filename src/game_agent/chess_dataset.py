from __future__ import annotations

import json
import importlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import chess
import cv2
import numpy as np

from .chess import ChessCalibration, Orientation
from .chess_perception import BoardExtractor, NORMALIZED_CELL_SIZE, load_board_image
from .geometry import Region


LABELS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "empty")
DATASET_VERSION = 1
_DLL_HANDLES: list[object] = []


def _prepare_windows_cairo() -> list[str]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    candidates = (
        os.environ.get("CAIRO_DLL_DIR"),
        r"C:\msys64\ucrt64\bin",
        r"C:\Program Files\GTK3-Runtime Win64\bin",
    )
    diagnostics: list[str] = []
    attempted: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in attempted:
            continue
        attempted.add(candidate)
        try:
            if not Path(candidate).is_dir():
                diagnostics.append(f"not found: {candidate}")
                continue
            _DLL_HANDLES.append(os.add_dll_directory(candidate))
            diagnostics.append(f"loaded: {candidate}")
            break
        except (FileNotFoundError, OSError) as error:
            diagnostics.append(f"failed: {candidate} ({error})")
    return diagnostics


def _load_cairosvg():
    try:
        return importlib.import_module("cairosvg")
    except ModuleNotFoundError as error:
        if error.name == "cairosvg":
            raise RuntimeError(
                "CairoSVG is not installed. Run: pip install -e .[ml]"
            ) from error
        initial_error: Exception = error
    except (ImportError, OSError) as error:
        initial_error = error

    diagnostics = _prepare_windows_cairo()
    if os.name == "nt" and diagnostics:
        try:
            return importlib.import_module("cairosvg")
        except (ImportError, OSError) as retry_error:
            final_error: Exception = retry_error
    else:
        final_error = initial_error

    detail = "; ".join(diagnostics) if diagnostics else "no usable DLL directory found"
    hint = (
        "Install a Cairo runtime and set CAIRO_DLL_DIR or "
        "CAIROCFFI_DLL_DIRECTORIES to its bin directory"
        if os.name == "nt"
        else "Install the system Cairo library and the cairosvg Python package"
    )
    raise RuntimeError(
        f"Unable to load CairoSVG: {final_error}. DLL diagnostics: {detail}. {hint}."
    ) from final_error


def _scale_svg_pieces(svg: str, piece_scale: float) -> str:
    def scale_piece(match: re.Match[str]) -> str:
        x, y = float(match.group(2)), float(match.group(3))
        return (
            f'{match.group(1)}transform="translate({x + 22.5:g}, {y + 22.5:g}) '
            f'scale({piece_scale:g}) translate(-22.5, -22.5)"'
        )

    return re.sub(
        r'(<use\b[^>]*?)transform="translate\(([-\d.]+), ([-\d.]+)\)"',
        scale_piece,
        svg,
    )


@dataclass(frozen=True)
class PositionSample:
    game_id: str
    fen: str
    stage: str


def _label(board: chess.Board, square_name: str) -> str:
    piece = board.piece_at(chess.parse_square(square_name))
    return piece.symbol() if piece else "empty"


def random_legal_positions(count: int, seed: int) -> list[PositionSample]:
    """Generate independent legal positions spanning opening, middle and late games."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    stages = (("opening", 2, 18), ("middlegame", 20, 65), ("endgame", 70, 150))
    samples: list[PositionSample] = []
    for index in range(count):
        stage, low, high = stages[index % len(stages)]
        board = chess.Board()
        target = rng.randint(low, high)
        for _ in range(target):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            if stage == "endgame":
                captures = [move for move in moves if board.is_capture(move)]
                if captures and rng.random() < 0.72:
                    moves = captures
            board.push(rng.choice(moves))
        samples.append(PositionSample(f"synthetic-{seed}-{index:06d}", board.fen(), stage))
    return samples


def grouped_split(game_ids: Iterable[str], seed: int) -> dict[str, str]:
    unique = sorted(set(game_ids))
    random.Random(seed).shuffle(unique)
    total = len(unique)
    validation = max(1, round(total * 0.1)) if total >= 3 else 0
    test = max(1, round(total * 0.1)) if total >= 3 else 0
    train_end = total - validation - test
    return {
        game_id: "train" if index < train_end else "validation" if index < train_end + validation else "test"
        for index, game_id in enumerate(unique)
    }


class SvgBoardRenderer:
    def __call__(
        self,
        board: chess.Board,
        size: int,
        light: str,
        dark: str,
        piece_scale: float,
    ) -> np.ndarray:
        cairosvg = _load_cairosvg()
        try:
            import chess.svg
        except ImportError as error:
            raise RuntimeError("python-chess SVG support is unavailable") from error
        colors = {"square light": light, "square dark": dark}
        svg = chess.svg.board(board, size=size, colors=colors, coordinates=False)
        if piece_scale != 1.0:
            svg = _scale_svg_pieces(svg, piece_scale)
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Unable to decode rendered Chess board")
        return image
def _write_board_samples(
    output: Path,
    board_image: np.ndarray,
    board: chess.Board,
    *,
    board_id: str,
    game_id: str,
    split: str,
    source: str,
    stage: str,
    orientation: Orientation,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    side_h, side_w = board_image.shape[:2]
    if side_h != side_w:
        raise ValueError("Board image must be square")
    calibration = ChessCalibration(Region(0, 0, side_w, side_h), orientation)
    tiles = BoardExtractor(calibration).squares(board_image)
    board_path = Path("boards") / split / f"{board_id}.png"
    absolute_board = output / board_path
    absolute_board.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(absolute_board), board_image)
    rows: list[dict[str, object]] = []
    for square, tile in tiles.items():
        label = _label(board, square)
        tile_path = Path("squares") / split / label / f"{board_id}-{square}.png"
        absolute_tile = output / tile_path
        absolute_tile.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(absolute_tile), tile):
            raise RuntimeError(f"Unable to write {absolute_tile}")
        rows.append({
            "path": tile_path.as_posix(), "board_path": board_path.as_posix(),
            "label": label, "square": square, "square_color": (chess.square_rank(chess.parse_square(square)) + chess.square_file(chess.parse_square(square))) % 2,
            "board_id": board_id, "game_id": game_id, "split": split,
            "source": source, "stage": stage, "fen": board.fen(),
            "orientation": orientation.value, **metadata,
        })
    return rows


def _save_manifest(output: Path, rows: list[dict[str, object]], seed: int) -> Path:
    counts = {label: sum(row["label"] == label for row in rows) for label in LABELS}
    for row in rows:
        row["balanced_weight"] = len(rows) / (len(LABELS) * max(1, counts[str(row["label"])]))
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    info = {
        "version": DATASET_VERSION, "labels": LABELS, "input_size": NORMALIZED_CELL_SIZE,
        "seed": seed, "samples": len(rows), "class_counts": counts,
    }
    (output / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return manifest


def generate_dataset(
    output: str | Path,
    count: int,
    seed: int,
    *,
    variants: int = 2,
    renderer: Callable[[chess.Board, int, str, str, float], np.ndarray] | None = None,
) -> Path:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    positions = random_legal_positions(count, seed)
    splits = grouped_split((sample.game_id for sample in positions), seed)
    rng = random.Random(seed)
    render = renderer or SvgBoardRenderer()
    palettes = (("#f0d9b5", "#b58863"), ("#eeeed2", "#769656"), ("#dee3e6", "#8ca2ad"))
    rows: list[dict[str, object]] = []
    for sample in positions:
        board = chess.Board(sample.fen)
        for variant in range(variants):
            size = rng.randint(512, 1024)
            light, dark = rng.choice(palettes)
            piece_scale = rng.uniform(0.90, 1.05)
            image = render(board, size, light, dark, piece_scale)
            rows.extend(_write_board_samples(
                output_path, image, board,
                board_id=f"{sample.game_id}-v{variant}", game_id=sample.game_id,
                split=splits[sample.game_id], source="synthetic-svg", stage=sample.stage,
                orientation=Orientation.WHITE_BOTTOM,
                metadata={"render_size": size, "light": light, "dark": dark, "piece_scale": piece_scale},
            ))
    return _save_manifest(output_path, rows, seed)


def import_verified_board(
    dataset: str | Path,
    image_path: str | Path,
    fen: str,
    calibration: ChessCalibration,
    *,
    split: str = "test",
    source_id: str = "verified-real",
) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    try:
        board = chess.Board(fen if " " in fen else f"{fen} w - - 0 1")
    except ValueError as error:
        raise ValueError("A valid user-confirmed FEN is required") from error
    image = load_board_image(image_path)
    board_image = BoardExtractor(calibration).board_image(image)
    output = Path(dataset)
    output.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", source_id)
    rows = _write_board_samples(
        output, board_image, board, board_id=safe_id, game_id=safe_id,
        split=split, source="verified-real", stage="user-confirmed",
        orientation=calibration.orientation, metadata={"original_image": str(image_path)},
    )
    manifest = output / "manifest.jsonl"
    existing = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()] if manifest.exists() else []
    return _save_manifest(output, existing + rows, seed=0)
