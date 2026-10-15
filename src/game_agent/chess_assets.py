from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np

from .chess import ChessCalibration, Orientation
from .chess_dataset import LABELS, _save_manifest, _write_board_samples, grouped_split, random_legal_positions
from .chess_perception import BoardExtractor, load_board_image
from .geometry import Region


ASSET_VERSION = 2


def _board(fen: str) -> chess.Board:
    try:
        return chess.Board(fen if " " in fen else f"{fen} w - - 0 1")
    except ValueError as error:
        raise ValueError("A valid user-confirmed FEN is required") from error


def _square_color(square: str) -> str:
    parsed = chess.parse_square(square)
    return "light" if (chess.square_rank(parsed) + chess.square_file(parsed)) % 2 else "dark"


def _backgrounds(tiles: dict[str, np.ndarray], board: chess.Board) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for color in ("light", "dark"):
        samples = [
            tile for square, tile in tiles.items()
            if board.piece_at(chess.parse_square(square)) is None and _square_color(square) == color
        ]
        if not samples:
            raise ValueError(f"No empty {color} square is available for background estimation")
        result[color] = np.median(np.stack(samples), axis=0).astype(np.uint8)
    return result


def _foreground(tile: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    tile_lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB).astype(np.int16)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.int16)
    distance = np.linalg.norm(tile_lab - background_lab, axis=2)
    border = np.concatenate((distance[0], distance[-1], distance[:, 0], distance[:, -1]))
    threshold = max(12.0, float(np.percentile(border, 95)) + 5.0)
    mask = (distance >= threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros_like(mask)
    minimum = max(8, round(mask.size * 0.002))
    center = np.asarray(mask.shape[::-1], dtype=float) / 2
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        component_center = np.asarray((x + width / 2, y + height / 2))
        if area >= minimum and np.linalg.norm(component_center - center) <= mask.shape[0] * 0.65:
            kept[labels == index] = 255
    alpha = cv2.GaussianBlur(kept, (3, 3), 0)
    coverage = float(np.count_nonzero(alpha > 16) / alpha.size)
    touches = float(np.mean(np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1])) > 16))
    quality = max(0.0, min(1.0, 1.0 - abs(coverage - 0.34) - touches * 2.0))
    accepted = 0.05 <= coverage <= 0.72 and touches <= 0.08
    rgba = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, {"coverage": coverage, "border_fraction": touches, "quality": quality, "accepted": accepted}


def _empty_square_mask(board: chess.Board, side: int, margin_fraction: float = 0.12) -> np.ndarray:
    mask = np.zeros((side, side), np.uint8)
    margin = max(0, round(side / 8 * margin_fraction))
    for row in range(8):
        for column in range(8):
            square = chess.square(column, 7 - row)
            if board.piece_at(square) is not None:
                continue
            left, right = round(column * side / 8) + margin, round((column + 1) * side / 8) - margin
            top, bottom = round(row * side / 8) + margin, round((row + 1) * side / 8) - margin
            mask[top:bottom, left:right] = 255
    return mask


def _align_empty_board(
    occupied: np.ndarray, empty: np.ndarray, board: chess.Board,
) -> tuple[np.ndarray, dict[str, object]]:
    if occupied.shape[0] != occupied.shape[1] or empty.shape[0] != empty.shape[1]:
        raise ValueError("Occupied and empty boards must both be square")
    scale = occupied.shape[0] / empty.shape[0]
    if not 0.98 <= scale <= 1.02:
        raise ValueError(f"Empty board scale differs too much ({scale:.4f}); capture both boards at the same size")
    if empty.shape[:2] != occupied.shape[:2]:
        empty = cv2.resize(empty, occupied.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    side = occupied.shape[0]
    alignment_mask = _empty_square_mask(board, side, 0.0)
    comparison_mask = _empty_square_mask(board, side)
    occupied_gray = cv2.cvtColor(occupied, cv2.COLOR_BGR2GRAY)
    empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        correlation, warp = cv2.findTransformECC(
            occupied_gray, empty_gray, warp, cv2.MOTION_TRANSLATION,
            criteria, alignment_mask, 3,
        )
    except cv2.error as error:
        raise ValueError("Unable to align the empty board; recapture both images without moving or resizing the board") from error
    dx, dy = float(warp[0, 2]), float(warp[1, 2])
    maximum_shift = max(2.0, side * 0.012)
    if abs(dx) > maximum_shift or abs(dy) > maximum_shift:
        raise ValueError(f"Empty board alignment shift is too large ({dx:.2f}, {dy:.2f} pixels)")
    aligned = cv2.warpAffine(
        empty, warp, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    difference = cv2.absdiff(occupied, aligned)
    values = difference[comparison_mask > 0]
    mean_difference = float(np.mean(values)) if values.size else float("inf")
    percentile_99 = float(np.percentile(values, 99)) if values.size else float("inf")
    if correlation < 0.90 or mean_difference > 10.0 or percentile_99 > 35.0:
        raise ValueError(
            "Empty board does not match the occupied board theme or alignment "
            f"(correlation={correlation:.3f}, mean difference={mean_difference:.2f})"
        )
    return aligned, {
        "method": "ecc-translation", "correlation": float(correlation),
        "translation": [dx, dy], "input_scale": scale,
        "empty_mean_difference": mean_difference, "empty_p99_difference": percentile_99,
    }


def _paired_foreground(
    tile: np.ndarray, background: np.ndarray, noise_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    observed = tile.astype(np.float32)
    base = background.astype(np.float32)
    distance = np.linalg.norm(observed - base, axis=2).astype(np.float32)
    background_threshold = max(3.0, float(noise_threshold))
    foreground_threshold = max(background_threshold + 8.0, background_threshold * 2.0)
    probable = distance > background_threshold
    raw_foreground = distance >= foreground_threshold

    # Keep every meaningful component that contains confident foreground. This retains
    # detached crowns/crosses while discarding compression speckles without assuming
    # that a chess piece has only one connected component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(probable.astype(np.uint8), 8)
    support = np.zeros_like(probable)
    discarded = 0
    minimum = max(2, round(probable.size * 0.0003))
    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= minimum and np.any(raw_foreground & component):
            support |= component
        else:
            discarded += area

    local_maximum = cv2.dilate(distance, np.ones((5, 5), np.uint8))
    alpha = np.divide(
        distance, np.maximum(local_maximum, foreground_threshold),
        out=np.zeros_like(distance), where=support,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    sure_foreground = support & raw_foreground & (alpha >= 0.98)
    sure_background = ~support
    filtered = cv2.bilateralFilter(alpha, 5, 0.12, 3.0)
    unknown = support & ~sure_foreground
    alpha[unknown] = filtered[unknown]
    alpha[sure_foreground] = 1.0
    alpha[sure_background] = 0.0
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    trimap = np.zeros_like(distance, np.uint8)
    trimap[unknown] = 128
    trimap[sure_foreground] = 255

    # Solve straight (unpremultiplied) foreground color only where alpha is stable.
    # Low-alpha pixels receive the nearest reliable foreground color instead of an
    # unstable division that amplifies board color and compression noise.
    reliable = support & (alpha >= 0.20)
    alpha_3 = alpha[:, :, None]
    raw_solution = np.zeros_like(observed)
    raw_solution[reliable] = (
        observed[reliable] - (1.0 - alpha_3[reliable]) * base[reliable]
    ) / alpha_3[reliable]
    clipped = np.any((raw_solution < -5.0) | (raw_solution > 260.0), axis=2) & reliable
    foreground = np.clip(raw_solution, 0.0, 255.0)
    uncertain_coordinates = np.argwhere(support & ~reliable)
    reliable_coordinates = np.argwhere(reliable)
    for start in range(0, len(uncertain_coordinates), 256):
        targets = uncertain_coordinates[start:start + 256]
        if not len(reliable_coordinates):
            break
        offsets = targets[:, None, :] - reliable_coordinates[None, :, :]
        nearest = np.argmin(np.sum(offsets * offsets, axis=2), axis=1)
        foreground[targets[:, 0], targets[:, 1]] = foreground[
            reliable_coordinates[nearest, 0], reliable_coordinates[nearest, 1]
        ]
    foreground[~support] = 0.0

    alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
    rgba = cv2.cvtColor(foreground.astype(np.uint8), cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_u8
    raw_rgba = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
    raw_rgba[:, :, 3] = alpha_u8
    reconstruction = np.clip(
        foreground * alpha_3 + base * (1.0 - alpha_3), 0, 255
    ).astype(np.uint8)

    coverage = float(np.mean(alpha > 0.02))
    border = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
    border_fraction = float(np.mean(border > 0.02))
    missed_change = float(np.mean((distance >= foreground_threshold) & (alpha <= 0.02)))
    component_count = int(sum(
        1 for index in range(1, count) if np.any(support & (labels == index))
    ))
    isolated_fraction = float(discarded / max(1, np.count_nonzero(probable)))
    soft = support & (alpha > 0.02) & (alpha < 0.98)
    soft_fraction = float(np.count_nonzero(soft) / max(1, np.count_nonzero(support)))
    reconstruction_mae = float(np.mean(cv2.absdiff(tile, reconstruction)))
    clip_fraction = float(np.count_nonzero(clipped) / max(1, np.count_nonzero(reliable)))
    edge = support & (alpha > 0.02) & (alpha < 0.98)
    reliable_u8 = reliable.astype(np.uint8)
    local_has_color = cv2.dilate(reliable_u8, np.ones((7, 7), np.uint8)) > 0
    local_maximum_color = cv2.dilate(
        np.where(reliable[:, :, None], foreground, 0.0).astype(np.float32),
        np.ones((7, 7), np.uint8),
    )
    local_minimum_color = cv2.erode(
        np.where(reliable[:, :, None], foreground, 255.0).astype(np.float32),
        np.ones((7, 7), np.uint8),
    )
    color_outlier = edge & local_has_color & np.any(
        (foreground < local_minimum_color - 20.0) | (foreground > local_maximum_color + 20.0),
        axis=2,
    )
    edge_color_outlier = float(np.count_nonzero(color_outlier) / max(1, np.count_nonzero(edge & local_has_color)))
    accepted = bool(
        0.05 <= coverage <= 0.72 and border_fraction <= 0.08
        and component_count >= 1 and missed_change <= 0.015
        and isolated_fraction <= 0.05 and reconstruction_mae <= 12.0
        and edge_color_outlier <= 0.15
        and np.isfinite(foreground).all() and np.isfinite(alpha).all()
    )
    quality = max(0.0, min(
        1.0, 1.0 - abs(coverage - 0.34) - border_fraction * 2.0
        - missed_change * 4.0 - isolated_fraction * 2.0 - reconstruction_mae / 80.0,
    ))
    difference_preview = np.clip(
        distance / max(1.0, float(np.percentile(distance, 99))) * 255.0, 0, 255
    ).astype(np.uint8)
    diagnostics = {
        "difference": difference_preview, "trimap": trimap,
        "alpha": alpha_u8, "raw_rgba": raw_rgba,
    }
    return rgba, reconstruction, {
        "coverage": coverage, "border_fraction": border_fraction,
        "component_count": component_count, "missed_change_fraction": missed_change,
        "isolated_alpha_fraction": isolated_fraction, "soft_alpha_fraction": soft_fraction,
        "decontamination_clip_fraction": clip_fraction,
        "edge_color_outlier_fraction": edge_color_outlier,
        "reconstruction_mae": reconstruction_mae, "quality": quality,
        "accepted": accepted, "background_threshold": background_threshold,
        "foreground_threshold": foreground_threshold,
    }, diagnostics


def extract_piece_assets(
    image_path: str | Path, calibration: ChessCalibration, fen: str,
    piece_style: str, output: str | Path, *, empty_board_path: str | Path | None = None,
) -> Path:
    board = _board(fen)
    image = load_board_image(image_path)
    board_image = BoardExtractor(calibration).board_image(image)
    local = ChessCalibration(Region(0, 0, board_image.shape[1], board_image.shape[0]), calibration.orientation)
    tiles = BoardExtractor(local).squares(board_image)
    alignment: dict[str, object] | None = None
    if empty_board_path:
        empty_image = load_board_image(empty_board_path)
        empty_board = BoardExtractor(calibration).board_image(empty_image)
        empty_board, alignment = _align_empty_board(board_image, empty_board, board)
        empty_tiles = BoardExtractor(local).squares(empty_board)
        backgrounds = {
            color: np.median(np.stack([
                tile for square, tile in empty_tiles.items() if _square_color(square) == color
            ]), axis=0).astype(np.uint8)
            for color in ("light", "dark")
        }
        empty_differences = []
        for square, tile in tiles.items():
            if board.piece_at(chess.parse_square(square)) is None:
                empty_differences.append(cv2.absdiff(tile, empty_tiles[square]))
        noise_threshold = max(7.0, float(np.percentile(np.concatenate([
            value.reshape(-1, 3) for value in empty_differences
        ]), 99.5)) * 1.5)
    else:
        empty_tiles = None
        noise_threshold = 0.0
        backgrounds = _backgrounds(tiles, board)
    destination = Path(output)
    (destination / "pieces").mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir(parents=True, exist_ok=True)
    (destination / "trimaps").mkdir(parents=True, exist_ok=True)
    (destination / "differences").mkdir(parents=True, exist_ok=True)
    (destination / "backgrounds").mkdir(parents=True, exist_ok=True)
    for color, background in backgrounds.items():
        cv2.imwrite(str(destination / "backgrounds" / f"{color}.png"), background)

    assets: list[dict[str, object]] = []
    preview_items: list[dict[str, np.ndarray | str]] = []
    for square, tile in tiles.items():
        piece = board.piece_at(chess.parse_square(square))
        if piece is None:
            continue
        color = _square_color(square)
        actual_background = empty_tiles[square] if empty_tiles is not None else backgrounds[color]
        if empty_tiles is not None:
            rgba, reconstruction, quality, diagnostics = _paired_foreground(
                tile, actual_background, noise_threshold
            )
            extraction = "paired-empty-board"
        else:
            rgba, quality = _foreground(tile, actual_background)
            reconstruction = _alpha_composite(actual_background, rgba)
            quality.update({
                "component_count": None, "missed_change_fraction": None,
                "reconstruction_mae": float(np.mean(cv2.absdiff(tile, reconstruction))),
            })
            diagnostics = {
                "difference": cv2.cvtColor(cv2.absdiff(tile, actual_background), cv2.COLOR_BGR2GRAY),
                "trimap": np.where(rgba[:, :, 3] > 240, 255, np.where(rgba[:, :, 3] > 0, 128, 0)).astype(np.uint8),
                "alpha": rgba[:, :, 3], "raw_rgba": rgba.copy(),
            }
            extraction = "single-image"
        name = f"{piece.symbol()}_{square}.png"
        path = destination / "pieces" / name
        if not cv2.imwrite(str(path), rgba):
            raise RuntimeError(f"Unable to write {path}")
        mask_path = destination / "masks" / name
        if not cv2.imwrite(str(mask_path), rgba[:, :, 3]):
            raise RuntimeError(f"Unable to write {mask_path}")
        trimap_path = destination / "trimaps" / name
        difference_path = destination / "differences" / name
        if not cv2.imwrite(str(trimap_path), diagnostics["trimap"]):
            raise RuntimeError(f"Unable to write {trimap_path}")
        if not cv2.imwrite(str(difference_path), diagnostics["difference"]):
            raise RuntimeError(f"Unable to write {difference_path}")
        assets.append({
            "label": piece.symbol(), "square": square, "square_color": color,
            "path": f"pieces/{name}", "mask_path": f"masks/{name}",
            "trimap_path": f"trimaps/{name}", "difference_path": f"differences/{name}",
            "source_image": str(image_path), "empty_board": str(empty_board_path) if empty_board_path else None,
            "extraction": extraction, **quality,
        })
        heatmap = cv2.applyColorMap(
            cv2.cvtColor(cv2.absdiff(tile, reconstruction), cv2.COLOR_BGR2GRAY),
            cv2.COLORMAP_TURBO,
        )
        preview_items.append({
            "label": piece.symbol(), "square": square, "original": tile, "background": actual_background,
            "rgba": rgba, "raw_rgba": diagnostics["raw_rgba"],
            "difference": diagnostics["difference"], "trimap": diagnostics["trimap"],
            "alpha": diagnostics["alpha"], "reconstruction": reconstruction, "heatmap": heatmap,
        })
    missing = sorted(set(LABELS[:-1]) - {str(item["label"]) for item in assets})
    if missing:
        raise ValueError(f"Source board does not contain all piece labels: {', '.join(missing)}")

    metadata = {
        "version": ASSET_VERSION, "piece_style": piece_style,
        "source_image": str(image_path), "source_fen": board.board_fen(),
        "orientation": calibration.orientation.value, "labels": LABELS,
        "empty_assets": {color: f"backgrounds/{color}.png" for color in backgrounds},
        "empty_board": str(empty_board_path) if empty_board_path else None,
        "alignment": alignment, "noise_threshold": noise_threshold,
        "preview_layout": {
            "asset-preview.png": ["original", "difference", "trimap", "soft-alpha", "raw-rgb-on-checker", "decontaminated-on-checker"],
            "compositing-preview.png": ["raw-light", "raw-dark", "decontaminated-light", "decontaminated-dark"],
            "reconstruction-preview.png": ["original", "reconstruction", "difference-heatmap"],
        },
        "assets": assets,
    }
    manifest = destination / "assets.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _asset_previews(destination, preview_items, backgrounds)
    return manifest


def _alpha_composite(background: np.ndarray, foreground: np.ndarray, scale: float = 1.0,
                     offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    result = background.copy()
    if scale != 1.0:
        side = max(1, round(foreground.shape[0] * scale))
        foreground = cv2.resize(foreground, (side, side), interpolation=cv2.INTER_AREA)
    x = (result.shape[1] - foreground.shape[1]) // 2 + offset[0]
    y = (result.shape[0] - foreground.shape[0]) // 2 + offset[1]
    left, top = max(0, x), max(0, y)
    right, bottom = min(result.shape[1], x + foreground.shape[1]), min(result.shape[0], y + foreground.shape[0])
    if left >= right or top >= bottom:
        return result
    source = foreground[top - y:bottom - y, left - x:right - x]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    result[top:bottom, left:right] = np.clip(
        source[:, :, :3] * alpha + result[top:bottom, left:right] * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    return result


def _checkerboard(side: int) -> np.ndarray:
    checker = np.empty((side, side, 3), np.uint8)
    block = max(4, side // 8)
    for y in range(0, side, block):
        for x in range(0, side, block):
            checker[y:y + block, x:x + block] = 210 if (x // block + y // block) % 2 else 150
    return checker


def _asset_previews(
    output: Path, items: list[dict[str, np.ndarray | str]], backgrounds: dict[str, np.ndarray],
) -> None:
    asset_rows, composite_rows, reconstruction_rows = [], [], []
    ordered = sorted(items, key=lambda entry: (LABELS.index(str(entry["label"])), str(entry["square"])))
    for item in ordered:
        rgba = item["rgba"]
        difference = cv2.cvtColor(item["difference"], cv2.COLOR_GRAY2BGR)
        trimap = cv2.cvtColor(item["trimap"], cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(item["alpha"], cv2.COLOR_GRAY2BGR)
        raw_transparent = _alpha_composite(_checkerboard(rgba.shape[0]), item["raw_rgba"])
        transparent = _alpha_composite(_checkerboard(rgba.shape[0]), rgba)
        asset_rows.append(np.hstack((
            item["original"], difference, trimap, mask, raw_transparent, transparent,
        )))
        composite_rows.append(np.hstack((
            _alpha_composite(backgrounds["light"], item["raw_rgba"]),
            _alpha_composite(backgrounds["dark"], item["raw_rgba"]),
            _alpha_composite(backgrounds["light"], rgba),
            _alpha_composite(backgrounds["dark"], rgba),
        )))
        reconstruction_rows.append(np.hstack((
            item["original"], item["reconstruction"], item["heatmap"],
        )))
    cv2.imwrite(str(output / "asset-preview.png"), np.vstack(asset_rows))
    cv2.imwrite(str(output / "compositing-preview.png"), np.vstack(composite_rows))
    cv2.imwrite(str(output / "reconstruction-preview.png"), np.vstack(reconstruction_rows))


def _manifest_path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Asset manifest requires {field}")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _safe_source_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Every asset source requires a non-empty source_id")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", value)
    if not safe:
        raise ValueError(f"Invalid source_id: {value!r}")
    return safe


def _real_asset_preview(output: Path, assets: list[dict[str, object]]) -> None:
    rows: list[np.ndarray] = []
    for label in LABELS[:-1]:
        cells: list[np.ndarray] = []
        for color in ("light", "dark"):
            item = next(
                entry for entry in assets
                if entry["label"] == label and entry["square_color"] == color
            )
            tile = cv2.imread(str(output / str(item["path"])), cv2.IMREAD_COLOR)
            if tile is None:
                raise ValueError(f"Unreadable real asset: {item['path']}")
            cell = tile.copy()
            cv2.rectangle(cell, (0, 0), (cell.shape[1], 17), (0, 0, 0), -1)
            cv2.putText(
                cell, f"{label} {color} {item['source_id']}", (3, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.hstack(cells))
    cv2.imwrite(str(output / "real-asset-preview.png"), np.vstack(rows))


def extract_real_asset_library(
    manifest_path: str | Path, output: str | Path, *, mode: str = "real-only",
) -> Path:
    if mode != "real-only":
        raise ValueError("Multi-source asset manifests currently require mode real-only")
    manifest_file = Path(manifest_path)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    base = manifest_file.parent
    piece_style = data.get("piece_style")
    if not isinstance(piece_style, str) or not piece_style:
        raise ValueError("Asset manifest requires piece_style")
    calibration_path = _manifest_path(base, data.get("config"), "config")
    empty_path = _manifest_path(base, data.get("empty_board"), "empty_board")
    calibration = ChessCalibration.load(calibration_path)
    empty_image = load_board_image(empty_path)
    empty_board = BoardExtractor(calibration).board_image(empty_image)
    if empty_board.shape[0] != empty_board.shape[1]:
        raise ValueError("Empty board must be square")
    local = ChessCalibration(
        Region(0, 0, empty_board.shape[1], empty_board.shape[0]), calibration.orientation,
    )
    empty_tiles = BoardExtractor(local).squares(empty_board)

    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Asset manifest requires at least one source")
    destination = Path(output)
    (destination / "tiles" / "empty").mkdir(parents=True, exist_ok=True)
    (destination / "backgrounds").mkdir(parents=True, exist_ok=True)

    empty_assets: dict[str, list[dict[str, object]]] = {"light": [], "dark": []}
    for square, tile in empty_tiles.items():
        color = _square_color(square)
        relative = Path("tiles") / "empty" / color / f"blank-{square}.png"
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination / relative), tile):
            raise RuntimeError(f"Unable to write {destination / relative}")
        empty_assets[color].append({
            "label": "empty", "square": square, "square_color": color,
            "path": relative.as_posix(), "source_id": "empty-board",
            "source_image": str(empty_path), "source_fen": None,
            "real": True, "valid": True, "image_size": list(tile.shape[:2][::-1]),
        })
    backgrounds = {
        color: np.median(np.stack([
            empty_tiles[str(item["square"])] for item in empty_assets[color]
        ]), axis=0).astype(np.uint8)
        for color in ("light", "dark")
    }
    for color, background in backgrounds.items():
        cv2.imwrite(str(destination / "backgrounds" / f"{color}.png"), background)

    assets: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every asset source must be an object")
        source_id = _safe_source_id(source.get("source_id"))
        if source_id in source_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")
        source_ids.add(source_id)
        image_path = _manifest_path(base, source.get("image"), f"image for {source_id}")
        board = _board(str(source.get("fen", "")))
        source_image = load_board_image(image_path)
        board_image = BoardExtractor(calibration).board_image(source_image)
        if board_image.shape != empty_board.shape:
            raise ValueError(
                f"Source {source_id} board size {board_image.shape[:2]} does not match "
                f"empty board {empty_board.shape[:2]}"
            )
        _, alignment = _align_empty_board(board_image, empty_board, board)
        tiles = BoardExtractor(local).squares(board_image)
        source_records.append({
            "source_id": source_id, "image": str(image_path),
            "fen": board.board_fen(), "alignment": alignment,
        })
        for square, tile in tiles.items():
            piece = board.piece_at(chess.parse_square(square))
            if piece is None:
                continue
            label = piece.symbol()
            color = _square_color(square)
            relative = Path("tiles") / label / color / f"{source_id}-{label}-{square}.png"
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination / relative), tile):
                raise RuntimeError(f"Unable to write {destination / relative}")
            assets.append({
                "label": label, "square": square, "square_color": color,
                "path": relative.as_posix(), "source_id": source_id,
                "source_image": str(image_path), "source_fen": board.board_fen(),
                "real": True, "valid": True, "accepted": True,
                "image_size": list(tile.shape[:2][::-1]),
            })

    counts = {
        label: {
            color: sum(
                item["label"] == label and item["square_color"] == color
                for item in assets
            )
            for color in ("light", "dark")
        }
        for label in LABELS[:-1]
    }
    missing = [
        f"{label}/{color}" for label, colors in counts.items()
        for color, count in colors.items() if count == 0
    ]
    statistics = {
        "counts": counts, "missing": missing,
        "source_counts": {
            source_id: sum(item["source_id"] == source_id for item in assets)
            for source_id in sorted(source_ids)
        },
    }
    (destination / "asset-statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8",
    )
    if missing:
        raise ValueError(
            "Real-only asset library is incomplete; missing true tiles for: "
            + ", ".join(missing)
        )
    metadata = {
        "version": ASSET_VERSION, "mode": "real-only", "piece_style": piece_style,
        "orientation": calibration.orientation.value, "labels": LABELS,
        "manifest": str(manifest_file), "empty_board": str(empty_path),
        "source_image": [record["image"] for record in source_records],
        "sources": source_records, "assets": assets, "empty_assets": empty_assets,
        "statistics": statistics,
    }
    result = destination / "assets.json"
    result.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _real_asset_preview(destination, assets)
    return result


@dataclass
class AssetLibrary:
    root: Path
    metadata: dict[str, object]
    backgrounds: dict[str, np.ndarray]
    assets: dict[str, dict[str, list[dict[str, object]]]]
    empty_assets: dict[str, list[dict[str, object]]]

    @classmethod
    def load(cls, root: str | Path) -> "AssetLibrary":
        path = Path(root)
        metadata = json.loads((path / "assets.json").read_text(encoding="utf-8"))
        backgrounds = {
            color: cv2.imread(str(path / "backgrounds" / f"{color}.png"))
            for color in ("light", "dark")
        }
        if any(value is None for value in backgrounds.values()):
            raise ValueError("Asset library backgrounds are missing or unreadable")
        assets = {label: {"light": [], "dark": []} for label in LABELS[:-1]}
        for item in metadata.get("assets", []):
            if item.get("accepted") or item.get("valid"):
                assets[str(item["label"])][str(item["square_color"])].append(item)
        mode = str(metadata.get("mode", "composite-allowed"))
        empty_assets: dict[str, list[dict[str, object]]] = {"light": [], "dark": []}
        if mode == "real-only":
            raw_empty = metadata.get("empty_assets", {})
            if isinstance(raw_empty, dict):
                for color in empty_assets:
                    values = raw_empty.get(color, [])
                    if isinstance(values, list):
                        empty_assets[color].extend(values)
        missing = [
            f"{label}/{color}" if mode == "real-only" else label
            for label, colors in assets.items()
            for color in (("light", "dark") if mode == "real-only" else ("all",))
            if not (colors[color] if color != "all" else colors["light"] + colors["dark"])
        ]
        if mode == "real-only":
            missing.extend(f"empty/{color}" for color, items in empty_assets.items() if not items)
        if missing:
            raise ValueError(f"No accepted assets for: {', '.join(missing)}; review asset previews")
        return cls(path, metadata, backgrounds, assets, empty_assets)

    def render(
        self, board: chess.Board, size: int, rng: random.Random, *,
        mode: str | None = None, return_usage: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, dict[str, object]]]:
        render_mode = mode or str(self.metadata.get("mode", "composite-allowed"))
        if render_mode not in {"real-only", "composite-allowed"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        library_mode = str(self.metadata.get("mode", "composite-allowed"))
        if render_mode != library_mode:
            raise ValueError(
                f"Asset library mode is {library_mode}; it cannot render as {render_mode}"
            )
        cell = size // 8
        size = cell * 8
        output = np.empty((size, size, 3), np.uint8)
        usage: dict[str, dict[str, object]] = {}
        for row in range(8):
            for column in range(8):
                square = chess.square(column, 7 - row)
                name = chess.square_name(square)
                color = _square_color(name)
                piece = board.piece_at(square)
                if render_mode == "real-only":
                    candidates = self.assets[piece.symbol()][color] if piece else self.empty_assets[color]
                    if not candidates:
                        label = piece.symbol() if piece else "empty"
                        raise ValueError(f"Missing real-only asset for {label}/{color}")
                    item = rng.choice(candidates)
                    tile = cv2.imread(str(self.root / str(item["path"])), cv2.IMREAD_COLOR)
                    if tile is None:
                        raise ValueError(f"Unreadable real-only asset: {item['path']}")
                    tile = cv2.resize(tile, (cell, cell), interpolation=cv2.INTER_AREA)
                else:
                    background = cv2.resize(self.backgrounds[color], (cell, cell), interpolation=cv2.INTER_AREA)
                    tile = background
                    item = {"source_id": "empty", "path": f"backgrounds/{color}.png"}
                    if piece:
                        candidates = self.assets[piece.symbol()]["light"] + self.assets[piece.symbol()]["dark"]
                        item = rng.choice(candidates)
                        foreground = cv2.imread(str(self.root / str(item["path"])), cv2.IMREAD_UNCHANGED)
                        foreground = cv2.resize(foreground, (cell, cell), interpolation=cv2.INTER_AREA)
                        tile = _alpha_composite(
                            background, foreground, rng.uniform(0.96, 1.03),
                            (rng.randint(-2, 2), rng.randint(-2, 2)),
                        )
                output[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = tile
                usage[name] = {
                    "path": str(item["path"]), "source_id": item.get("source_id"),
                    "source_image": item.get("source_image"), "square_color": color,
                }
        return (output, usage) if return_usage else output


def _augment_board(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, object]]:
    contrast = rng.uniform(0.94, 1.06)
    brightness = rng.uniform(-8.0, 8.0)
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    saturation = rng.uniform(0.94, 1.06)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blur = rng.choice((0, 0, 3))
    if blur:
        adjusted = cv2.GaussianBlur(adjusted, (blur, blur), 0)
    jpeg_quality = rng.randint(82, 100)
    encoded, buffer = cv2.imencode(".jpg", adjusted, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded:
        raise RuntimeError("Unable to encode composite board")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Unable to decode composite board")
    return decoded, {
        "contrast": contrast, "brightness": brightness, "saturation": saturation,
        "blur_kernel": blur, "jpeg_quality": jpeg_quality,
    }


def _augment_real_board(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, object]]:
    """Apply only whole-board degradations; preserve the Chess.com palette."""
    blur = rng.choice((0, 0, 3))
    adjusted = cv2.GaussianBlur(image, (blur, blur), 0) if blur else image.copy()
    scale = rng.uniform(0.90, 1.0)
    if scale < 0.995:
        side = max(8, round(adjusted.shape[0] * scale))
        adjusted = cv2.resize(adjusted, (side, side), interpolation=cv2.INTER_AREA)
        adjusted = cv2.resize(adjusted, image.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    jpeg_quality = rng.randint(86, 100)
    encoded, buffer = cv2.imencode(".jpg", adjusted, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded:
        raise RuntimeError("Unable to encode real-only board")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Unable to decode real-only board")
    return decoded, {"blur_kernel": blur, "resolution_scale": scale, "jpeg_quality": jpeg_quality}


def generate_real_style_dataset(
    assets: str | Path, output: str | Path, count: int, seed: int, *, variants: int = 2,
    mode: str | None = None, augmentation: bool = True,
) -> Path:
    library = AssetLibrary.load(assets)
    mode = mode or str(library.metadata.get("mode", "composite-allowed"))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    positions = random_legal_positions(count, seed)
    splits = grouped_split((sample.game_id for sample in positions), seed)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    preview = destination / "previews"
    preview.mkdir(exist_ok=True)
    for sample_index, sample in enumerate(positions):
        board = chess.Board(sample.fen)
        for variant in range(variants):
            size = 768 if mode == "real-only" else rng.randint(512, 896)
            image, usage = library.render(board, size, rng, mode=mode, return_usage=True)
            if augmentation:
                image, augmentation_info = (
                    _augment_real_board(image, rng) if mode == "real-only"
                    else _augment_board(image, rng)
                )
            else:
                augmentation_info = None
            board_id = f"real-style-{seed}-{sample_index:06d}-v{variant}"
            rows.extend(_write_board_samples(
                destination, image, board, board_id=board_id, game_id=sample.game_id,
                split=splits[sample.game_id], source="real-style-composite", stage=sample.stage,
                orientation=Orientation.WHITE_BOTTOM,
                metadata={
                    "piece_style": library.metadata["piece_style"],
                    "asset_version": library.metadata["version"],
                    "asset_source": library.metadata["source_image"], "seed": seed,
                    "augmentation": augmentation_info, "render_mode": mode,
                },
            ))
            for row in rows[-64:]:
                row["asset_used"] = usage[str(row["square"])]
            if sample_index < 3 and variant == 0:
                cv2.imwrite(str(preview / f"{board_id}.png"), image)
    manifest = _save_manifest(destination, rows, seed)
    dataset_info = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
    dataset_info.update({
        "source": "real-style-composite", "piece_style": library.metadata["piece_style"],
        "asset_version": library.metadata["version"], "asset_source": library.metadata["source_image"],
        "render_mode": mode, "augmentation_enabled": augmentation,
        "independent_real_test": False,
    })
    (destination / "dataset.json").write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")
    return manifest
