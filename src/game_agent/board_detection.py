from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import chess
import cv2
import numpy as np

from .capture import ScreenFrame
from .chess import ChessCalibration, Orientation
from .chess_perception import BoardState, TemplateLibrary, save_debug_image
from .geometry import Region


class BoardDetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BoardDetection:
    calibration: ChessCalibration
    confidence: float
    state: BoardState

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        board = self.calibration.board
        return (
            (float(board.left), float(board.top)),
            (float(board.right), float(board.top)),
            (float(board.right), float(board.bottom)),
            (float(board.left), float(board.bottom)),
        )


def _checker_score(image: np.ndarray) -> float:
    colors = [[], []]
    height, width = image.shape[:2]
    for row in range(8):
        y0, y1 = round(row * height / 8), round((row + 1) * height / 8)
        for column in range(8):
            x0, x1 = round(column * width / 8), round((column + 1) * width / 8)
            tile = image[y0:y1, x0:x1]
            margin_y, margin_x = max(1, tile.shape[0] // 8), max(1, tile.shape[1] // 8)
            pixels = tile[margin_y:-margin_y, margin_x:-margin_x].reshape(-1, 3)
            colors[(row + column) % 2].append(np.median(pixels, axis=0))
    groups = [np.asarray(group, dtype=np.float32) for group in colors]
    means = [group.mean(axis=0) for group in groups]
    separation = float(np.linalg.norm(means[0] - means[1]))
    within = float(sum(np.linalg.norm(group - mean, axis=1).mean() for group, mean in zip(groups, means)))
    if separation < 18:
        return 0.0
    return max(0.0, min(1.0, separation / (separation + 2.0 * within + 1e-6)))


def _grid_score(image: np.ndarray) -> float:
    """Cheaply score whether expected 8x8 boundaries coincide with color edges."""
    height, width = image.shape[:2]
    scale = min(1.0, 256.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]
    band = max(1, round(min(height, width) / 256))
    vertical: list[float] = []
    horizontal: list[float] = []
    for boundary in range(1, 8):
        x = round(boundary * width / 8)
        for row in range(8):
            y0, y1 = round(row * height / 8), round((row + 1) * height / 8)
            margin = max(1, (y1 - y0) // 6)
            left = image[y0 + margin:y1 - margin, x - 2 * band:x - band]
            right = image[y0 + margin:y1 - margin, x + band:x + 2 * band]
            if left.size and right.size:
                vertical.append(float(np.linalg.norm(
                    np.median(left.reshape(-1, 3), axis=0)
                    - np.median(right.reshape(-1, 3), axis=0)
                )))
        y = round(boundary * height / 8)
        for column in range(8):
            x0, x1 = round(column * width / 8), round((column + 1) * width / 8)
            margin = max(1, (x1 - x0) // 6)
            above = image[y - 2 * band:y - band, x0 + margin:x1 - margin]
            below = image[y + band:y + 2 * band, x0 + margin:x1 - margin]
            if above.size and below.size:
                horizontal.append(float(np.linalg.norm(
                    np.median(above.reshape(-1, 3), axis=0)
                    - np.median(below.reshape(-1, 3), axis=0)
                )))
    if not vertical or not horizontal:
        return 0.0
    vertical_score = min(1.0, float(np.mean(vertical)) / 150.0)
    horizontal_score = min(1.0, float(np.mean(horizontal)) / 150.0)
    return max(0.0, min(vertical_score, horizontal_score))


def _geometry_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    scale = min(1.0, 256.0 / max(height, width))
    sample = image if scale == 1.0 else cv2.resize(
        image, (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return 0.55 * _checker_score(sample) + 0.45 * _grid_score(sample)


def _border_score(image: np.ndarray, region: Region) -> float:
    band = max(1, min(4, round(region.width / 256)))
    if (
        region.left < band or region.top < band
        or region.right + band > image.shape[1]
        or region.bottom + band > image.shape[0]
    ):
        return 0.0
    contrasts: list[float] = []
    for cell in range(8):
        x0 = region.left + round(cell * region.width / 8)
        x1 = region.left + round((cell + 1) * region.width / 8)
        margin = max(1, (x1 - x0) // 6)
        for inside, outside in (
            (
                image[region.top:region.top + band, x0 + margin:x1 - margin],
                image[region.top - band:region.top, x0 + margin:x1 - margin],
            ),
            (
                image[region.bottom - band:region.bottom, x0 + margin:x1 - margin],
                image[region.bottom:region.bottom + band, x0 + margin:x1 - margin],
            ),
        ):
            contrasts.append(float(np.linalg.norm(
                np.median(inside.reshape(-1, 3), axis=0)
                - np.median(outside.reshape(-1, 3), axis=0)
            )))
        y0 = region.top + round(cell * region.height / 8)
        y1 = region.top + round((cell + 1) * region.height / 8)
        margin = max(1, (y1 - y0) // 6)
        for inside, outside in (
            (
                image[y0 + margin:y1 - margin, region.left:region.left + band],
                image[y0 + margin:y1 - margin, region.left - band:region.left],
            ),
            (
                image[y0 + margin:y1 - margin, region.right - band:region.right],
                image[y0 + margin:y1 - margin, region.right:region.right + band],
            ),
        ):
            contrasts.append(float(np.linalg.norm(
                np.median(inside.reshape(-1, 3), axis=0)
                - np.median(outside.reshape(-1, 3), axis=0)
            )))
    return max(0.0, min(1.0, float(np.mean(contrasts)) / 300.0))


def _candidate_geometry_score(image: np.ndarray, region: Region) -> float:
    crop = image[region.top:region.bottom, region.left:region.right]
    return 0.75 * _geometry_score(crop) + 0.25 * _border_score(image, region)


def _edge_refined_region(image: np.ndarray, region: Region) -> Region:
    """Snap a contour candidate to nearby pixel discontinuities."""
    radius = max(4, min(16, round(region.width * 0.02)))
    height, width = image.shape[:2]

    def horizontal(center: int) -> int:
        start = max(1, center - radius)
        stop = min(height - 1, center + radius)
        x0, x1 = max(0, region.left), min(width, region.right)
        return max(
            range(start, stop + 1),
            key=lambda y: float(np.mean(np.abs(
                image[y, x0:x1].astype(np.int16)
                - image[y - 1, x0:x1].astype(np.int16)
            ))),
        )

    def vertical(center: int) -> int:
        start = max(1, center - radius)
        stop = min(width - 1, center + radius)
        y0, y1 = max(0, region.top), min(height, region.bottom)
        return max(
            range(start, stop + 1),
            key=lambda x: float(np.mean(np.abs(
                image[y0:y1, x].astype(np.int16)
                - image[y0:y1, x - 1].astype(np.int16)
            ))),
        )

    left, right = vertical(region.left), vertical(region.right)
    top, bottom = horizontal(region.top), horizontal(region.bottom)
    side = round(((right - left) + (bottom - top)) / 2)
    return Region(left, top, side, side)


def _state_match(state: BoardState, expected_placement: str) -> float:
    expected = chess.Board(f"{expected_placement} w - - 0 1")
    matches = 0
    for square in chess.SQUARES:
        name = chess.square_name(square)
        symbol = state.squares[name].symbol
        piece = expected.piece_at(square)
        if symbol != "unknown" and symbol == (piece.symbol() if piece else "."):
            matches += 1
    return matches / 64.0


def _recognition_quality(state: BoardState) -> float:
    return float(np.mean([result.best_score for result in state.squares.values()]))


class BoardDetector:
    def __init__(
        self,
        *,
        min_side: int = 160,
        min_confidence: float = 0.68,
        ambiguity_margin: float = 0.06,
        debug_directory: str | Path | None = None,
        profiler: Callable[[str, float], None] | None = None,
    ) -> None:
        self.min_side = min_side
        self.min_confidence = min_confidence
        self.ambiguity_margin = ambiguity_margin
        self.debug_directory = Path(debug_directory) if debug_directory else None
        self.profiler = profiler
        self._attempt = 0
        self._previous_summary: dict[str, object] | None = None

    def _record(self, stage: str, started: float) -> None:
        if self.profiler:
            self.profiler(stage, perf_counter() - started)

    def _candidate_regions(self, image: np.ndarray) -> list[Region]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[Region] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < self.min_side or abs(width - height) / max(width, height) > 0.08:
                continue
            region = Region(x, y, width, height)
            if any(
                abs(region.left - other.left) < 6
                and abs(region.top - other.top) < 6
                and abs(region.width - other.width) < 12
                for other in regions
            ):
                continue
            regions.append(region)
        return regions

    @staticmethod
    def _region_variants(
        image: np.ndarray,
        region: Region,
        limit: int = 6,
    ) -> list[tuple[Region, float]]:
        base_side = round((region.width + region.height) / 2)
        radius = max(4, min(16, round(base_side * 0.02)))
        variants: set[Region] = set()
        edge_refined = _edge_refined_region(image, region)
        variants.add(edge_refined)
        for delta_y in range(-1, 2):
            for delta_x in range(-1, 2):
                for delta_side in range(-1, 2):
                    side = base_side + delta_side
                    left, top = region.left + delta_x, region.top + delta_y
                    if left < 0 or top < 0 or left + side > image.shape[1] or top + side > image.shape[0]:
                        continue
                    variants.add(Region(left, top, side, side))

        for delta in range(-radius, radius + 1):
            variants.add(Region(
                region.left + delta, region.top, base_side, base_side,
            ))
            variants.add(Region(
                region.left, region.top + delta, base_side, base_side,
            ))
            variants.add(Region(
                region.left, region.top, base_side + delta, base_side + delta,
            ))
            center_shift = round(-delta / 2.0)
            variants.add(Region(
                region.left + center_shift,
                region.top + center_shift,
                base_side + delta,
                base_side + delta,
            ))

        ranked: list[tuple[Region, float]] = []
        for candidate in variants:
            if (
                candidate.left < 0
                or candidate.top < 0
                or candidate.right > image.shape[1]
                or candidate.bottom > image.shape[0]
            ):
                continue
            score = _candidate_geometry_score(image, candidate)
            if score >= 0.35:
                ranked.append((candidate, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        edge_item = next((item for item in ranked if item[0] == edge_refined), None)
        selected = ([edge_item] if edge_item else []) + [
            item for item in ranked if item[0] != edge_refined
        ][:limit]
        if ranked:
            nearest = min(
                ranked,
                key=lambda item: (
                    abs(item[0].left - region.left)
                    + abs(item[0].top - region.top)
                    + abs(item[0].width - base_side)
                ),
            )
            if nearest not in selected:
                selected.append(nearest)
        return selected

    def _write_diagnostics(
        self,
        frame: ScreenFrame,
        raw_regions: list[Region],
        refined: list[tuple[Region, float]],
        candidates: list[BoardDetection],
        reason: str,
    ) -> None:
        if not self.debug_directory:
            return
        directory = self.debug_directory
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"attempt-{self._attempt:04d}"
        cv2.imwrite(str(directory / f"{prefix}-frame.png"), frame.image)
        overlay = frame.image.copy()
        for region in raw_regions:
            cv2.rectangle(
                overlay, (region.left, region.top), (region.right, region.bottom),
                (255, 128, 0), 2,
            )
        for region, score in refined:
            cv2.rectangle(
                overlay, (region.left, region.top), (region.right, region.bottom),
                (0, 220, 255), 1,
            )
            cv2.putText(
                overlay, f"{score:.3f}", (region.left, max(12, region.top - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(directory / f"{prefix}-candidates.png"), overlay)
        best = max(candidates, key=lambda item: item.confidence) if candidates else None
        summary: dict[str, object] = {
            "raw": len(raw_regions),
            "refined": len(refined),
            "evaluated": len(candidates),
            "reason": reason,
            "confidence": best.confidence if best else None,
            "fen": best.state.piece_placement() if best else None,
            "unknown": len(best.state.unknown_squares) if best else None,
            "region": best.calibration.board if best else None,
        }
        previous = self._previous_summary
        lines = [f"{key}={value}" for key, value in summary.items()]
        for index, candidate in enumerate(
            sorted(candidates, key=lambda item: item.confidence, reverse=True)[:10], 1
        ):
            lines.append(
                f"candidate_{index}={candidate.calibration.board},"
                f"{candidate.calibration.orientation.value},"
                f"confidence={candidate.confidence:.6f},"
                f"fen={candidate.state.piece_placement()},"
                f"unknown={len(candidate.state.unknown_squares)}"
            )
        if previous:
            confidence_delta = (
                float(summary["confidence"]) - float(previous["confidence"])
                if summary["confidence"] is not None and previous["confidence"] is not None
                else "n/a"
            )
            lines.extend([
                f"previous_confidence={previous['confidence']}",
                f"confidence_delta={confidence_delta}",
                f"same_fen={summary['fen'] == previous['fen']}",
                f"region_changed={summary['region'] != previous['region']}",
            ])
        (directory / f"{prefix}.txt").write_text("\n".join(lines), encoding="utf-8")
        self._previous_summary = summary

    def detect(
        self,
        frame: ScreenFrame,
        templates: TemplateLibrary,
        *,
        expected_placement: str,
        orientation: Orientation | None = None,
    ) -> BoardDetection:
        self._attempt += 1
        candidates: list[BoardDetection] = []
        evaluated: set[tuple[Region, Orientation]] = set()
        orientations = [orientation] if orientation else list(Orientation)
        recognition_elapsed = 0.0

        def evaluate(local: Region, checker: float, candidate_orientations=orientations) -> None:
            nonlocal recognition_elapsed
            crop = frame.image[local.top:local.bottom, local.left:local.right]
            absolute = Region(
                frame.region.left + local.left,
                frame.region.top + local.top,
                local.width,
                local.height,
            )
            for candidate_orientation in candidate_orientations:
                key = (local, candidate_orientation)
                if key in evaluated:
                    continue
                evaluated.add(key)
                calibration = ChessCalibration(absolute, candidate_orientation)
                started = perf_counter()
                state = templates.recognize(crop, calibration, align=False)
                recognition_elapsed += perf_counter() - started
                match = _state_match(state, expected_placement)
                known = 1.0 - len(state.unknown_squares) / 64.0
                quality = _recognition_quality(state)
                confidence = (
                    0.20 * checker + 0.50 * match
                    + 0.15 * known + 0.15 * quality
                )
                candidates.append(BoardDetection(calibration, confidence, state))

        started = perf_counter()
        raw_regions = self._candidate_regions(frame.image)
        self._record("Candidate Detection", started)
        started = perf_counter()
        refined: list[tuple[Region, float]] = []
        for raw_region in raw_regions:
            refined.extend(self._region_variants(frame.image, raw_region))
        refined.sort(key=lambda item: item[1], reverse=True)
        unique_refined: list[tuple[Region, float]] = []
        for item in refined:
            region = item[0]
            if any(
                abs(region.left - existing.left) <= 2
                and abs(region.top - existing.top) <= 2
                and abs(region.width - existing.width) <= 2
                for existing, _ in unique_refined
            ):
                continue
            unique_refined.append(item)
            if len(unique_refined) >= 8:
                break
        refined = unique_refined
        self._record("Candidate Refinement", started)

        for local, geometric_score in refined:
            evaluate(local, geometric_score)

        # Only the best coarse recognition receives a small exact refinement.
        # This preserves pixel accuracy without running templates over every
        # coordinate generated during the geometric phase.
        if candidates:
            for axis, radius in (
                ("side", 5), ("top", 5), ("left", 5),
                ("side", 2), ("top", 2), ("left", 2),
                ("side", 1), ("top", 1), ("left", 1),
            ):
                seed = max(candidates, key=lambda item: item.confidence)
                board = seed.calibration.board
                local_seed = Region(
                    board.left - frame.region.left,
                    board.top - frame.region.top,
                    board.width,
                    board.height,
                )
                for delta in range(-radius, radius + 1):
                    if axis == "top":
                        local = Region(local_seed.left, local_seed.top + delta, local_seed.width, local_seed.height)
                    elif axis == "left":
                        local = Region(local_seed.left + delta, local_seed.top, local_seed.width, local_seed.height)
                    else:
                        center_shift = round(-delta / 2.0)
                        local = Region(
                            local_seed.left + center_shift,
                            local_seed.top + center_shift,
                            local_seed.width + delta,
                            local_seed.height + delta,
                        )
                    if (
                        local.left < 0 or local.top < 0
                        or local.right > frame.image.shape[1]
                        or local.bottom > frame.image.shape[0]
                    ):
                        continue
                    geometric_score = _candidate_geometry_score(frame.image, local)
                    if geometric_score >= 0.35:
                        evaluate(local, geometric_score, [seed.calibration.orientation])

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        if not candidates or candidates[0].confidence < self.min_confidence:
            if self.profiler:
                self.profiler("Piece Recognition", recognition_elapsed)
            self._write_diagnostics(
                frame, raw_regions, refined, candidates,
                "No reliable Chess board candidate found",
            )
            raise BoardDetectionError("No reliable Chess board candidate found")
        best = candidates[0]
        best_board = best.calibration.board
        best_local = Region(
            best_board.left - frame.region.left,
            best_board.top - frame.region.top,
            best_board.width,
            best_board.height,
        )
        best_crop = frame.image[
            best_local.top:best_local.bottom,
            best_local.left:best_local.right,
        ]
        started = perf_counter()
        best_state = templates.recognize(best_crop, best.calibration)
        recognition_elapsed += perf_counter() - started
        best_checker = _geometry_score(best_crop)
        best_match = _state_match(best_state, expected_placement)
        best_known = 1.0 - len(best_state.unknown_squares) / 64.0
        best_confidence = (
            0.20 * best_checker + 0.50 * best_match
            + 0.15 * best_known + 0.15 * _recognition_quality(best_state)
        )
        best = BoardDetection(best.calibration, best_confidence, best_state)
        if self.profiler:
            self.profiler("Piece Recognition", recognition_elapsed)
        if best.confidence < self.min_confidence:
            self._write_diagnostics(
                frame, raw_regions, refined, [best, *candidates],
                "Final aligned recognition fell below confidence threshold",
            )
            raise BoardDetectionError("No reliable Chess board candidate found")
        for other in candidates[1:]:
            first, second = best.calibration.board, other.calibration.board
            tolerance = max(4, round(first.width * 0.03))
            same_region = (
                abs(first.left - second.left) <= tolerance
                and abs(first.top - second.top) <= tolerance
                and abs(first.width - second.width) <= tolerance
            )
            if best.confidence - other.confidence < self.ambiguity_margin:
                if same_region and other.calibration.orientation == best.calibration.orientation:
                    continue
                if same_region and orientation is None:
                    self._write_diagnostics(
                        frame, raw_regions, refined, candidates,
                        "Board orientation is ambiguous",
                    )
                    raise BoardDetectionError(
                        "Board orientation is ambiguous; specify it manually"
                    )
                if not same_region:
                    self._write_diagnostics(
                        frame, raw_regions, refined, candidates,
                        "Multiple Chess board candidates are ambiguous",
                    )
                    raise BoardDetectionError("Multiple Chess board candidates are ambiguous")
        self._write_diagnostics(frame, raw_regions, refined, [best, *candidates], "success")
        if self.debug_directory:
            directory = self.debug_directory
            directory.mkdir(parents=True, exist_ok=True)
            local = Region(
                best.calibration.board.left - frame.region.left,
                best.calibration.board.top - frame.region.top,
                best.calibration.board.width,
                best.calibration.board.height,
            )
            crop = frame.image[local.top:local.bottom, local.left:local.right]
            cv2.imwrite(str(directory / "detected-board.png"), crop)
            save_debug_image(
                directory / "detected-grid.png", crop, best.calibration, best.state
            )
            unknown = [
                f"{square}: best={best.state.squares[square].best_label} "
                f"score={best.state.squares[square].best_score:.4f} "
                f"second={best.state.squares[square].second_label} "
                f"second_score={best.state.squares[square].second_score:.4f} "
                f"margin={best.state.squares[square].margin:.4f}"
                for square in best.state.unknown_squares
            ]
            (directory / "detection.txt").write_text(
                "\n".join([
                    f"region={best.calibration.board}",
                    f"corners={best.corners}",
                    f"orientation={best.calibration.orientation.value}",
                    f"confidence={best.confidence:.4f}",
                    *unknown,
                ]),
                encoding="utf-8",
            )
        return best
