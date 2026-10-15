from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .chess import ChessCalibration, Orientation


INITIAL_POSITION = {
    **{f"{file_}1": piece for file_, piece in zip("abcdefgh", "RNBQKBNR")},
    **{f"{file_}2": "P" for file_ in "abcdefgh"},
    **{f"{file_}7": "p" for file_ in "abcdefgh"},
    **{f"{file_}8": piece for file_, piece in zip("abcdefgh", "rnbqkbnr")},
}

NORMALIZED_CELL_SIZE = 96
FEATURE_VERSION = 2


def screen_cell_square(row: int, column: int, orientation: Orientation) -> str:
    if not 0 <= row < 8 or not 0 <= column < 8:
        raise ValueError("Board row and column must be in range 0..7")
    if orientation is Orientation.WHITE_BOTTOM:
        file_index, rank = column, 8 - row
    else:
        file_index, rank = 7 - column, row + 1
    return f"{chr(ord('a') + file_index)}{rank}"


class BoardExtractor:
    def __init__(self, calibration: ChessCalibration) -> None:
        self.calibration = calibration

    def board_image(self, image: np.ndarray) -> np.ndarray:
        board = self.calibration.board
        local_board = self.calibration.local_board
        if image.shape[:2] == (local_board.height, local_board.width):
            return image[
                local_board.top:local_board.bottom,
                local_board.left:local_board.right,
            ]
        if board.left < 0 or board.top < 0 or board.right > image.shape[1] or board.bottom > image.shape[0]:
            raise ValueError(
                "Image is neither board-only nor a (0,0)-origin image containing the calibrated board"
            )
        return image[board.top:board.bottom, board.left:board.right]

    def squares(self, image: np.ndarray) -> dict[str, np.ndarray]:
        board = self.board_image(image)
        x_edges, y_edges = self.grid_edges(image)
        result: dict[str, np.ndarray] = {}
        for row in range(8):
            y0, y1 = round(y_edges[row]), round(y_edges[row + 1])
            for column in range(8):
                x0, x1 = round(x_edges[column]), round(x_edges[column + 1])
                square = screen_cell_square(row, column, self.calibration.orientation)
                tile = board[y0:y1, x0:x1]
                interpolation = (
                    cv2.INTER_AREA
                    if tile.shape[0] > NORMALIZED_CELL_SIZE
                    or tile.shape[1] > NORMALIZED_CELL_SIZE
                    else cv2.INTER_CUBIC
                )
                result[square] = cv2.resize(
                    tile,
                    (NORMALIZED_CELL_SIZE, NORMALIZED_CELL_SIZE),
                    interpolation=interpolation,
                )
        return result

    def grid_edges(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        board = self.board_image(image)
        return (
            np.linspace(0.0, float(board.shape[1]), 9),
            np.linspace(0.0, float(board.shape[0]), 9),
        )

    def normalized_board(self, image: np.ndarray) -> np.ndarray:
        board = self.board_image(image)
        target = NORMALIZED_CELL_SIZE * 8
        interpolation = (
            cv2.INTER_AREA
            if board.shape[0] > target or board.shape[1] > target
            else cv2.INTER_CUBIC
        )
        return cv2.resize(board, (target, target), interpolation=interpolation)


def _feature(tile: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    border = np.concatenate((
        gray[:6, :].reshape(-1), gray[-6:, :].reshape(-1),
        gray[6:-6, :6].reshape(-1), gray[6:-6, -6:].reshape(-1),
    ))
    background = float(np.median(border))
    delta = gray.astype(np.float32) - background
    foreground = np.abs(delta)
    foreground[foreground < 6.0] = 0.0
    peak = float(foreground.max())
    if peak > 0.0:
        foreground *= 255.0 / peak
    normalized = foreground.astype(np.uint8)
    edges = cv2.Canny(normalized, 35, 100)
    shape = cv2.addWeighted(normalized, 0.85, edges, 0.45, 0)
    # Chess UIs commonly draw rank/file labels along the tile edge. They are
    # background decoration, not part of the piece silhouette.
    shape[:11, :] = shape[-11:, :] = 0
    shape[:, :11] = shape[:, -11:] = 0
    polarity = np.zeros_like(shape)
    center = np.s_[11:-11, 11:-11]
    foreground_mask = foreground[center] > 0
    if np.any(foreground_mask):
        foreground_level = float(np.percentile(gray[center][foreground_mask], 75))
        polarity[20:28, 20:28] = 255 if foreground_level >= 205.0 else 128
    return np.stack((shape, polarity))


def _square_color(square: str) -> int:
    return ((ord(square[0]) - ord("a")) + int(square[1])) % 2


def _feature_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_i = first.astype(np.int16)
    second_i = second.astype(np.int16)
    if first_i.ndim == 3:
        return _feature_similarity(first_i[0], second_i[0])
    foreground = (first_i > 8) | (second_i > 8)
    if not np.any(foreground):
        return 1.0
    return 1.0 - float(np.mean(np.abs(first_i[foreground] - second_i[foreground]))) / 255.0


def _feature_variants(feature: np.ndarray) -> np.ndarray:
    _, height, width = feature.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    variants = [feature]
    blurred = feature.copy()
    blurred[0] = cv2.GaussianBlur(feature[0], (3, 3), 0.6)
    variants.append(blurred)
    for operation in (cv2.dilate, cv2.erode):
        transformed = feature.copy()
        transformed[0] = operation(feature[0], np.ones((2, 2), np.uint8))
        variants.append(transformed)
    for scale in (0.96, 1.04):
        matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
        transformed = feature.copy()
        transformed[0] = cv2.warpAffine(feature[0], matrix, (width, height), borderValue=0)
        variants.append(transformed)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed = feature.copy()
        transformed[0] = cv2.warpAffine(feature[0], matrix, (width, height), borderValue=0)
        variants.append(transformed)
    return np.stack(variants)


@dataclass(frozen=True)
class SquareRecognition:
    symbol: str
    confidence: float
    best_label: str | None = None
    best_score: float = 0.0
    margin: float = 0.0
    second_label: str | None = None
    second_score: float = 0.0

    @property
    def color(self) -> str | None:
        if self.symbol in (".", "unknown"):
            return None
        return "white" if self.symbol.isupper() else "black"

    @property
    def piece_type(self) -> str:
        if self.symbol == ".":
            return "empty"
        if self.symbol == "unknown":
            return "unknown"
        return {
            "k": "king", "q": "queen", "r": "rook",
            "b": "bishop", "n": "knight", "p": "pawn",
        }[self.symbol.lower()]


@dataclass
class BoardState:
    squares: dict[str, SquareRecognition]

    @property
    def unknown_squares(self) -> list[str]:
        return [square for square in _fen_square_order() if self.squares[square].symbol == "unknown"]

    def rows(self) -> list[list[str]]:
        return [[self.squares[f"{file_}{rank}"].symbol for file_ in "abcdefgh"] for rank in range(8, 0, -1)]

    def piece_placement(self) -> str:
        fen_rows = []
        for row in self.rows():
            parts: list[str] = []
            empty = 0
            for symbol in row:
                if symbol == ".":
                    empty += 1
                    continue
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append("?" if symbol == "unknown" else symbol)
            if empty:
                parts.append(str(empty))
            fen_rows.append("".join(parts))
        return "/".join(fen_rows)


def _fen_square_order():
    for rank in range(8, 0, -1):
        for file_ in "abcdefgh":
            yield f"{file_}{rank}"


class TemplateLibrary:
    def __init__(self, labels: np.ndarray, colors: np.ndarray, features: np.ndarray) -> None:
        self.labels = labels.astype(str)
        self.colors = colors.astype(np.uint8)
        self.features = features.astype(np.uint8)
        prototype_labels: list[str] = []
        prototype_features: list[np.ndarray] = []
        for label in dict.fromkeys(self.labels):
            for color in (0, 1):
                indices = np.flatnonzero((self.labels == label) & (self.colors == color))
                if indices.size:
                    prototype_labels.append(label)
                    prototype_features.append(
                        np.median(self.features[indices], axis=0).astype(np.uint8)
                    )
        self._prototype_labels = np.asarray(prototype_labels)
        self._prototype_features = np.stack(prototype_features)
        self.feature_variants = np.stack([
            _feature_variants(feature) for feature in self._prototype_features
        ])
        self._feature_variants_i16 = self.feature_variants.astype(np.int16)
        self._label_groups = {
            label: np.flatnonzero(self._prototype_labels == label)
            for label in dict.fromkeys(self._prototype_labels)
        }

    @classmethod
    def learn(cls, image: np.ndarray, calibration: ChessCalibration) -> "TemplateLibrary":
        tiles = BoardExtractor(calibration).squares(image)
        labels, colors, features = [], [], []
        for square in _fen_square_order():
            labels.append(INITIAL_POSITION.get(square, "."))
            colors.append(_square_color(square))
            features.append(_feature(tiles[square]))
        return cls(np.asarray(labels), np.asarray(colors), np.stack(features))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            feature_version=np.asarray(FEATURE_VERSION, dtype=np.uint8),
            labels=self.labels,
            colors=self.colors,
            features=self.features,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary":
        with np.load(path, allow_pickle=False) as data:
            version = int(data["feature_version"]) if "feature_version" in data else 1
            if version != FEATURE_VERSION:
                raise ValueError(
                    f"Template feature version {version} is obsolete; "
                    "run chess-learn again"
                )
            return cls(data["labels"], data["colors"], data["features"])

    def recognize(
        self,
        image: np.ndarray,
        calibration: ChessCalibration,
        *,
        min_score: float = 0.72,
        min_margin: float = 0.025,
        align: bool = True,
    ) -> BoardState:
        tiles = BoardExtractor(calibration).squares(image)
        results: dict[str, SquareRecognition] = {}
        square_names = list(tiles)
        features = np.stack([_feature(tiles[square]) for square in square_names]).astype(np.int16)
        for color in (0, 1):
            square_positions = [
                position for position, square in enumerate(square_names)
                if _square_color(square) == color
            ]
            variants = (
                self._feature_variants_i16
                if align
                else self._prototype_features[:, None, ...].astype(np.int16)
            )[..., 0, :, :]
            observed = features[square_positions, None, None, 0, ...]
            foreground = (observed > 8) | (variants[None, ...] > 8)
            differences = np.abs(observed - variants[None, ...])
            counts = foreground.sum(axis=(-2, -1))
            totals = np.where(foreground, differences, 0).sum(axis=(-2, -1))
            mean_differences = np.zeros_like(totals, dtype=np.float64)
            np.divide(totals, counts, out=mean_differences, where=counts > 0)
            similarities = np.where(
                counts > 0, 1.0 - mean_differences / 255.0, 1.0
            ).max(axis=2)

            labels = list(self._label_groups)
            label_scores = np.stack([
                similarities[:, group].max(axis=1)
                for group in self._label_groups.values()
            ], axis=1)
            for row, position in enumerate(square_positions):
                polarity = int(features[position, 1, 24, 24])
                if polarity:
                    white = polarity == 255
                    for label_index, label in enumerate(labels):
                        if label != "." and label.isupper() != white:
                            label_scores[row, label_index] = -np.inf
            order = np.argsort(-label_scores, axis=1, kind="stable")
            for row, position in enumerate(square_positions):
                best_index, second_index = order[row, :2]
                best_label, second_label = labels[best_index], labels[second_index]
                best_score = float(label_scores[row, best_index])
                second_score = float(label_scores[row, second_index])
                margin = best_score - second_score
                confidence = max(0.0, min(1.0, best_score * min(1.0, margin / 0.10)))
                symbol = best_label if best_score >= min_score and margin >= min_margin else "unknown"
                results[square_names[position]] = SquareRecognition(
                    symbol, confidence, best_label, best_score, margin,
                    second_label, second_score,
                )
        return BoardState(results)


def load_board_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def save_debug_image(
    path: str | Path,
    image: np.ndarray,
    calibration: ChessCalibration,
    state: BoardState,
) -> Path:
    board = BoardExtractor(calibration).board_image(image).copy()
    for row in range(8):
        y0, y1 = round(row * board.shape[0] / 8), round((row + 1) * board.shape[0] / 8)
        for column in range(8):
            x0, x1 = round(column * board.shape[1] / 8), round((column + 1) * board.shape[1] / 8)
            square = screen_cell_square(row, column, calibration.orientation)
            result = state.squares[square]
            color = (0, 0, 255) if result.symbol == "unknown" else (0, 255, 0)
            label = f"{result.symbol}:{result.confidence:.2f}"
            if result.symbol == "unknown":
                label = (
                    f"?->{result.best_label}:{result.best_score:.2f} "
                    f"2={result.second_label}:{result.second_score:.2f}"
                )
            cv2.rectangle(board, (x0, y0), (x1 - 1, y1 - 1), color, 1)
            cv2.putText(board, label, (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), board):
        raise RuntimeError(f"Unable to write debug image: {output}")
    return output
