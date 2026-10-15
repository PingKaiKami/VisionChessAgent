from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .geometry import Region
from .input_controller import InputController


class Orientation(str, Enum):
    WHITE_BOTTOM = "white-bottom"
    BLACK_BOTTOM = "black-bottom"


@dataclass(frozen=True)
class ChessCalibration:
    board: Region
    orientation: Orientation = Orientation.WHITE_BOTTOM

    @property
    def local_board(self) -> Region:
        """Board region inside a board-only image; screen input still uses ``board``."""
        return Region(0, 0, self.board.width, self.board.height)

    def square_center(self, square: str) -> tuple[int, int]:
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            raise ValueError(f"Invalid chess square: {square!r}")
        file_index = ord(square[0]) - ord("a")
        rank_index = int(square[1]) - 1
        if self.orientation is Orientation.WHITE_BOTTOM:
            column, row = file_index, 7 - rank_index
        else:
            column, row = 7 - file_index, rank_index
        x = self.board.left + round((column + 0.5) * self.board.width / 8)
        y = self.board.top + round((row + 0.5) * self.board.height / 8)
        return x, y

    def save(self, path: str | Path) -> None:
        data = {"board": asdict(self.board), "orientation": self.orientation.value}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChessCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Region(**data["board"]), Orientation(data["orientation"]))


class ChessInputAdapter:
    def __init__(self, calibration: ChessCalibration, controller: InputController) -> None:
        self.calibration = calibration
        self.controller = controller

    def move(self, source: str, target: str, *, duration: float = 0.4, drag: bool = False) -> None:
        start = self.calibration.square_center(source)
        end = self.calibration.square_center(target)
        if drag:
            self.controller.drag(start, end, duration=duration)
        else:
            self.controller.click(*start, duration=duration / 2)
            self.controller.click(*end, duration=duration / 2)
