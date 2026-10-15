from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .chess import ChessCalibration
from .chess_dataset import LABELS
from .chess_perception import BoardExtractor, BoardState, SquareRecognition


MODEL_VERSION = 1
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CNN support is optional; install it with: pip install -e .[ml]"
        ) from error
    return torch


def build_model(num_classes: int = len(LABELS)):
    torch = require_torch()
    nn = torch.nn

    class SmallChessCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[Any] = []
            channels = 3
            for output in (32, 64, 128):
                blocks.extend((
                    nn.Conv2d(channels, output, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                ))
                channels = output
            self.features = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, inputs):
            features = self.pool(self.features(inputs)).flatten(1)
            return self.classifier(features)

    return SmallChessCNN()


def board_state_from_probabilities(
    square_names: list[str],
    probabilities: np.ndarray,
    labels: tuple[str, ...] | list[str],
    min_confidence: float,
    min_margin: float,
) -> BoardState:
    if probabilities.shape != (len(square_names), len(labels)):
        raise ValueError("Probability matrix shape does not match squares and labels")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    squares: dict[str, SquareRecognition] = {}
    for row, square in enumerate(square_names):
        best_index, second_index = order[row, :2]
        confidence = float(probabilities[row, best_index])
        second_score = float(probabilities[row, second_index])
        margin = confidence - second_score
        best_label, second_label = labels[best_index], labels[second_index]
        accepted = confidence >= min_confidence and margin >= min_margin
        symbol = ("." if best_label == "empty" else best_label) if accepted else "unknown"
        squares[square] = SquareRecognition(
            symbol=symbol,
            confidence=confidence if accepted else max(0.0, confidence * min(1.0, margin / max(min_margin, 1e-6))),
            best_label="." if best_label == "empty" else best_label,
            best_score=confidence,
            margin=margin,
            second_label="." if second_label == "empty" else second_label,
            second_score=second_score,
        )
    return BoardState(squares)


class CNNPerception:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.torch = require_torch()
        if device == "auto":
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.device = self.torch.device(device)
        try:
            checkpoint = self.torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=True
            )
        except TypeError as error:
            raise RuntimeError(
                "This PyTorch version cannot safely load weights-only checkpoints"
            ) from error
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Invalid CNN checkpoint")
        if int(checkpoint.get("model_version", -1)) != MODEL_VERSION:
            raise ValueError("Unsupported CNN checkpoint version")
        self.labels = tuple(checkpoint.get("labels", LABELS))
        self.input_size = int(checkpoint.get("input_size", 96))
        self.mean = np.asarray(checkpoint.get("mean", DEFAULT_MEAN), dtype=np.float32)
        self.std = np.asarray(checkpoint.get("std", DEFAULT_STD), dtype=np.float32)
        thresholds = checkpoint.get("thresholds", {})
        self.min_confidence = float(thresholds.get("confidence", 0.80))
        self.min_margin = float(thresholds.get("margin", 0.20))
        self.model = build_model(len(self.labels)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def recognize(self, image: np.ndarray, calibration: ChessCalibration, **_: object) -> BoardState:
        tiles = BoardExtractor(calibration).squares(image)
        names = list(tiles)
        batch = []
        for square in names:
            rgb = cv2.cvtColor(tiles[square], cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (self.input_size, self.input_size):
                rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            array = rgb.astype(np.float32) / 255.0
            batch.append(((array - self.mean) / self.std).transpose(2, 0, 1))
        inputs = self.torch.from_numpy(np.stack(batch)).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(inputs), dim=1).cpu().numpy()
        return board_state_from_probabilities(
            names, probabilities, self.labels, self.min_confidence, self.min_margin
        )
