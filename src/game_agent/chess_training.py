from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .chess import ChessCalibration, Orientation
from .chess_cnn import (
    DEFAULT_MEAN, DEFAULT_STD, MODEL_VERSION, CNNPerception, build_model, require_torch,
)
from .chess_dataset import LABELS
from .chess_perception import TemplateLibrary, load_board_image
from .geometry import Region


def load_manifest(dataset: str | Path, split: str) -> list[dict[str, object]]:
    path = Path(dataset) / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["split"] == split]


class RandomResolution:
    def __init__(self, minimum: int = 56, maximum: int = 96) -> None:
        self.minimum, self.maximum = minimum, maximum

    def __call__(self, image: Image.Image) -> Image.Image:
        side = random.randint(self.minimum, self.maximum)
        return image.resize((side, side), Image.Resampling.BILINEAR).resize(
            (96, 96), Image.Resampling.BILINEAR
        )


class RandomJpeg:
    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(55, 95))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def build_transform(training: bool):
    try:
        from torchvision import transforms
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("Training requires the optional ML dependencies: pip install -e .[ml]") from error
    operations = [transforms.Resize((96, 96))]
    if training:
        operations.extend((
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10, hue=0.02),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.20),
            transforms.RandomApply([RandomResolution()], p=0.25),
            transforms.RandomApply([RandomJpeg()], p=0.25),
        ))
    operations.extend((transforms.ToTensor(), transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)))
    return transforms.Compose(operations)


class ChessSquareDataset:
    def __init__(self, root: str | Path, split: str, training: bool) -> None:
        self.root = Path(root)
        self.rows = load_manifest(root, split)
        self.transform = build_transform(training)
        self.label_to_index = {label: index for index, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / str(row["path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[str(row["label"])]


def _dataset_sources(
    dataset: str | Path | None,
    synthetic_dataset: str | Path | None,
    real_dataset: str | Path | None,
) -> list[tuple[str, Path]]:
    if dataset:
        if synthetic_dataset or real_dataset:
            raise ValueError("--dataset cannot be combined with --synthetic-dataset/--real-dataset")
        return [("dataset", Path(dataset))]
    sources = []
    if synthetic_dataset:
        sources.append(("synthetic", Path(synthetic_dataset)))
    if real_dataset:
        sources.append(("real", Path(real_dataset)))
    if not sources:
        raise ValueError("A dataset or synthetic/real dataset pair is required")
    return sources


def _data_summary(sources: list[tuple[str, Path]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, root in sources:
        manifest = root / "manifest.jsonl"
        if not manifest.exists():
            result[name] = {"path": str(root), "samples": None, "sources": [], "class_counts": {}}
            continue
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        result[name] = {
            "path": str(root), "samples": len(rows),
            "sources": sorted({str(row.get("source", "unknown")) for row in rows}),
            "class_counts": {
                label: sum(str(row["label"]) == label for row in rows) for label in LABELS
            },
        }
    return result


def _combined_data(sources: list[tuple[str, Path]], split: str, training: bool):
    torch = require_torch()
    datasets = [ChessSquareDataset(root, split, training) for _, root in sources]
    return datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)


def _training_weights(sources: list[tuple[str, Path]], datasets, real_ratio: float):
    torch = require_torch()
    parts = list(datasets.datasets) if hasattr(datasets, "datasets") else [datasets]
    weights: list[float] = []
    has_mix = len(sources) == 2 and {name for name, _ in sources} == {"synthetic", "real"}
    for (name, _), data in zip(sources, parts):
        multiplier = 1.0
        if has_mix:
            target = real_ratio if name == "real" else 1.0 - real_ratio
            multiplier = target / max(1, len(data))
        weights.extend(float(row.get("balanced_weight", 1.0)) * multiplier for row in data.rows)
    return torch.as_tensor(weights, dtype=torch.double)


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def metric_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions)
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    predicted_totals = matrix.sum(axis=0)
    per_class = {}
    for index, label in enumerate(LABELS):
        recall = float(correct[index] / totals[index]) if totals[index] else None
        precision = float(correct[index] / predicted_totals[index]) if predicted_totals[index] else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else 0.0,
        "per_class_accuracy": {label: values["recall"] for label, values in per_class.items()},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def calibrate_thresholds(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    order = np.argsort(-probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), order[:, 0]]
    margin = confidence - probabilities[np.arange(len(probabilities)), order[:, 1]]
    predictions = order[:, 0]
    best = (0.0, 0.50, 0.05, 1.0)
    for threshold in np.arange(0.50, 0.951, 0.05):
        for margin_threshold in np.arange(0.05, 0.401, 0.05):
            known = (confidence >= threshold) & (margin >= margin_threshold)
            coverage = float(known.mean())
            if coverage < 0.95:
                continue
            accuracy = float(np.mean(predictions[known] == targets[known])) if known.any() else 0.0
            candidate = (accuracy, float(threshold), float(margin_threshold), coverage)
            if (candidate[0], candidate[1] + candidate[2]) > (best[0], best[1] + best[2]):
                best = candidate
    return {"confidence": best[1], "margin": best[2], "coverage": best[3], "accuracy": best[0]}


def _run_epoch(model, loader, device, criterion, optimizer=None):
    torch = require_torch()
    model.train(optimizer is not None)
    loss_total, targets, predictions, probabilities = 0.0, [], [], []
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_total += float(loss.item()) * len(labels)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                targets.append(labels.cpu().numpy())
                predictions.append(probs.argmax(1).cpu().numpy())
                probabilities.append(probs.cpu().numpy())
    return (
        loss_total / max(1, len(loader.dataset)),
        np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities),
    )


def resolve_training_device(torch, requested: str):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "none (CPU-only PyTorch build)"
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            f"PyTorch={torch.__version__}, CUDA build={cuda_build}. "
            "Possible causes: a CPU-only PyTorch wheel, an incompatible/missing NVIDIA "
            "driver, or the wrong virtual environment. Diagnose with `nvidia-smi` and "
            "`python -c \"import torch; print(torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available())\"`. Installing the CUDA Toolkit alone does not "
            "convert a CPU-only PyTorch wheel into a CUDA build."
        )
    return torch.device(requested)


def print_training_environment(torch, device) -> None:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "n/a"
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build version: {torch.version.cuda or 'none (CPU-only build)'}")
    print(f"CUDA available: {cuda_available}")
    print(f"Training device: {device}")
    print(f"GPU name: {gpu_name}")


def _checkpoint_payload(model, thresholds, validation_metrics, entry):
    cpu_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return {
        "model_version": MODEL_VERSION,
        "state_dict": cpu_state,
        "labels": LABELS,
        "input_size": 96,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        **entry,
    }


def _epoch_checkpoint_path(output: Path, epoch: int) -> Path:
    return output.with_name(f"{output.stem}_epoch_{epoch:03d}{output.suffix}")


def _save_curve(path: Path, history: list[dict[str, float]]) -> None:
    image = np.full((360, 720, 3), 255, np.uint8)
    colors = {"train_loss": (190, 80, 30), "validation_loss": (30, 80, 190),
              "train_accuracy": (30, 150, 30), "validation_accuracy": (160, 30, 160)}
    for key, color in colors.items():
        values = np.asarray([entry[key] for entry in history], dtype=float)
        if "loss" in key:
            maximum = max(1e-6, float(values.max()))
            values = 1.0 - values / maximum
        points = np.asarray([
            (30 + round(i * 660 / max(1, len(values) - 1)), 330 - round(float(value) * 300))
            for i, value in enumerate(values)
        ], np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], False, color, 2)
        cv2.putText(image, key, (510, 22 + 20 * list(colors).index(key)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    cv2.imwrite(str(path), image)


def train_model(
    dataset: str | Path | None, checkpoint: str | Path, *, epochs: int, batch_size: int,
    learning_rate: float, device_name: str, seed: int, patience: int = 5,
    num_workers: int = 0, save_epochs: int = 0,
    init_checkpoint: str | Path | None = None,
    synthetic_dataset: str | Path | None = None,
    real_dataset: str | Path | None = None,
    real_ratio: float = 0.5,
) -> dict[str, object]:
    if save_epochs < 0:
        raise ValueError("save_epochs must be zero or greater")
    if not 0.0 <= real_ratio <= 1.0:
        raise ValueError("real_ratio must be between zero and one")
    if init_checkpoint and Path(init_checkpoint).resolve() == Path(checkpoint).resolve():
        raise ValueError("Fine-tuning output must not overwrite the initial checkpoint")
    torch = require_torch()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = resolve_training_device(torch, device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print_training_environment(torch, device)
    pin_memory = device.type == "cuda"
    sources = _dataset_sources(dataset, synthetic_dataset, real_dataset)
    train_data = _combined_data(sources, "train", training=True)
    validation_data = _combined_data(sources, "validation", training=False)
    test_data = _combined_data(sources, "test", training=False)
    weights = _training_weights(sources, train_data, real_ratio)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, **loader_options)
    validation_loader = torch.utils.data.DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **loader_options)
    model = build_model().to(device)
    if init_checkpoint:
        initial = torch.load(str(init_checkpoint), map_location="cpu", weights_only=True)
        if int(initial.get("model_version", -1)) != MODEL_VERSION:
            raise ValueError("Initial checkpoint model_version is incompatible")
        if tuple(initial.get("labels", ())) != tuple(LABELS):
            raise ValueError("Initial checkpoint labels are incompatible")
        if int(initial.get("input_size", -1)) != 96:
            raise ValueError("Initial checkpoint input_size is incompatible")
        model.load_state_dict(initial["state_dict"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output = Path(checkpoint); output.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    saved_checkpoints: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_p, _ = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_y, val_p, val_probs = _run_epoch(model, validation_loader, device, criterion)
        entry = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                 "train_accuracy": float(np.mean(train_y == train_p)), "validation_accuracy": float(np.mean(val_y == val_p))}
        history.append(entry)
        print(f"epoch={epoch} train_loss={train_loss:.4f} validation_loss={val_loss:.4f} train_accuracy={entry['train_accuracy']:.4f} validation_accuracy={entry['validation_accuracy']:.4f}")
        improved = val_loss < best_loss
        periodic = save_epochs > 0 and epoch % save_epochs == 0
        payload = None
        if improved or periodic:
            thresholds = calibrate_thresholds(val_probs, val_y)
            payload = _checkpoint_payload(
                model, thresholds, metric_summary(val_y, val_p), entry
            )
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(payload, output)
        else:
            stale += 1
        if periodic:
            periodic_path = _epoch_checkpoint_path(output, epoch)
            torch.save(payload, periodic_path)
            saved_checkpoints.append({"epoch": epoch, "path": str(periodic_path)})
        if not improved and patience > 0 and stale >= patience:
            break
    checkpoint_data = torch.load(str(output), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_data["state_dict"])
    test_loss, test_y, test_p, _ = _run_epoch(model, test_loader, device, criterion)
    report = {
        "device": str(device), "history": history, "test_loss": test_loss,
        "test": metric_summary(test_y, test_p), "thresholds": checkpoint_data["thresholds"],
        "save_interval": save_epochs, "saved_checkpoints": saved_checkpoints,
        "best_epoch": best_epoch, "completed_epochs": len(history),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "real_ratio": real_ratio if len(sources) > 1 else None,
        "data_sources": _data_summary(sources),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_curve(output.with_suffix(".curve.png"), history)
    return report


def evaluate_backends(
    dataset: str | Path, model_path: str | Path, *, templates_path: str | Path | None = None,
    compare_model_path: str | Path | None = None, output: str | Path | None = None,
) -> dict[str, object]:
    rows = load_manifest(dataset, "test")
    boards: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        boards.setdefault(str(row["board_id"]), []).append(row)
    backends: dict[str, object] = {"cnn": CNNPerception(model_path)}
    if compare_model_path:
        backends["cnn_compare"] = CNNPerception(compare_model_path)
    if templates_path:
        backends["template"] = TemplateLibrary.load(templates_path)
    label_index = {label: index for index, label in enumerate(LABELS)}
    report: dict[str, object] = {"dataset": "held-out", "sources": sorted({str(row["source"]) for row in rows})}
    for name, backend in backends.items():
        targets, predictions, unknown, elapsed = [], [], 0, 0.0
        board_results: list[bool] = []
        high_confidence_errors = 0
        rb_confusion = {"r_as_b": 0, "b_as_r": 0}
        style_totals: dict[str, list[int]] = {}
        category_totals: dict[str, list[int]] = {}
        breakdown: dict[str, dict[str, list[int]]] = {
            "board_size": {"windowed": [0, 0], "fullscreen": [0, 0]},
            "source": {}, "stage": {}, "square_color": {"dark": [0, 0], "light": [0, 0]},
        }
        accepted_correct = 0
        for board_rows in boards.values():
            row = board_rows[0]
            image = load_board_image(Path(dataset) / str(row["board_path"]))
            height, width = image.shape[:2]
            calibration = ChessCalibration(Region(0, 0, width, height), Orientation(str(row["orientation"])))
            started = time.perf_counter(); state = backend.recognize(image, calibration); elapsed += time.perf_counter() - started
            bucket = "fullscreen" if max(width, height) >= 900 else "windowed"
            this_board_correct = True
            for item in board_rows:
                target = label_index[str(item["label"])]
                result = state.squares[str(item["square"])]
                targets.append(target)
                if result.symbol == "unknown":
                    unknown += 1; prediction = label_index.get(result.best_label or "empty", label_index["empty"])
                else:
                    prediction = label_index["empty" if result.symbol == "." else result.symbol]
                predictions.append(prediction)
                correct = int(prediction == target and result.symbol != "unknown")
                this_board_correct = this_board_correct and bool(correct)
                accepted_correct += correct
                if result.symbol != "unknown" and not correct and result.best_score >= 0.90:
                    high_confidence_errors += 1
                target_label = str(item["label"])
                predicted_label = LABELS[prediction]
                if target_label == "r" and predicted_label == "b":
                    rb_confusion["r_as_b"] += 1
                if target_label == "b" and predicted_label == "r":
                    rb_confusion["b_as_r"] += 1
                style = str(item.get("piece_style", "unspecified"))
                style_totals.setdefault(style, [0, 0, 0])
                style_totals[style][0] += correct
                style_totals[style][1] += int(result.symbol == "unknown")
                style_totals[style][2] += 1
                source = str(item.get("source", "unknown"))
                category = (
                    "real-screenshot" if source == "verified-real"
                    else "real-style-composite" if source == "real-style-composite"
                    else "synthetic"
                )
                category_totals.setdefault(category, [0, 0, 0])
                category_totals[category][0] += correct
                category_totals[category][1] += int(result.symbol == "unknown")
                category_totals[category][2] += 1
                keys = {
                    "board_size": bucket, "source": str(item["source"]),
                    "stage": str(item["stage"]),
                    "square_color": "light" if int(item["square_color"]) else "dark",
                }
                for category, key in keys.items():
                    breakdown[category].setdefault(key, [0, 0])
                    breakdown[category][key][0] += correct
                    breakdown[category][key][1] += 1
            board_results.append(this_board_correct)
        target_array, prediction_array = np.asarray(targets), np.asarray(predictions)
        metrics = metric_summary(target_array, prediction_array)
        metrics.update({
            "accepted_accuracy": accepted_correct / max(1, len(targets) - unknown),
            "unknown_rate": unknown / max(1, len(targets)),
            "full_board_accuracy": float(np.mean(board_results)) if board_results else 0.0,
            "high_confidence_misclassifications": high_confidence_errors,
            "rook_bishop_confusion": rb_confusion,
            "piece_style": {
                key: {"accuracy": correct / max(1, total), "unknown_rate": unknown_count / max(1, total), "samples": total}
                for key, (correct, unknown_count, total) in style_totals.items()
            },
            "evaluation_sets": {
                key: {"accuracy": correct / max(1, total), "unknown_rate": unknown_count / max(1, total), "samples": total}
                for key, (correct, unknown_count, total) in category_totals.items()
            },
            "average_board_inference_seconds": elapsed / max(1, len(boards)),
            "breakdown_accuracy": {
                category: {key: correct / max(1, total) for key, (correct, total) in values.items()}
                for category, values in breakdown.items()
            },
        })
        report[name] = metrics
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
