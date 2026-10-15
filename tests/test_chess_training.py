import unittest
import tempfile
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from game_agent.chess_training import (
    _run_epoch, _training_weights, calibrate_thresholds, confusion_matrix, resolve_training_device,
    train_model,
)


class ChessTrainingTests(unittest.TestCase):
    def _mock_training(self, directory, losses, save_epochs, patience=0, init_checkpoint=None):
        import torch

        class TinyDataset:
            rows = [{"balanced_weight": 1.0}, {"balanced_weight": 1.0}]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return torch.zeros(3, 96, 96), index

        progress = {"epoch": 0, "validation": 0}
        targets = np.asarray([0, 1])
        predictions = np.asarray([0, 1])
        probabilities = np.zeros((2, 13), dtype=np.float32)
        probabilities[0, 0] = probabilities[1, 1] = 0.95
        probabilities[0, 1] = probabilities[1, 0] = 0.05

        def run_epoch(model, _loader, _device, _criterion, optimizer=None):
            if optimizer is not None:
                progress["epoch"] += 1
                with torch.no_grad():
                    next(model.parameters()).fill_(float(progress["epoch"]))
                loss = 1.0 / progress["epoch"]
            elif progress["validation"] < len(losses):
                loss = losses[progress["validation"]]
                progress["validation"] += 1
            else:
                loss = 0.5
            return loss, targets, predictions, probabilities

        checkpoint = Path(directory) / "experiment.pt"
        with patch("game_agent.chess_training.ChessSquareDataset", return_value=TinyDataset()):
            with patch("game_agent.chess_training._run_epoch", side_effect=run_epoch):
                with patch("game_agent.chess_training._save_curve"):
                    with patch("game_agent.chess_training.print_training_environment"):
                        report = train_model(
                            directory, checkpoint, epochs=len(losses), batch_size=2,
                            learning_rate=0.001, device_name="cpu", seed=1,
                            patience=patience, save_epochs=save_epochs,
                            init_checkpoint=init_checkpoint,
                        )
        return checkpoint, report

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_periodic_checkpoints_every_five_epochs_keep_epoch_weights(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = self._mock_training(
                directory, [1.0 - index * 0.05 for index in range(10)], 5
            )
            self.assertEqual([5, 10], [item["epoch"] for item in report["saved_checkpoints"]])
            self.assertEqual(10, report["best_epoch"])
            self.assertEqual(10, report["completed_epochs"])
            for epoch in (5, 10):
                path = checkpoint.with_name(f"experiment_epoch_{epoch:03d}.pt")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(epoch, saved["epoch"])
                self.assertEqual(float(epoch), float(next(iter(saved["state_dict"].values())).flatten()[0]))

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_save_every_epoch_does_not_change_best_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = self._mock_training(directory, [0.1, 0.2, 0.3], 1)
            self.assertEqual([1, 2, 3], [item["epoch"] for item in report["saved_checkpoints"]])
            best = torch.load(checkpoint, map_location="cpu", weights_only=True)
            last = torch.load(
                checkpoint.with_name("experiment_epoch_003.pt"),
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(1, best["epoch"])
            self.assertEqual(3, last["epoch"])
            self.assertEqual(1, report["best_epoch"])

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_zero_interval_and_zero_patience_run_all_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report = self._mock_training(directory, [0.1, 0.2, 0.3, 0.4], 0, 0)
            self.assertEqual([], report["saved_checkpoints"])
            self.assertEqual(0, report["save_interval"])
            self.assertEqual(4, report["completed_epochs"])
            self.assertEqual([], list(Path(directory).glob("experiment_epoch_*.pt")))
            self.assertTrue(checkpoint.exists())

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_early_stopping_preserves_periodic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            _, report = self._mock_training(directory, [0.1, 0.2, 0.3, 0.4], 2, 2)
            self.assertEqual(3, report["completed_epochs"])
            self.assertEqual([2], [item["epoch"] for item in report["saved_checkpoints"]])
            self.assertTrue(Path(report["saved_checkpoints"][0]["path"]).exists())

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_periodic_checkpoint_loads_on_available_devices(self):
        import torch
        from game_agent.chess_cnn import CNNPerception

        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self._mock_training(directory, [0.1], 1)
            periodic = checkpoint.with_name("experiment_epoch_001.pt")
            CNNPerception(periodic, device="cpu")
            loaded = torch.load(periodic, map_location="cpu", weights_only=True)
            self.assertTrue(all(value.device.type == "cpu" for value in loaded["state_dict"].values()))
            if torch.cuda.is_available():
                CNNPerception(periodic, device="cuda")

    def test_negative_save_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "save_epochs"):
            train_model(
                "unused", "unused.pt", epochs=1, batch_size=1,
                learning_rate=0.001, device_name="cpu", seed=1,
                save_epochs=-1,
            )

    def test_fine_tuning_refuses_to_overwrite_initial_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            train_model(
                "unused", "same.pt", epochs=1, batch_size=1,
                learning_rate=0.001, device_name="cpu", seed=1,
                init_checkpoint="same.pt",
            )

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_v2_checkpoint_initializes_fine_tuning_and_mixed_weights(self):
        import torch
        from game_agent.chess_cnn import DEFAULT_MEAN, DEFAULT_STD, MODEL_VERSION, build_model
        from game_agent.chess_dataset import LABELS

        with tempfile.TemporaryDirectory() as directory:
            initial_model = build_model()
            with torch.no_grad():
                for parameter in initial_model.parameters():
                    parameter.fill_(0.25)
            initial = Path(directory) / "v2.pt"
            torch.save({
                "model_version": MODEL_VERSION, "state_dict": initial_model.state_dict(),
                "labels": LABELS, "input_size": 96, "mean": DEFAULT_MEAN, "std": DEFAULT_STD,
                "thresholds": {"confidence": .8, "margin": .2}, "validation_metrics": {},
            }, initial)
            checkpoint, report = self._mock_training(directory, [0.1], 0, init_checkpoint=initial)
            self.assertEqual(str(initial), report["init_checkpoint"])
            self.assertTrue(checkpoint.exists())

        class Rows(torch.utils.data.Dataset):
            def __init__(self):
                self.rows = [{"balanced_weight": 1.0}] * 2
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return index
        combined = torch.utils.data.ConcatDataset([Rows(), Rows()])
        weights = _training_weights(
            [("synthetic", Path("s")), ("real", Path("r"))], combined, .75
        )
        self.assertAlmostEqual(.25, float(weights[:2].sum()))
        self.assertAlmostEqual(.75, float(weights[2:].sum()))

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_auto_uses_cpu_and_explicit_cuda_fails_clearly_when_unavailable(self):
        import torch

        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual("cpu", resolve_training_device(torch, "auto").type)
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch wheel"):
                resolve_training_device(torch, "cuda")

    @unittest.skipUnless(find_spec("torch"), "PyTorch optional dependency is not installed")
    def test_run_epoch_training_and_validation_batches(self):
        import torch

        inputs = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        labels = torch.tensor([0, 1, 2, 1])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, labels), batch_size=4
        )
        model = torch.nn.Linear(4, 3)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        before = model.weight.detach().clone()

        train_loss, train_targets, train_predictions, train_probabilities = _run_epoch(
            model, loader, torch.device("cpu"), criterion, optimizer
        )

        self.assertFalse(torch.equal(before, model.weight.detach()))
        self.assertGreater(train_loss, 0.0)
        self.assertEqual((4,), train_targets.shape)
        self.assertEqual((4,), train_predictions.shape)
        self.assertEqual((4, 3), train_probabilities.shape)
        np.testing.assert_allclose(train_probabilities.sum(axis=1), 1.0, atol=1e-6)

        validation_before = model.weight.detach().clone()
        validation_loss, targets, predictions, probabilities = _run_epoch(
            model, loader, torch.device("cpu"), criterion
        )

        self.assertTrue(torch.equal(validation_before, model.weight.detach()))
        self.assertGreater(validation_loss, 0.0)
        self.assertEqual((4,), targets.shape)
        self.assertEqual((4,), predictions.shape)
        self.assertEqual((4, 3), probabilities.shape)

    def test_validation_calibrates_thresholds_without_test_data(self):
        probabilities = np.zeros((20, 13), np.float32)
        targets = np.arange(20) % 13
        probabilities[np.arange(20), targets] = 0.92
        probabilities[np.arange(20), (targets + 1) % 13] = 0.08
        thresholds = calibrate_thresholds(probabilities, targets)
        self.assertGreaterEqual(thresholds["coverage"], 0.95)
        self.assertEqual(1.0, thresholds["accuracy"])

    def test_confusion_matrix_shape(self):
        matrix = confusion_matrix(np.array([0, 1, 1]), np.array([0, 2, 1]))
        self.assertEqual((13, 13), matrix.shape)
        self.assertEqual(1, matrix[1, 2])


if __name__ == "__main__":
    unittest.main()
