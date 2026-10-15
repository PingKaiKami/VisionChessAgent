# VisionChessAgent

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-blueviolet)

**A vision-only autonomous chess agent for research and education.**

VisionChessAgent explores how an agent can **observe a game through screenshots, recognize its state, plan actions, control the mouse, verify results, and recover safely from interruptions**. Chess is the project's first and final experimental environment; broader, cross-game agent research will continue in a separate repository.

> **Research use only — not a cheating tool.** This project is intended for offline, simulated, educational, or explicitly authorized chess environments. Do **not** use it against human opponents to gain an unfair advantage, or in violation of any platform's fair-play rules.

## How it works

```text
Screen capture (MSS)
    ↓
Board detection / calibration (OpenCV)
    ↓
Piece recognition (CNN or template matching)
    ↓
State tracking and legality checks (python-chess)
    ↓
Move planning (local Stockfish)
    ↓
Mouse input (PyAutoGUI)
    ↓
Visual verification → Safe recovery → Next observation
```

The agent observes through **screen pixels** and acts through **mouse input**. It does not read game memory, intercept network traffic, or rely on game-internal APIs or the browser DOM. Runtime operation does not require an LLM or cloud vision service.

## Features

- **Vision:** screenshot capture, chessboard calibration/detection, and 13-class CNN or template-based piece recognition.
- **Learning pipeline:** synthetic data, verified screenshot imports, real-style piece assets, CNN training/fine-tuning, checkpoint comparison, and evaluation.
- **Closed-loop play:** Stockfish planning, legal-move tracking, visual action verification, and opponent-move synchronization.
- **Safe execution:** foreground-window checks, interruption-aware pause/recovery, configurable timeouts, and optional debug output.
- **Full-game mode:** standard-position games without a fixed move limit, plus promotion handling for the verified white-bottom Chess.com-style menu.

## Requirements and installation

- Windows with **Python 3.11+**.
- A **local Stockfish executable** (not bundled with this repository).
- An offline or explicitly authorized chess interface.
- A model trained by the user for CNN mode; pretrained weights are not distributed.

```powershell
git clone https://github.com/PingKaiKami/VisionChessAgent.git
cd VisionChessAgent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ml]"
python -m pip install pytest
python -m pytest
```

For GPU training, install a CUDA-compatible PyTorch build for your system. Synthetic SVG rendering may also require the Cairo runtime on Windows.

GitHub Actions runs the complete mocked test suite on Windows with Python 3.11 for every push and pull request. CI does not launch Chess.com, Stockfish, GUI automation, or live mouse input.

## Quick start

Pretrained CNN weights are not distributed. The following workflow trains a model from scratch and keeps the resulting checkpoint local.

**1. Install the project and optional ML dependencies.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ml]"
python -m pip install pytest
```

**2. Verify the environment.**

```powershell
python -m pytest -q
game-agent --help
```

**3. Build a training dataset from material you are permitted to use.** This built-in example renders legal positions with the SVG chess pieces exposed through `python-chess`:

```powershell
game-agent chess-dataset `
  --output ".\dataset_synthetic" `
  --count 1000 `
  --variants 2 `
  --seed 42
```

Synthetic generation may require a local Cairo runtime. Confirm the applicable licenses before using or redistributing SVG artwork, generated data, or derived weights. Synthetic images are useful for pipeline validation but do not guarantee accurate recognition of a real game interface. For real-interface adaptation, collect appropriately licensed screenshots, verify their FEN labels, and keep independent validation and test images.

**4. Train a CNN from scratch.** Omitting `--init-checkpoint` ensures that training does not start from existing weights.

```powershell
game-agent chess-train `
  --dataset ".\dataset_synthetic" `
  --checkpoint ".\models\chess_cnn.pt" `
  --epochs 30 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --device auto `
  --patience 5 `
  --save-epochs 0
```

**5. Evaluate the trained checkpoint on held-out data.**

```powershell
game-agent chess-evaluate `
  --dataset ".\dataset_synthetic" `
  --model ".\models\chess_cnn.pt" `
  --output ".\models\chess_cnn_evaluation.json"
```

**6. Calibrate, capture, and inspect recognition.** The calibration values are screen-space `X Y WIDTH HEIGHT`; X can be negative on a multi-monitor setup.

```powershell
game-agent calibrate --board X Y WIDTH HEIGHT --orientation white-bottom --output chess-calibration.json
game-agent screenshot --config chess-calibration.json --output board_start.png
game-agent chess-scan `
  --image board_start.png `
  --config chess-calibration.json `
  --backend cnn `
  --model ".\models\chess_cnn.pt"
```

Template matching remains available when no CNN checkpoint is available. Learn templates from a user-provided standard opening screenshot with `chess-learn`, then use `--backend template --templates .\chess-templates.npz` with `chess-scan` or `chess-play`. Its robustness and accuracy are not equivalent to a CNN trained for the target interface.

**7. Dry-run first, then use live input only in an authorized offline environment.** Start from a standard opening position as White. The supported automatic promotion UI expects white pieces at the bottom.

```powershell
# Dry-run: plans one move without controlling the mouse.
game-agent chess-play `
  --engine-path ".\stockfish\src\stockfish.exe" `
  --config chess-calibration.json `
  --backend cnn `
  --model ".\models\chess_cnn.pt" `
  --window-title "Chess" `
  --start-delay 5

# Live mode: only in an explicitly authorized offline test environment.
game-agent chess-play `
  --engine-path ".\stockfish\src\stockfish.exe" `
  --config chess-calibration.json `
  --backend cnn `
  --model ".\models\chess_cnn.pt" `
  --window-title "Chess" `
  --start-delay 5 `
  --max-moves 0 `
  --verify-timeout 60 `
  --max-retries 120 `
  --pause-timeout 300 `
  --pointer-idle-seconds 2 `
  --promotion-mode auto `
  --enable-live
```

For a consistent target interface, keep the board theme and piece style fixed and disable overlays such as coordinates, last-move highlights, and suggestion arrows if they cover piece pixels. The prior real-screen development test used Chess.com's Analysis board, not a live match. Real-game execution has not been validated and must not be used for assistance or automation against human opponents.

**Safety:** Live mouse input is disabled unless `--enable-live` is supplied. Use **Ctrl+C** to stop; PyAutoGUI also includes a screen-corner fail-safe. Watch live tests and do not treat recovery as permission to operate an unintended window. The agent does **not** support arbitrary midgame initialization.

## Data and training commands

| Command | Purpose |
| --- | --- |
| `chess-dataset` | Generate synthetic, labeled chess positions |
| `chess-import` | Import a real screenshot with a user-confirmed FEN |
| `chess-assets` | Extract verified real chess-piece assets |
| `chess-real-dataset` | Generate boards using real-style chess tiles |
| `chess-train` | Train or fine-tune the CNN; supports `--init-checkpoint` and `--save-epochs` |
| `chess-evaluate` | Evaluate and compare recognition models |

`chess_asset_sources.json` is an **example manifest** referencing local screenshots that are not included. Users are responsible for obtaining suitable rights to all source images, generated datasets, and derived models. Samples synthesized from the same source images do **not** count as an independent real-world test set.

## Repository structure

```text
src/game_agent/          Vision, chess control, dataset, training, CLI, and safety code
tests/                   Unit and regression tests
models/README.md         Model training, provenance, and rights guidance
chess_asset_sources.json Example real-image asset manifest
THIRD_PARTY_NOTICES.md   Dependency and visual-asset license notices
pyproject.toml           Dependencies and CLI entry points
```

Calibration files, screenshots, training datasets, model checkpoints, training reports, and the local Stockfish binary are excluded from version control. Users must train or otherwise lawfully obtain their own CNN checkpoint.

## Research scope

This project is a **chess-specific proof of concept**, not a universal game-playing system. Its central question is how reliably a screen-observing agent can close the loop between **perception, planning, action, verification, and recovery** in an environment with explicit rules.

The experience gained here will inform a **new, separate universal-agent repository**. Potential follow-up research includes cross-game transfer, uncertainty-aware state estimation, and real-time visual control.

## License

Original project source code is Copyright (c) 2026 PingKaiKami and licensed under the [MIT License](LICENSE). Dependencies remain under their respective licenses; notably, `python-chess` is GPL-3.0-or-later and Stockfish is GPL-3.0. See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for details and obligations.

The MIT License does **not** cover Chess.com screenshots or artwork, generated datasets derived from them, or user-trained model weights. No pretrained checkpoint is distributed; see the [model provenance notice](models/README.md). Training a model yourself does not remove restrictions inherited from third-party source material.

## Responsible use

Use this code only where automation is permitted. Do not deploy it in rated games, matches against human players, or any situation in which engine assistance or automated input would violate fair-play expectations. The purpose is to study visual agents and reliable interaction, **not to cheat at chess**.
