# Third-Party Notices

VisionChessAgent's original source code is licensed under the MIT License. The
following separately maintained dependencies retain their own licenses. They
are installed from their upstream distributions and are not relicensed by this
project.

| Dependency | License | Notes |
| --- | --- | --- |
| [python-chess](https://github.com/niklasf/python-chess) | GPL-3.0-or-later | Direct runtime dependency. Distribution of a combined application must comply with the GPL. |
| [Stockfish](https://github.com/official-stockfish/Stockfish) | GPL-3.0 | Invoked as a user-supplied external UCI executable; no Stockfish binary or source is included. Redistributors of Stockfish must meet its GPL source and notice obligations. |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Optional ML dependency; upstream wheels also contain components under additional compatible licenses documented by PyTorch. |
| [Torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | Optional ML dependency. No pretrained Torchvision weights are used. |
| [OpenCV](https://opencv.org/license/) | Apache-2.0 | Used through the `opencv-python-headless` package. |
| [MSS](https://github.com/BoboTiG/python-mss) | MIT | Screen capture dependency. |
| [PyAutoGUI](https://github.com/asweigart/pyautogui) | BSD-3-Clause | Mouse and keyboard automation dependency. |
| [CairoSVG](https://github.com/Kozea/CairoSVG) | LGPL-3.0-or-later | Optional synthetic-dataset rendering dependency. |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | Numerical runtime dependency; binary wheels include their own third-party notices. |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND | Optional image dependency; binary wheels include their own third-party notices. |

The dependency names and trademarks belong to their respective owners. Refer
to each installed distribution and upstream repository for the complete,
version-specific license texts and notices.

## Visual assets and datasets

Chess.com screenshots, board themes, and piece artwork are not included in this
repository and are not covered by the MIT License. Users must provide their own
lawfully obtained images and confirm that their intended use is permitted.
Generated datasets and derivatives retain any restrictions arising from their
source material.

The repository does not distribute pretrained CNN weights. User-trained models
have separate provenance and licensing considerations and may inherit
restrictions from their training data.
