# User-Trained CNN Models

This repository provides the CNN architecture, dataset tools, training code,
evaluation code, and checkpoint loader. It does **not** distribute pretrained
CNN weights. Users must train a model from scratch or otherwise obtain weights
that they are legally permitted to use.

## Separate rights

- **Source code:** Original VisionChessAgent source code is covered by the MIT
  License in the repository root, subject to the separate licenses of its
  dependencies.
- **Training data:** The user must establish the right to use, transform, and
  where relevant redistribute every screenshot, piece image, board theme, SVG,
  label set, and generated derivative used for training.
- **Model weights:** A trained checkpoint may inherit legal or contractual
  restrictions from its source data. Training the weights yourself does not
  automatically make them MIT-licensed or free of third-party restrictions.
- **Third-party visual assets:** Chess.com artwork, screenshots, and other
  third-party visuals are not included and are not licensed by this project.

The built-in synthetic pipeline uses SVG chess pieces exposed through
`python-chess`. Synthetic output is useful for testing the training pipeline,
but users must still review the relevant dependency and artwork licenses before
using or distributing generated data or weights. Synthetic-only training also
does not guarantee recognition accuracy on an unseen real interface.

For a real interface, use lawfully obtained images with user-confirmed labels,
keep independent validation and test sources, and record the origin, license,
transformations, and allowed uses of every source. Refer to
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for dependency details.

## Historical note

A locally trained checkpoint was previously committed while the repository was
private. Removing it from the current tree does not remove it from Git history.
Publishing this repository without that historical object requires an explicit,
coordinated history-cleaning decision; no history rewrite is performed here.
