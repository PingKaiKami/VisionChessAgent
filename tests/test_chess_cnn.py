import unittest

import numpy as np

from game_agent.chess_cnn import board_state_from_probabilities, require_torch
from game_agent.chess_dataset import LABELS


class ChessCnnTests(unittest.TestCase):
    def test_batch_probabilities_create_compatible_board_state(self):
        squares = [f"{file_}{rank}" for rank in range(8, 0, -1) for file_ in "abcdefgh"]
        probabilities = np.full((64, len(LABELS)), 0.001, np.float32)
        probabilities[:, LABELS.index("empty")] = 0.98
        probabilities[squares.index("e1"), :] = 0.001
        probabilities[squares.index("e1"), LABELS.index("K")] = 0.97
        state = board_state_from_probabilities(squares, probabilities, LABELS, 0.8, 0.2)
        self.assertEqual("K", state.squares["e1"].symbol)
        self.assertEqual(".", state.squares["e4"].symbol)
        self.assertEqual("8/8/8/8/8/8/8/4K3", state.piece_placement())

    def test_uncertain_prediction_is_unknown(self):
        probabilities = np.zeros((1, len(LABELS)), np.float32)
        probabilities[0, LABELS.index("k")] = 0.45
        probabilities[0, LABELS.index("q")] = 0.40
        state = board_state_from_probabilities(["d8"], probabilities, LABELS, 0.8, 0.2)
        self.assertEqual("unknown", state.squares["d8"].symbol)
        self.assertEqual("k", state.squares["d8"].best_label)

    def test_missing_torch_has_actionable_optional_dependency_error(self):
        try:
            require_torch()
        except RuntimeError as error:
            self.assertIn(".[ml]", str(error))


if __name__ == "__main__":
    unittest.main()
