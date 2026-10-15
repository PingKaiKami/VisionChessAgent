import unittest

from game_agent.chess import ChessCalibration, ChessInputAdapter
from game_agent.geometry import Region
from game_agent.input_controller import InputController, PointerInterferenceError
from game_agent.safety import SafetyError, SafetyGuard
from game_agent.window import WindowInfo


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.point = (0, 0)
        self.interfere = False

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "moveTo" and not self.interfere:
                self.point = (args[0], args[1])
        return call

    def position(self):
        return self.point


def matching_window():
    return WindowInfo(1, "Offline Chess", Region(0, 0, 1200, 1000))


class InputControllerTests(unittest.TestCase):
    def setUp(self):
        self.board = Region(100, 100, 800, 800)
        self.guard = SafetyGuard("Chess", self.board, matching_window)

    def test_e2_to_e4_dry_run_flow(self):
        controller = InputController(self.guard)
        ChessInputAdapter(ChessCalibration(self.board), controller).move("e2", "e4")
        names = [action[0] for action in controller.actions]
        self.assertEqual(["moveTo", "click", "moveTo", "click"], names)
        self.assertEqual((550, 750), controller.actions[0][1:3])
        self.assertEqual((550, 550), controller.actions[2][1:3])

    def test_live_requires_matching_foreground_window(self):
        wrong = lambda: WindowInfo(2, "Email", Region(0, 0, 1200, 1000))
        controller = InputController(
            SafetyGuard("Chess", self.board, wrong), enable_live=True, backend=FakeBackend()
        )
        with self.assertRaises(SafetyError):
            controller.click(200, 200)

    def test_same_title_different_window_handle_is_rejected(self):
        windows = iter([
            WindowInfo(1, "Offline Chess", Region(0, 0, 1200, 1000)),
            WindowInfo(2, "Offline Chess", Region(0, 0, 1200, 1000)),
        ])
        guard = SafetyGuard("Chess", self.board, lambda: next(windows))
        guard.verify()
        with self.assertRaisesRegex(SafetyError, "handle"):
            guard.verify()

    def test_context_releases_held_key_after_exception(self):
        backend = FakeBackend()
        with self.assertRaises(RuntimeError):
            with InputController(
                self.guard, enable_live=True, backend=backend, sleeper=lambda _: None
            ) as controller:
                controller.key_down("w")
                raise RuntimeError("test")
        self.assertIn(("keyUp", ("w",), {}), backend.calls)

    def test_drag_always_releases_mouse(self):
        backend = FakeBackend()
        controller = InputController(self.guard, enable_live=True, backend=backend)
        controller.drag((200, 200), (300, 300), duration=0)
        self.assertFalse(controller.held_buttons)
        self.assertEqual("mouseUp", backend.calls[-1][0])

    def test_move_keeps_negative_absolute_screen_coordinates(self):
        board = Region(-900, 100, 800, 800)
        window = lambda: WindowInfo(3, "Offline Chess", Region(-1000, 0, 1000, 1000))
        controller = InputController(SafetyGuard("Chess", board, window))

        ChessInputAdapter(ChessCalibration(board), controller).move("e2", "e4")

        self.assertEqual((-450, 750), controller.actions[0][1:3])
        self.assertEqual((-450, 550), controller.actions[2][1:3])

    def test_pointer_interference_prevents_click_and_releases_inputs(self):
        backend = FakeBackend()
        backend.interfere = True
        controller = InputController(self.guard, enable_live=True, backend=backend)

        with self.assertRaises(PointerInterferenceError):
            controller.click(200, 200, duration=0)

        self.assertFalse(controller.held_buttons)
        self.assertNotIn("click", [name for name, _, _ in backend.calls])


if __name__ == "__main__":
    unittest.main()
