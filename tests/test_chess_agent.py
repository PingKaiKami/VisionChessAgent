import types
import unittest
from unittest.mock import MagicMock

import chess

from game_agent.chess_agent import (
    AgentState,
    ActionSafetyError,
    AutonomousChessAgent,
    BoardSyncError,
    BoardTracker,
    FocusLostError,
    PauseTimeoutError,
    PerceptionUncertainError,
    PointerInterferenceError,
    PromotionController,
    PromotionUIError,
    StockfishEngine,
    ScreenBoardObserver,
    VerificationError,
)
from game_agent.chess_perception import BoardState, SquareRecognition
from game_agent.board_detection import BoardDetection
from game_agent.capture import ScreenFrame
from game_agent.chess import ChessCalibration, Orientation
from game_agent.geometry import Region
from game_agent.safety import SafetyError
import numpy as np


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def visual_state(board: chess.Board, *, unknown: str | None = None) -> BoardState:
    squares = {}
    for square in chess.SQUARES:
        name = chess.square_name(square)
        piece = board.piece_at(square)
        symbol = piece.symbol() if piece else "."
        squares[name] = SquareRecognition("unknown" if name == unknown else symbol, 1.0)
    return BoardState(squares)


class StockfishTests(unittest.TestCase):
    def test_engine_move_is_parsed_and_engine_is_closed(self):
        backend = MagicMock()
        backend.play.return_value = types.SimpleNamespace(
            move=chess.Move.from_uci("e2e4")
        )
        factory = MagicMock(return_value=backend)

        with StockfishEngine("stockfish.exe", 0.1, engine_factory=factory) as engine:
            move = engine.choose_move(chess.Board())

        self.assertEqual("e2e4", move.uci())
        factory.assert_called_once_with("stockfish.exe")
        backend.quit.assert_called_once_with()

    def test_engine_is_closed_on_abnormal_exit(self):
        backend = MagicMock()
        with self.assertRaises(RuntimeError):
            with StockfishEngine(
                "stockfish.exe", 0.1, engine_factory=lambda _: backend
            ):
                raise RuntimeError("test")
        backend.quit.assert_called_once_with()


class BoardTrackerTests(unittest.TestCase):
    def test_infers_e7e5_after_e2e4(self):
        tracker = BoardTracker()
        tracker.board.push_uci("e2e4")
        expected = tracker.board.copy()
        expected.push_uci("e7e5")

        move = tracker.infer_opponent_move(visual_state(expected))

        self.assertEqual("e7e5", move.uci())
        self.assertEqual(expected.fen(), tracker.board.fen())

    def test_board_and_visual_state_stay_synchronized(self):
        tracker = BoardTracker()
        expected = tracker.board.copy()
        expected.push_uci("e2e4")

        moves = tracker.sync_after_agent(
            chess.Move.from_uci("e2e4"), visual_state(expected)
        )

        self.assertEqual(["e2e4"], [move.uci() for move in moves])
        self.assertEqual(expected.fen(), tracker.board.fen())

    def test_fast_opponent_reply_synchronizes_two_plies(self):
        tracker = BoardTracker()
        expected = tracker.board.copy()
        expected.push_uci("e2e4")
        expected.push_uci("e7e5")

        moves = tracker.sync_after_agent(
            chess.Move.from_uci("e2e4"), visual_state(expected)
        )

        self.assertEqual(["e2e4", "e7e5"], [move.uci() for move in moves])
        self.assertEqual(expected.fen(), tracker.board.fen())

    def test_queen_capture_then_immediate_king_recapture_synchronizes(self):
        tracker = BoardTracker()
        tracker.board = chess.Board.empty()
        tracker.board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        tracker.board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        tracker.board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
        tracker.board.set_piece_at(chess.D8, chess.Piece(chess.QUEEN, chess.BLACK))
        expected = tracker.board.copy()
        expected.push_uci("d1d8")
        expected.push_uci("e8d8")

        moves = tracker.sync_after_agent(
            chess.Move.from_uci("d1d8"), visual_state(expected)
        )

        self.assertEqual(["d1d8", "e8d8"], [move.uci() for move in moves])
        self.assertEqual(expected.fen(), tracker.board.fen())

    def test_unknown_state_is_never_guessed(self):
        tracker = BoardTracker()
        tracker.board.push_uci("e2e4")
        with self.assertRaises(PerceptionUncertainError):
            tracker.infer_opponent_move(visual_state(tracker.board, unknown="e7"))

    def test_wrong_agent_result_does_not_update_board(self):
        tracker = BoardTracker()
        wrong = chess.Board()
        wrong.push_uci("d2d4")

        with self.assertRaises(BoardSyncError):
            tracker.sync_after_agent(
                chess.Move.from_uci("e2e4"), visual_state(wrong)
            )

        self.assertEqual(chess.STARTING_FEN, tracker.board.fen())


class AutonomousAgentTests(unittest.TestCase):
    def test_dry_run_reports_move_without_mouse_action(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        agent = AutonomousChessAgent(
            engine,
            lambda: visual_state(chess.Board()),
            adapter,
        )

        result = agent.run(10)

        self.assertEqual("DRY-RUN move: e2e4", result)
        adapter.move.assert_not_called()

    def test_uncertain_initial_position_performs_no_action(self):
        engine = MagicMock()
        adapter = MagicMock()
        agent = AutonomousChessAgent(
            engine,
            lambda: visual_state(chess.Board(), unknown="a1"),
            adapter,
            enable_live=True,
            wait_timeout=0,
        )

        with self.assertRaises(PerceptionUncertainError):
            agent.run(1)

        engine.choose_move.assert_not_called()
        adapter.move.assert_not_called()

    def test_mismatched_result_pauses_then_times_out_after_first_action(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        wrong = chess.Board()
        wrong.push_uci("d2d4")
        initial = visual_state(chess.Board())
        states = iter([initial, initial, visual_state(wrong), visual_state(wrong)])
        agent = AutonomousChessAgent(
            engine,
            lambda: next(states),
            adapter,
            enable_live=True,
            wait_timeout=0,
            settle_delay=0,
            pause_timeout=0,
            sleeper=lambda _: None,
        )

        with self.assertRaises(PauseTimeoutError):
            agent.run(2)

        adapter.move.assert_called_once_with("e2", "e4")

    def test_fast_reply_completes_one_agent_turn_without_desync(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        after_reply = chess.Board()
        after_reply.push_uci("e2e4")
        after_reply.push_uci("e7e5")
        initial = visual_state(chess.Board())
        states = iter([initial, initial, visual_state(after_reply)])
        agent = AutonomousChessAgent(
            engine,
            lambda: next(states),
            adapter,
            enable_live=True,
            wait_timeout=0,
            settle_delay=0,
            sleeper=lambda _: None,
        )

        result = agent.run(1)

        self.assertEqual("Stopped after 1 agent moves", result)
        self.assertEqual(after_reply.fen(), agent.tracker.board.fen())

    def test_animation_requires_consecutive_stable_results(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        initial = visual_state(chess.Board())
        after_move = chess.Board()
        after_move.push_uci("e2e4")
        animation = visual_state(chess.Board(), unknown="e2")
        states = iter([
            initial, initial,
            initial, initial,
            animation,
            visual_state(after_move), visual_state(after_move),
        ])
        agent = AutonomousChessAgent(
            engine,
            lambda: next(states),
            adapter,
            enable_live=True,
            stable_observations=2,
            max_retries=10,
            settle_delay=0,
            poll_interval=0,
            sleeper=lambda _: None,
        )

        result = agent.run(1)

        self.assertEqual("Stopped after 1 agent moves", result)
        self.assertEqual(after_move.fen(), agent.tracker.board.fen())
        adapter.move.assert_called_once_with("e2", "e4")

    def test_profile_records_single_live_precheck_stockfish_and_verification(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        after_move = chess.Board()
        after_move.push_uci("e2e4")
        states = iter([
            visual_state(chess.Board()),
            visual_state(chess.Board()),
            visual_state(after_move),
        ])
        timings = []
        agent = AutonomousChessAgent(
            engine,
            lambda: next(states),
            adapter,
            enable_live=True,
            settle_delay=0,
            sleeper=lambda _: None,
            profiler=lambda stage, elapsed: timings.append((stage, elapsed)),
        )

        agent.run(1)

        self.assertEqual(2, sum(stage == "Precheck" for stage, _ in timings))
        self.assertEqual(1, sum(stage == "Stockfish" for stage, _ in timings))
        self.assertEqual(1, sum(stage == "Move Verification" for stage, _ in timings))

    def test_pointer_interference_times_out_without_committing_or_retrying(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        initial = visual_state(chess.Board())
        agent = AutonomousChessAgent(
            engine,
            lambda: initial,
            adapter,
            enable_live=True,
            post_action_check=lambda _move: False,
            pause_timeout=0,
        )

        with self.assertRaises(PauseTimeoutError):
            agent.run(2)

        adapter.move.assert_called_once_with("e2", "e4")
        self.assertEqual(chess.STARTING_FEN, agent.tracker.board.fen())

    def test_focus_loss_pauses_recovers_board_and_retries_safely(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        initial = chess.Board()
        after = initial.copy(); after.push_uci("e2e4")
        states = iter([
            visual_state(initial), visual_state(initial), visual_state(initial),
            visual_state(initial), visual_state(after),
        ])
        focus_results = iter([False, False, True, True, True])

        def focus_check():
            if not next(focus_results):
                raise SafetyError("Foreground window does not match Chess")

        clock = FakeClock()
        release = MagicMock()
        agent = AutonomousChessAgent(
            engine, lambda: next(states), adapter, enable_live=True,
            pre_action_check=focus_check, release_inputs=release,
            pause_timeout=10, poll_interval=1, settle_delay=0,
            sleeper=clock.sleep, monotonic=clock.monotonic,
        )

        self.assertEqual("Stopped after 1 agent moves", agent.run(1))
        adapter.move.assert_called_once_with("e2", "e4")
        release.assert_called()
        self.assertIn(AgentState.PAUSED_FOCUS, [state for state, _ in agent.state_history])
        self.assertIn(AgentState.RECOVERING, [state for state, _ in agent.state_history])

    def test_partial_action_completed_is_committed_without_second_click(self):
        engine = MagicMock()
        move = chess.Move.from_uci("e2e4")
        engine.choose_move.return_value = move
        adapter = MagicMock()
        after = chess.Board(); after.push(move)
        states = iter([
            visual_state(chess.Board()), visual_state(chess.Board()), visual_state(after),
        ])
        clock = FakeClock()
        agent = AutonomousChessAgent(
            engine, lambda: next(states), adapter, enable_live=True,
            post_action_check=lambda _move: False,
            release_inputs=MagicMock(), pause_timeout=5, poll_interval=1,
            settle_delay=0, sleeper=clock.sleep, monotonic=clock.monotonic,
        )

        agent.run(1)

        adapter.move.assert_called_once_with("e2", "e4")
        self.assertEqual(after.fen(), agent.tracker.board.fen())
        self.assertIn(AgentState.PAUSED_POINTER, [state for state, _ in agent.state_history])

    def test_pointer_recovery_waits_for_idle_period_before_board_check(self):
        clock = FakeClock()
        positions = iter([(0, 0), (10, 10), (10, 10), (10, 10)])
        observed_at = []

        def observe():
            observed_at.append(clock.now)
            return visual_state(chess.Board())

        agent = AutonomousChessAgent(
            MagicMock(), observe, MagicMock(),
            pre_action_check=lambda: None,
            pointer_position=lambda: next(positions), pointer_idle_seconds=2,
            pause_timeout=10, poll_interval=1,
            sleeper=clock.sleep, monotonic=clock.monotonic,
        )

        result = agent._recover_interrupted_move(
            chess.Move.from_uci("e2e4"), AgentState.PAUSED_POINTER, "pointer moved",
        )

        self.assertEqual("retry", result)
        self.assertGreaterEqual(observed_at[0], 4.0)

    def test_unlimited_run_reaches_confirmed_checkmate(self):
        first = chess.Move.from_uci("f2f3")
        second = chess.Move.from_uci("g2g4")
        board0 = chess.Board()
        board1 = board0.copy(); board1.push(first)
        board2 = board1.copy(); board2.push_uci("e7e5")
        board3 = board2.copy(); board3.push(second)
        mate = board3.copy(); mate.push_uci("d8h4")
        states = iter([
            visual_state(board0), visual_state(board0), visual_state(board1),
            visual_state(board2), visual_state(board2), visual_state(board3),
            visual_state(mate),
        ])
        engine = MagicMock()
        engine.choose_move.side_effect = [first, second]
        adapter = MagicMock()
        agent = AutonomousChessAgent(
            engine, lambda: next(states), adapter, enable_live=True,
            settle_delay=0, poll_interval=0, sleeper=lambda _: None,
        )

        result = agent.run(0)

        self.assertTrue(result.startswith("Game over:"))
        self.assertTrue(agent.tracker.board.is_checkmate())
        self.assertEqual(AgentState.FINISHED, agent.state)
        self.assertEqual(2, adapter.move.call_count)

    def test_unknown_recovery_times_out_without_commit(self):
        engine = MagicMock()
        engine.choose_move.return_value = chess.Move.from_uci("e2e4")
        adapter = MagicMock()
        calls = 0

        def observe():
            nonlocal calls
            calls += 1
            return visual_state(chess.Board(), unknown="e2") if calls >= 3 else visual_state(chess.Board())

        clock = FakeClock()
        agent = AutonomousChessAgent(
            engine, observe, adapter, enable_live=True,
            post_action_check=lambda _move: False,
            pause_timeout=2, poll_interval=1, settle_delay=0,
            sleeper=clock.sleep, monotonic=clock.monotonic,
        )

        with self.assertRaises(PauseTimeoutError):
            agent.run(1)

        self.assertEqual(chess.STARTING_FEN, agent.tracker.board.fen())
        self.assertIn(AgentState.PAUSED_MANUAL, [state for state, _ in agent.state_history])


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = ChessCalibration(Region(100, 100, 800, 800))

    @staticmethod
    def menu_images(column=6):
        before = np.zeros((800, 800, 3), np.uint8)
        after = before.copy()
        after[:400, column * 100:(column + 1) * 100] = 220
        return before, after

    def test_all_promotion_pieces_map_down_from_target_square(self):
        expected_y = {chess.QUEEN: 150, chess.KNIGHT: 250, chess.ROOK: 350, chess.BISHOP: 450}
        for piece, y in expected_y.items():
            with self.subTest(piece=piece):
                controller = PromotionController(self.calibration, MagicMock(), MagicMock())
                move = chess.Move(chess.G7, chess.G8, promotion=piece)
                self.assertEqual((750, y), controller.option_point(move))
        controller = PromotionController(self.calibration, MagicMock(), MagicMock())
        self.assertEqual(
            (150, 150), controller.option_point(chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)),
        )

    def test_confirmed_menu_clicks_expected_option_once(self):
        expected_y = {"q": 150, "n": 250, "r": 350, "b": 450}
        for suffix, y in expected_y.items():
            with self.subTest(suffix=suffix):
                before, after = self.menu_images()
                captures = iter([before, after, after])
                click = MagicMock()
                action = MagicMock()
                controller = PromotionController(
                    self.calibration, lambda: next(captures), click,
                    stable_observations=2, sleeper=lambda _: None,
                )
                point = controller.execute(chess.Move.from_uci(f"g7g8{suffix}"), action)
                self.assertEqual((750, y), point)
                action.assert_called_once_with()
                click.assert_called_once_with(750, y)

    def test_missing_or_wrong_column_menu_never_clicks(self):
        before, wrong = self.menu_images(column=5)
        clock = FakeClock()
        click = MagicMock()
        captures = [before]

        def capture():
            if captures:
                return captures.pop()
            return wrong

        controller = PromotionController(
            self.calibration, capture, click, timeout=1, poll_interval=0.5,
            sleeper=clock.sleep, monotonic=clock.monotonic,
        )
        with self.assertRaises(PromotionUIError):
            controller.execute(chess.Move.from_uci("g7g8q"), MagicMock())
        click.assert_not_called()

    def test_unsupported_orientation_pauses_instead_of_guessing(self):
        controller = PromotionController(
            ChessCalibration(Region(0, 0, 800, 800), Orientation.BLACK_BOTTOM),
            MagicMock(), MagicMock(),
        )
        with self.assertRaisesRegex(PromotionUIError, "white-bottom"):
            controller.option_point(chess.Move.from_uci("g7g8q"))

    def test_open_menu_can_resume_without_repeating_pawn_move(self):
        before, after = self.menu_images()
        frames = [before, before, after, after]
        clock = FakeClock()
        click = MagicMock()
        action = MagicMock()
        controller = PromotionController(
            self.calibration, lambda: frames.pop(0), click,
            timeout=0, stable_observations=2,
            sleeper=clock.sleep, monotonic=clock.monotonic,
        )
        move = chess.Move.from_uci("g7g8q")
        with self.assertRaises(PromotionUIError):
            controller.execute(move, action)

        self.assertTrue(controller.resume_existing(move))
        action.assert_called_once_with()
        click.assert_called_once_with(750, 150)

    def test_promotion_focus_loss_does_not_issue_another_option_click(self):
        before, after = self.menu_images()
        captures = iter([before, after])
        click = MagicMock(side_effect=SafetyError("focus lost"))
        controller = PromotionController(
            self.calibration, lambda: next(captures), click,
            stable_observations=1,
        )
        with self.assertRaises(SafetyError):
            controller.execute(chess.Move.from_uci("g7g8q"), MagicMock())
        click.assert_called_once_with(750, 150)

    def test_already_completed_promotion_is_committed_without_reexecution(self):
        board = chess.Board("k7/6P1/8/8/8/8/8/K7 w - - 0 1")
        move = chess.Move.from_uci("g7g8q")
        after = board.copy(); after.push(move)
        clock = FakeClock()
        promotion = MagicMock()
        agent = AutonomousChessAgent(
            MagicMock(), lambda: visual_state(after), MagicMock(),
            pre_action_check=lambda: None, pause_timeout=3, poll_interval=1,
            sleeper=clock.sleep, monotonic=clock.monotonic,
            promotion_controller=promotion,
        )
        agent.tracker.board = board

        recovered = agent._recover_interrupted_move(
            move, AgentState.PAUSED_MANUAL, "test completed promotion",
        )

        self.assertEqual([move], recovered)
        self.assertEqual(after.fen(), agent.tracker.board.fen())
        promotion.resume_existing.assert_not_called()

    def test_promotion_and_fast_reply_match_exact_piece_and_commit_once(self):
        tracker = BoardTracker()
        tracker.board = chess.Board("k7/6P1/8/8/8/8/8/K7 w - - 0 1")
        move = chess.Move.from_uci("g7g8n")
        expected = tracker.board.copy(); expected.push(move)
        reply = next(iter(expected.legal_moves)); expected.push(reply)

        moves = tracker.sync_after_agent(move, visual_state(expected))

        self.assertEqual([move, reply], moves)
        self.assertEqual(expected.fen(), tracker.board.fen())


class ObserverTests(unittest.TestCase):
    def test_cached_capture_does_not_scan_full_desktop(self):
        calibration = ChessCalibration(Region(-800, 100, 400, 400))
        templates = MagicMock()
        expected = visual_state(chess.Board())
        templates.recognize.return_value = expected
        detector = MagicMock()
        observer = ScreenBoardObserver(calibration, templates, detector=detector)
        observer.capture = MagicMock()
        observer.capture.capture.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        self.assertIs(expected, observer())

        observer.capture.capture.assert_called_once_with(calibration.board)
        detector.detect.assert_not_called()

    def test_relocation_updates_cached_calibration(self):
        initial = ChessCalibration(Region(10, 20, 400, 400))
        moved = ChessCalibration(Region(-900, 120, 400, 400))
        state = visual_state(chess.Board())
        detector = MagicMock()
        detector.detect.return_value = BoardDetection(moved, 0.95, state)
        callback = MagicMock()
        observer = ScreenBoardObserver(
            initial,
            MagicMock(),
            detector=detector,
            on_relocated=callback,
        )
        observer.capture = MagicMock()
        observer.capture.capture_frame.return_value = ScreenFrame(
            np.zeros((10, 10, 3), dtype=np.uint8), Region(-1000, 0, 10, 10)
        )

        result = observer.relocate(chess.Board().board_fen())

        self.assertIs(moved, observer.calibration)
        self.assertIs(moved, result.calibration)
        callback.assert_called_once_with(moved)
        detector.detect.assert_called_once_with(
            observer.capture.capture_frame.return_value,
            observer.templates,
            expected_placement=chess.Board().board_fen(),
            orientation=None,
        )


if __name__ == "__main__":
    unittest.main()
