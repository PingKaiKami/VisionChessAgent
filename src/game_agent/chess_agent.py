from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import cv2
import numpy as np

from .capture import ScreenCapture
from .board_detection import BoardDetection, BoardDetectionError, BoardDetector
from .chess import ChessCalibration, ChessInputAdapter, Orientation
from .chess_perception import BoardState, TemplateLibrary, save_debug_image
from .input_controller import PointerInterferenceError as InputPointerInterferenceError
from .safety import SafetyError


class ChessAgentError(RuntimeError):
    pass


class PerceptionUncertainError(ChessAgentError):
    pass


class BoardSyncError(ChessAgentError):
    pass


class VerificationError(ChessAgentError):
    pass


class OpponentTimeoutError(ChessAgentError):
    pass


class ManualInterventionRequired(ChessAgentError):
    pass


class ActionSafetyError(ChessAgentError):
    pass


class PendingBoardChange(BoardSyncError):
    pass


class FocusLostError(ActionSafetyError):
    pass


class PointerInterferenceError(ActionSafetyError):
    pass


class PauseTimeoutError(ChessAgentError):
    pass


class PromotionUIError(ActionSafetyError):
    pass


class AgentState(str, Enum):
    RUNNING = "RUNNING"
    PAUSED_FOCUS = "PAUSED_FOCUS"
    PAUSED_POINTER = "PAUSED_POINTER"
    PAUSED_MANUAL = "PAUSED_MANUAL"
    RECOVERING = "RECOVERING"
    FINISHED = "FINISHED"
    TIMED_OUT = "TIMED_OUT"
    STOPPED = "STOPPED"


PROMOTION_INDEX = {
    chess.QUEEN: 0,
    chess.KNIGHT: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
}


class PromotionController:
    """Detect and operate the verified white-bottom Chess.com promotion menu."""

    def __init__(
        self,
        calibration: ChessCalibration,
        capture_board: Callable[[], np.ndarray],
        click: Callable[..., None],
        *,
        poll_interval: float = 0.2,
        timeout: float = 5.0,
        stable_observations: int = 2,
        sleeper=time.sleep,
        monotonic=time.monotonic,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.calibration = calibration
        self.capture_board = capture_board
        self.click = click
        self.poll_interval = max(0.0, poll_interval)
        self.timeout = max(0.0, timeout)
        self.stable_observations = max(1, stable_observations)
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.logger = logger
        self._before: np.ndarray | None = None

    def option_point(self, move: chess.Move) -> tuple[int, int]:
        if self.calibration.orientation is not Orientation.WHITE_BOTTOM:
            raise PromotionUIError("Unsupported promotion orientation; white-bottom is required")
        if chess.square_rank(move.to_square) != 7 or move.promotion not in PROMOTION_INDEX:
            raise PromotionUIError(f"Unsupported promotion move: {move.uci()}")
        x, y = self.calibration.square_center(chess.square_name(move.to_square))
        y += round(PROMOTION_INDEX[move.promotion] * self.calibration.board.height / 8)
        point = (x, y)
        if not self.calibration.board.contains_point(*point):
            raise PromotionUIError(f"Promotion option is outside the board: {point}")
        return point

    def detect_menu(
        self, before: np.ndarray, after: np.ndarray, move: chess.Move,
    ) -> tuple[bool, dict[str, object]]:
        if before.shape != after.shape or before.ndim != 3:
            return False, {"reason": "capture-shape-mismatch"}
        if self.calibration.orientation is not Orientation.WHITE_BOTTOM:
            return False, {"reason": "unsupported-orientation"}
        height, width = after.shape[:2]
        column = chess.square_file(move.to_square)
        x0, x1 = round(column * width / 8), round((column + 1) * width / 8)
        changed_fractions: list[float] = []
        mean_differences: list[float] = []
        for index in range(4):
            y0, y1 = round(index * height / 8), round((index + 1) * height / 8)
            difference = cv2.absdiff(before[y0:y1, x0:x1], after[y0:y1, x0:x1])
            magnitude = np.max(difference.astype(np.float32), axis=2)
            changed_fractions.append(float(np.mean(magnitude >= 15.0)))
            mean_differences.append(float(np.mean(magnitude)))
        changed_cells = sum(
            fraction >= 0.08 and mean >= 8.0
            for fraction, mean in zip(changed_fractions, mean_differences)
        )
        found = changed_cells >= 3
        return found, {
            "column": column, "changed_cells": changed_cells,
            "changed_fractions": changed_fractions,
            "mean_differences": mean_differences,
        }

    def execute(self, move: chess.Move, move_action: Callable[[], None]) -> tuple[int, int]:
        point = self.option_point(move)
        before = self.capture_board()
        self._before = before
        move_action()
        deadline = self.monotonic() + self.timeout
        stable = 0
        last_metrics: dict[str, object] = {"reason": "not-observed"}
        while True:
            after = self.capture_board()
            found, last_metrics = self.detect_menu(before, after, move)
            stable = stable + 1 if found else 0
            if stable >= self.stable_observations:
                if self.logger:
                    self.logger(
                        f"Promotion menu confirmed for {move.uci()}; "
                        f"piece={chess.piece_name(move.promotion)}; point={point}; metrics={last_metrics}"
                    )
                self.click(*point)
                self._before = None
                return point
            if self.monotonic() >= deadline:
                raise PromotionUIError(
                    f"Promotion menu was not confirmed at the expected column: {last_metrics}"
                )
            self.sleeper(self.poll_interval)

    def resume_existing(self, move: chess.Move) -> bool:
        """Finish a still-open, visually reconfirmed menu without repeating the pawn move."""
        if self._before is None:
            return False
        stable = 0
        last_metrics: dict[str, object] = {}
        for index in range(self.stable_observations):
            found, last_metrics = self.detect_menu(self._before, self.capture_board(), move)
            stable = stable + 1 if found else 0
            if stable == 0:
                return False
            if index + 1 < self.stable_observations:
                self.sleeper(self.poll_interval)
        point = self.option_point(move)
        if self.logger:
            self.logger(
                f"Recovered open promotion menu for {move.uci()}; point={point}; "
                f"metrics={last_metrics}"
            )
        self.click(*point)
        self._before = None
        return True


class StockfishEngine:
    def __init__(self, path: str, think_time: float, *, engine_factory=None) -> None:
        self.path = path
        self.think_time = think_time
        self.engine_factory = engine_factory or chess.engine.SimpleEngine.popen_uci
        self._engine = None

    def __enter__(self) -> "StockfishEngine":
        self._engine = self.engine_factory(self.path)
        return self

    def choose_move(self, board: chess.Board) -> chess.Move:
        if self._engine is None:
            raise ChessAgentError("Stockfish engine is not running")
        result = self._engine.play(board, chess.engine.Limit(time=self.think_time))
        if result.move is None or result.move not in board.legal_moves:
            raise ChessAgentError("Stockfish did not return a legal move")
        return result.move

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class BoardTracker:
    def __init__(self) -> None:
        self.board = chess.Board()

    @staticmethod
    def _visual_placement(state: BoardState) -> str:
        if state.unknown_squares:
            raise PerceptionUncertainError(
                f"Unknown squares: {', '.join(state.unknown_squares)}"
            )
        return state.piece_placement()

    def validate_initial(self, state: BoardState) -> None:
        if self._visual_placement(state) != chess.Board().board_fen():
            raise BoardSyncError("Visual board is not the standard initial position")

    def matches_current(self, state: BoardState) -> bool:
        return self._visual_placement(state) == self.board.board_fen()

    def match_opponent_move(self, state: BoardState) -> chess.Move:
        placement = self._visual_placement(state)
        matches: list[chess.Move] = []
        for move in self.board.legal_moves:
            candidate = self.board.copy(stack=False)
            candidate.push(move)
            if candidate.board_fen() == placement:
                matches.append(move)
        if len(matches) != 1:
            raise BoardSyncError(f"Expected one legal opponent move, found {len(matches)}")
        return matches[0]

    def match_after_agent(self, expected: chess.Move, state: BoardState) -> list[chess.Move]:
        if expected not in self.board.legal_moves:
            raise BoardSyncError(f"Expected move is not legal: {expected.uci()}")
        placement = self._visual_placement(state)
        after_agent = self.board.copy(stack=False)
        after_agent.push(expected)
        matches: list[list[chess.Move]] = []
        if after_agent.board_fen() == placement:
            matches.append([expected])
        for reply in after_agent.legal_moves:
            after_reply = after_agent.copy(stack=False)
            after_reply.push(reply)
            if after_reply.board_fen() == placement:
                matches.append([expected, reply])
        if len(matches) != 1:
            raise BoardSyncError(f"Expected one matching result sequence, found {len(matches)}")
        return matches[0]

    def commit(self, moves: list[chess.Move]) -> None:
        for move in moves:
            if move not in self.board.legal_moves:
                raise BoardSyncError(f"Cannot commit illegal move: {move.uci()}")
            self.board.push(move)

    def infer_opponent_move(self, state: BoardState) -> chess.Move:
        move = self.match_opponent_move(state)
        self.commit([move])
        return move

    def sync_after_agent(self, expected: chess.Move, state: BoardState) -> list[chess.Move]:
        moves = self.match_after_agent(expected, state)
        self.commit(moves)
        return moves


class ScreenBoardObserver:
    def __init__(
        self,
        calibration: ChessCalibration,
        templates: TemplateLibrary,
        *,
        debug_directory: str | Path | None = None,
        detector: BoardDetector | None = None,
        orientation: Orientation | None = None,
        on_relocated: Callable[[ChessCalibration], None] | None = None,
        profiler: Callable[[str, float], None] | None = None,
    ) -> None:
        self.calibration = calibration
        self.templates = templates
        self.capture = ScreenCapture()
        self.debug_directory = Path(debug_directory) if debug_directory else None
        self.detector = detector
        self.orientation = orientation
        self.on_relocated = on_relocated
        self.profiler = profiler
        self._sequence = 0

    def capture_board_image(self) -> np.ndarray:
        return self.capture.capture(self.calibration.board)

    def __call__(self) -> BoardState:
        started = time.perf_counter()
        image = self.capture_board_image()
        if self.profiler:
            self.profiler("Full Screenshot", time.perf_counter() - started)
        started = time.perf_counter()
        state = self.templates.recognize(image, self.calibration)
        if self.profiler:
            self.profiler("Piece Recognition", time.perf_counter() - started)
        if self.debug_directory:
            self._sequence += 1
            save_debug_image(
                self.debug_directory / f"board-{self._sequence:04d}.png",
                image,
                self.calibration,
                state,
            )
        return state

    def relocate(self, expected_placement: str) -> BoardDetection:
        if self.detector is None:
            raise BoardDetectionError("Automatic board detection is disabled")
        started = time.perf_counter()
        frame = self.capture.capture_frame()
        if self.profiler:
            self.profiler("Full Screenshot", time.perf_counter() - started)
        result = self.detector.detect(
            frame,
            self.templates,
            expected_placement=expected_placement,
            orientation=self.orientation,
        )
        self.calibration = result.calibration
        if self.on_relocated:
            self.on_relocated(result.calibration)
        return result


class AutonomousChessAgent:
    def __init__(
        self,
        engine: StockfishEngine,
        observer: Callable[[], BoardState],
        adapter: ChessInputAdapter,
        *,
        enable_live: bool = False,
        poll_interval: float = 1.0,
        wait_timeout: float = 60.0,
        settle_delay: float = 1.0,
        max_retries: int = 20,
        stable_observations: int = 1,
        pre_action_check: Callable[[], object] | None = None,
        post_action_check: Callable[[chess.Move], bool] | None = None,
        pause_timeout: float = 300.0,
        pointer_idle_seconds: float = 2.0,
        pointer_position: Callable[[], tuple[int, int] | None] | None = None,
        release_inputs: Callable[[], None] | None = None,
        promotion_controller: PromotionController | None = None,
        promotion_mode: str = "auto",
        sleeper=time.sleep,
        monotonic=time.monotonic,
        profiler: Callable[[str, float], None] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.engine = engine
        self.observer = observer
        self.adapter = adapter
        self.enable_live = enable_live
        self.poll_interval = max(0.0, poll_interval)
        self.wait_timeout = max(0.0, wait_timeout)
        self.settle_delay = max(0.0, settle_delay)
        self.stable_observations = max(1, stable_observations)
        self.max_retries = max(self.stable_observations, max_retries)
        self.pre_action_check = pre_action_check
        self.post_action_check = post_action_check
        self.pause_timeout = max(0.0, pause_timeout)
        self.pointer_idle_seconds = max(0.0, pointer_idle_seconds)
        self.pointer_position = pointer_position
        self.release_inputs = release_inputs
        self.promotion_controller = promotion_controller
        if promotion_mode not in {"auto", "manual"}:
            raise ValueError("promotion_mode must be auto or manual")
        self.promotion_mode = promotion_mode
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.profiler = profiler
        self.logger = logger
        self.tracker = BoardTracker()
        self.state = AgentState.RUNNING
        self.state_history: list[tuple[AgentState, str]] = [(self.state, "initialized")]

    def _transition(self, state: AgentState, reason: str) -> None:
        if state == self.state and self.state_history[-1][1] == reason:
            return
        self.state = state
        self.state_history.append((state, reason))
        if self.logger:
            self.logger(f"Agent state={state.value}; reason={reason}")

    def _release_inputs(self) -> None:
        if self.release_inputs:
            self.release_inputs()

    def _check_focus(self) -> bool:
        if not self.pre_action_check:
            return True
        try:
            self.pre_action_check()
            return True
        except RuntimeError:
            return False

    @staticmethod
    def _positions_near(
        left: tuple[int, int] | None, right: tuple[int, int] | None, tolerance: int = 2,
    ) -> bool:
        if left is None or right is None:
            return left == right
        return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance

    def _recover_interrupted_move(
        self, move: chess.Move, paused_state: AgentState, reason: str,
    ) -> str | list[chess.Move]:
        self._release_inputs()
        self._transition(paused_state, reason)
        deadline = self.monotonic() + self.pause_timeout
        focus_stable = 0
        pointer_last: tuple[int, int] | None = None
        pointer_stable_since: float | None = None
        while True:
            now = self.monotonic()
            ready_to_check = False
            if self._check_focus():
                focus_stable += 1
            else:
                focus_stable = 0
                self._transition(AgentState.PAUSED_FOCUS, "waiting for Chess window focus")
            if focus_stable >= max(2, self.stable_observations):
                if paused_state is AgentState.PAUSED_POINTER and self.pointer_position:
                    current = self.pointer_position()
                    if self._positions_near(current, pointer_last):
                        pointer_stable_since = pointer_stable_since or now
                    else:
                        pointer_last = current
                        pointer_stable_since = now
                    if now - pointer_stable_since < self.pointer_idle_seconds:
                        self._transition(AgentState.PAUSED_POINTER, "waiting for pointer to become idle")
                    else:
                        ready_to_check = True
                else:
                    ready_to_check = True

                if ready_to_check:
                    if self.state is not AgentState.PAUSED_MANUAL:
                        self._transition(AgentState.RECOVERING, f"rechecking {move.uci()}")
                    try:
                        state = self.observer()
                        if self.tracker.matches_current(state):
                            self._transition(AgentState.RUNNING, f"{move.uci()} was not executed; safe to retry")
                            return "retry"
                        moves = self.tracker.match_after_agent(move, state)
                        self.tracker.commit(moves)
                        self._transition(
                            AgentState.RUNNING,
                            f"recovered completed sequence: {' '.join(item.uci() for item in moves)}",
                        )
                        return moves
                    except (ChessAgentError, BoardDetectionError) as error:
                        self._transition(AgentState.PAUSED_MANUAL, f"board remains uncertain: {error}")
                        if move.promotion and self.promotion_controller and self.promotion_mode == "auto":
                            try:
                                if self.promotion_controller.resume_existing(move):
                                    self._transition(
                                        AgentState.RECOVERING,
                                        f"completed confirmed open promotion menu for {move.uci()}",
                                    )
                            except SafetyError:
                                self._transition(
                                    AgentState.PAUSED_FOCUS,
                                    "promotion menu is open but Chess window is not focused",
                                )
                            except InputPointerInterferenceError:
                                self._transition(
                                    AgentState.PAUSED_POINTER,
                                    "pointer interference while resuming promotion",
                                )
            if now >= deadline:
                self._transition(AgentState.TIMED_OUT, f"pause timeout after {self.pause_timeout:g}s")
                self._release_inputs()
                raise PauseTimeoutError(
                    f"Unable to recover {move.uci()} within {self.pause_timeout:g} seconds"
                )
            self._sleep(max(0.05, self.poll_interval))

    def _record(self, stage: str, started: float) -> None:
        if self.profiler:
            self.profiler(stage, time.perf_counter() - started)

    def _sleep(self, duration: float) -> None:
        started = time.perf_counter()
        self.sleeper(duration)
        self._record("Sleep / Polling", started)

    @staticmethod
    def _value_key(value) -> tuple:
        if isinstance(value, chess.Move):
            return (value.uci(),)
        if isinstance(value, list):
            return tuple(move.uci() for move in value)
        return (str(value),)

    def _stable_match(
        self,
        matcher,
        error_type,
        message: str,
        *,
        allow_relocate: bool = True,
        preserve_uncertain: bool = False,
    ):
        deadline = self.monotonic() + self.wait_timeout
        attempts = 0
        relocation_failures = 0
        stable_count = 0
        last_key = None
        last_error: Exception | None = None
        relocated = False
        while True:
            try:
                state = self.observer()
                value = matcher(state)
                key = (state.piece_placement(), self._value_key(value))
                stable_count = stable_count + 1 if key == last_key else 1
                last_key = key
                if stable_count >= self.stable_observations:
                    return value
            except (ChessAgentError, BoardDetectionError) as error:
                last_error = error
                stable_count = 0
                last_key = None
                if not isinstance(error, PendingBoardChange):
                    relocation_failures += 1
            attempts += 1
            if allow_relocate and not relocated and relocation_failures >= 2 and hasattr(self.observer, "relocate"):
                try:
                    self.observer.relocate(self.tracker.board.board_fen())
                    relocated = True
                    attempts = 0
                    continue
                except BoardDetectionError as error:
                    last_error = error
                    relocated = True
            if attempts >= self.max_retries or self.monotonic() >= deadline:
                if preserve_uncertain and isinstance(last_error, PerceptionUncertainError):
                    raise last_error
                raise error_type(f"{message}: {last_error}")
            self._sleep(self.poll_interval)

    def _match_current(self, state: BoardState) -> str:
        if not self.tracker.matches_current(state):
            raise BoardSyncError("Visual board differs from internal board")
        return self.tracker.board.board_fen()

    def _precheck(self) -> None:
        started = time.perf_counter()
        self._stable_match(
            self._match_current,
            BoardSyncError,
            "Precheck failed",
        )
        if self.pre_action_check:
            try:
                self.pre_action_check()
            except SafetyError as error:
                raise FocusLostError(f"Action safety check failed: {error}") from error
            except RuntimeError as error:
                raise FocusLostError(f"Action safety check failed: {error}") from error
        self._record("Precheck", started)

    def _wait_after_agent(self, expected: chess.Move) -> list[chess.Move]:
        self._sleep(self.settle_delay)
        moves = self._stable_match(
            lambda state: self.tracker.match_after_agent(expected, state),
            VerificationError,
            f"Unable to verify {expected.uci()}",
            allow_relocate=False,
        )
        self.tracker.commit(moves)
        return moves

    def _wait_for_opponent(self) -> chess.Move:
        def match_changed(state: BoardState) -> chess.Move:
            if self.tracker.matches_current(state):
                raise PendingBoardChange("Opponent has not moved")
            return self.tracker.match_opponent_move(state)

        move = self._stable_match(
            match_changed,
            OpponentTimeoutError,
            "Unable to infer opponent move",
        )
        self.tracker.commit([move])
        return move

    def _act(self, move: chess.Move) -> None:
        try:
            action = lambda: self.adapter.move(
                chess.square_name(move.from_square), chess.square_name(move.to_square)
            )
            if move.promotion:
                if self.promotion_mode != "auto" or self.promotion_controller is None:
                    raise ManualInterventionRequired(
                        f"Promotion requires manual handling: {move.uci()}"
                    )
                self.promotion_controller.execute(move, action)
            else:
                action()
            if not move.promotion and self.post_action_check and not self.post_action_check(move):
                raise PointerInterferenceError("Pointer position changed during move execution")
        except SafetyError as error:
            raise FocusLostError(f"Chess window lost focus during move: {error}") from error
        except InputPointerInterferenceError as error:
            raise PointerInterferenceError(str(error)) from error
        except ActionSafetyError:
            raise
        except ManualInterventionRequired:
            raise
        except RuntimeError as error:
            raise FocusLostError(f"Move safety check failed: {error}") from error
        except Exception as error:
            raise ActionSafetyError(f"Move execution failed: {error}") from error

    def _verify(self, move: chess.Move) -> list[chess.Move]:
        started = time.perf_counter()
        result = self._wait_after_agent(move)
        self._record("Move Verification", started)
        return result

    def run(self, max_moves: int) -> str:
        if max_moves < 0:
            raise ValueError("max_moves must be zero or greater")
        try:
            started = time.perf_counter()
            self._stable_match(
                lambda state: self.tracker.validate_initial(state) or "initial",
                BoardSyncError,
                "Initial board verification failed",
                preserve_uncertain=True,
            )
            self._record("Precheck", started)
            agent_moves = 0
            while (
                (max_moves == 0 or agent_moves < max_moves)
                and not self.tracker.board.is_game_over(claim_draw=False)
            ):
                if self.tracker.board.turn == chess.BLACK:
                    try:
                        self._wait_for_opponent()
                    except OpponentTimeoutError:
                        self._transition(AgentState.TIMED_OUT, "opponent move timeout")
                        raise
                    continue

                started = time.perf_counter()
                move = self.engine.choose_move(self.tracker.board)
                self._record("Stockfish", started)
                if self.logger:
                    detail = (
                        f"; promotion={chess.piece_name(move.promotion)}"
                        if move.promotion else ""
                    )
                    self.logger(f"Stockfish move={move.uci()}{detail}")
                if not self.enable_live:
                    return f"DRY-RUN move: {move.uci()}"

                completed = False
                while not completed:
                    if move.promotion and self.promotion_mode == "manual":
                        recovery = self._recover_interrupted_move(
                            move, AgentState.PAUSED_MANUAL,
                            f"manual promotion requested for {move.uci()}",
                        )
                    else:
                        try:
                            self._precheck()
                            self._act(move)
                            self._verify(move)
                            recovery = [move]
                        except FocusLostError as error:
                            recovery = self._recover_interrupted_move(
                                move, AgentState.PAUSED_FOCUS, str(error),
                            )
                        except PointerInterferenceError as error:
                            recovery = self._recover_interrupted_move(
                                move, AgentState.PAUSED_POINTER, str(error),
                            )
                        except (PromotionUIError, ManualInterventionRequired, VerificationError) as error:
                            recovery = self._recover_interrupted_move(
                                move, AgentState.PAUSED_MANUAL, str(error),
                            )
                    if recovery == "retry":
                        continue
                    completed = True
                    agent_moves += 1

            if self.tracker.board.is_game_over(claim_draw=False):
                self._transition(AgentState.FINISHED, "python-chess confirmed board-state game over")
                return f"Game over: {self.tracker.board.outcome(claim_draw=False)}"
            self._transition(AgentState.STOPPED, f"maximum agent moves reached: {agent_moves}")
            return f"Stopped after {agent_moves} agent moves"
        except KeyboardInterrupt:
            self._transition(AgentState.STOPPED, "stopped by user")
            self._release_inputs()
            raise
        except ChessAgentError:
            if self.state not in {AgentState.TIMED_OUT, AgentState.PAUSED_MANUAL}:
                self._transition(AgentState.STOPPED, "unrecoverable agent error")
            self._release_inputs()
            raise
