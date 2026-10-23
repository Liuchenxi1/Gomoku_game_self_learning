"""Minimal, framework-independent Gomoku environment for AlphaZero.

Rules: freestyle Gomoku (5 or more consecutive stones wins, no forbidden moves).
Board: 0 empty, +1 black, -1 white. Black moves first.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StepResult:
    winner: int | None  # +1 black, -1 white, 0 draw, None ongoing
    terminated: bool


class GomokuEnv:
    def __init__(self, board_size: int = 9, win_length: int = 5):
        if board_size < 1 or not (2 <= win_length <= board_size):
            raise ValueError("Require board_size >= win_length >= 2")
        self.size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.winner: int | None = None
        self.terminated = False
        self.move_count = 0
        self.last_move: tuple[int, int] | None = None
        return self.observation()

    def observation(self) -> np.ndarray:
        """(3, H, W) float32 planes from the player-to-move's perspective.

        Plane 0: own stones; plane 1: opponent stones;
        plane 2: all ones if current player is black, otherwise zeros.
        """
        own = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        to_move = np.full((self.size, self.size), float(self.current_player == 1), dtype=np.float32)
        return np.stack((own, opponent, to_move), axis=0)

    def legal_moves(self) -> np.ndarray:
        """Return flattened zero-based action indices, row * size + column."""
        if self.terminated:
            return np.array([], dtype=np.int64)
        return np.flatnonzero(self.board.ravel() == 0)

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.size * self.size, dtype=np.bool_)
        mask[self.legal_moves()] = True
        return mask

    def step(self, action: int) -> StepResult:
        if self.terminated:
            raise ValueError("Game has ended; call reset() before playing again")
        if not isinstance(action, (int, np.integer)) or not (0 <= action < self.size * self.size):
            raise ValueError(f"Invalid action index: {action}")
        row, col = divmod(int(action), self.size)
        if self.board[row, col] != 0:
            raise ValueError(f"Cell ({row}, {col}) is occupied")

        played_by = self.current_player
        self.board[row, col] = played_by
        self.move_count += 1
        self.last_move = (row, col)

        if self._is_winning_move(row, col, played_by):
            self.winner = played_by
            self.terminated = True
        elif self.move_count == self.size * self.size:
            self.winner = 0
            self.terminated = True

        # Advance only if ongoing: keeps terminal observation defined from
        # the last player's point of view. Training labels can use winner.
        if not self.terminated:
            self.current_player = -played_by
        return StepResult(winner=self.winner, terminated=self.terminated)

    def _is_winning_move(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_length:
                return True
        return False

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        header = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        lines = [header]
        for r, cells in enumerate(self.board):
            lines.append(f"{r:2d} " + " ".join(f"{symbols[int(v)]:>2}" for v in cells))
        return "\n".join(lines)
