"""Gymnasium environment for classic 2048."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTIONS = (0, 1, 2, 3)  # Up, Right, Down, Left
ACTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
SIZE = 4


def encode_board(board: np.ndarray) -> np.ndarray:
    """Log2-scaled observation in [0, 1], shape (4, 4). Empty cells stay 0."""
    out = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    out[mask] = np.log2(board[mask]) / 16.0
    return out


def _slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Slide and merge one row left. Returns (new_row, score_gained, merges)."""
    nonzero = row[row != 0]
    merges = 0
    score = 0
    merged: list[int] = []
    skip = False
    for i, val in enumerate(nonzero):
        if skip:
            skip = False
            continue
        if i + 1 < len(nonzero) and nonzero[i + 1] == val:
            new_val = val * 2
            merged.append(new_val)
            score += new_val
            merges += 1
            skip = True
        else:
            merged.append(int(val))
    new_row = np.zeros(SIZE, dtype=np.int32)
    new_row[: len(merged)] = merged
    return new_row, score, merges


def move_board(board: np.ndarray, action: int) -> tuple[np.ndarray, int, int, bool]:
    """Apply action. Returns (new_board, score, merges, moved)."""
    rotated = np.rot90(board, k=action)
    score = 0
    merges = 0
    rows = []
    for r in range(SIZE):
        new_row, s, m = _slide_row_left(rotated[r])
        rows.append(new_row)
        score += s
        merges += m
    new_rotated = np.stack(rows)
    new_board = np.rot90(new_rotated, k=-action)
    moved = not np.array_equal(board, new_board)
    return new_board.astype(np.int32), score, merges, moved


def legal_actions(board: np.ndarray) -> list[int]:
    return [a for a in ACTIONS if move_board(board, a)[3]]


def has_moves(board: np.ndarray) -> bool:
    if np.any(board == 0):
        return True
    for i in range(SIZE):
        for j in range(SIZE):
            v = board[i, j]
            if j + 1 < SIZE and board[i, j + 1] == v:
                return True
            if i + 1 < SIZE and board[i + 1, j] == v:
                return True
    return False


class Game2048Env(gym.Env):
    """Classic 4x4 2048 with score reward and optional max-tile shaping."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_tile_bonus: float = 0.0,
        win_tile: int = 2048,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_tile_bonus = max_tile_bonus
        self.win_tile = win_tile
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(SIZE, SIZE), dtype=np.float32
        )
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.steps = 0
        self._np_random: np.random.Generator = np.random.default_rng()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self.score = 0
        self.steps = 0
        self._spawn()
        self._spawn()
        return encode_board(self.board), self._info(0, False)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = int(action)
        new_board, gained, merges, moved = move_board(self.board, action)
        if not moved:
            # Invalid move: small penalty, episode continues if other moves exist
            terminated = not has_moves(self.board)
            return encode_board(self.board), -0.1, terminated, False, self._info(0, False)

        prev_max = int(self.board.max())
        self.board = new_board
        self.score += gained
        self.steps += 1
        self._spawn()

        reward = float(gained)
        new_max = int(self.board.max())
        if self.max_tile_bonus > 0 and new_max > prev_max:
            reward += self.max_tile_bonus * np.log2(new_max)

        terminated = not has_moves(self.board)
        truncated = False
        return encode_board(self.board), reward, terminated, truncated, self._info(merges, moved)

    def _spawn(self) -> None:
        empties = list(zip(*np.where(self.board == 0)))
        if not empties:
            return
        i, j = empties[int(self._np_random.integers(0, len(empties)))]
        self.board[i, j] = 2 if self._np_random.random() < 0.9 else 4

    def _info(self, merges: int, moved: bool) -> dict[str, Any]:
        return {
            "score": self.score,
            "max_tile": int(self.board.max()) if self.board.size else 0,
            "empties": int(np.sum(self.board == 0)),
            "merges": merges,
            "moved": moved,
            "steps": self.steps,
            "board": self.board.copy(),
            "won": bool(self.board.max() >= self.win_tile),
            "legal_actions": legal_actions(self.board),
        }

    def render(self) -> Optional[str]:
        lines = ["+" + ("------+" * SIZE)]
        for row in self.board:
            cells = [f"{v:^6}" if v else "      " for v in row]
            lines.append("|" + "|".join(cells) + "|")
            lines.append("+" + ("------+" * SIZE))
        return "\n".join(lines)

    def clone_board(self) -> np.ndarray:
        return self.board.copy()
