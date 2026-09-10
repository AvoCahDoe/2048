"""Classic 2048 heuristic (empty / monotonicity / smoothness / corner)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from twenty48.env.game2048 import legal_actions, move_board


# Snake / corner weight matrix (prefer high tiles in a descending snake)
SNAKE = np.array(
    [
        [15, 14, 13, 12],
        [8, 9, 10, 11],
        [7, 6, 5, 4],
        [0, 1, 2, 3],
    ],
    dtype=np.float64,
)


def evaluate_board(board: np.ndarray) -> float:
    """Scalar board evaluation used by greedy and expectimax."""
    empties = float(np.sum(board == 0))
    snake = float(np.sum(board * SNAKE))

    # Smoothness: penalize large adjacent differences
    smooth = 0.0
    for i in range(4):
        for j in range(4):
            if j + 1 < 4:
                smooth -= abs(float(board[i, j]) - float(board[i, j + 1]))
            if i + 1 < 4:
                smooth -= abs(float(board[i, j]) - float(board[i + 1, j]))

    # Monotonicity along rows/cols (prefer non-increasing toward corner)
    mono = 0.0
    for i in range(4):
        row = board[i]
        mono += _mono_line(row)
        col = board[:, i]
        mono += _mono_line(col)

    max_tile = float(board.max())
    return (
        2.7 * empties
        + 1.0 * snake
        + 0.1 * smooth
        + 1.0 * mono
        + 0.5 * np.log2(max_tile + 1)
    )


def _mono_line(line: np.ndarray) -> float:
    inc = 0.0
    dec = 0.0
    for i in range(3):
        a, b = float(line[i]), float(line[i + 1])
        if a > b:
            dec += a - b
        elif b > a:
            inc += b - a
    return -min(inc, dec)


class HeuristicAgent:
    name = "heuristic"

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> int:
        board = info["board"] if info and "board" in info else None
        if board is None:
            raise ValueError("HeuristicAgent requires info['board']")
        actions = legal_actions(board)
        if not actions:
            return 0
        best_score = -1e18
        best: list[int] = []
        for a in actions:
            new_board, gained, _, _ = move_board(board, a)
            score = evaluate_board(new_board) + 0.1 * gained
            if score > best_score + 1e-9:
                best_score = score
                best = [a]
            elif abs(score - best_score) < 1e-9:
                best.append(a)
        return int(self._rng.choice(best))
