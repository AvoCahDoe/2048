"""Expectimax search with probabilistic tile spawns."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from twenty48.agents.heuristic import evaluate_board
from twenty48.env.game2048 import legal_actions, move_board


class ExpectimaxAgent:
    """Shallow expectimax. depth=1 is fast; depth=2 for stronger offline runs."""

    name = "expectimax"

    def __init__(
        self,
        depth: int = 1,
        seed: Optional[int] = None,
        max_empties: int = 4,
    ):
        self.depth = depth
        self.max_empties = max_empties
        self._rng = np.random.default_rng(seed)
        self._cache: dict[tuple, float] = {}

    def reset(self) -> None:
        self._cache.clear()

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> int:
        board = info["board"] if info and "board" in info else None
        if board is None:
            raise ValueError("ExpectimaxAgent requires info['board']")
        self._cache.clear()
        actions = legal_actions(board)
        if not actions:
            return 0
        best_score = -1e18
        best: list[int] = []
        for a in actions:
            new_board, gained, _, _ = move_board(board, a)
            # After a player move the env spawns a tile → chance node
            score = gained + self._chance(new_board, self.depth)
            if score > best_score + 1e-9:
                best_score = score
                best = [a]
            elif abs(score - best_score) < 1e-9:
                best.append(a)
        return int(self._rng.choice(best))

    def _chance(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), depth, "c")
        if key in self._cache:
            return self._cache[key]
        empties = list(zip(*np.where(board == 0)))
        if not empties:
            val = evaluate_board(board)
            self._cache[key] = val
            return val
        # Prefer sparse empties (harder positions); sample if many
        if len(empties) > self.max_empties:
            idx = self._rng.choice(len(empties), size=self.max_empties, replace=False)
            empties = [empties[int(i)] for i in idx]
        total = 0.0
        p_cell = 1.0 / len(empties)
        for i, j in empties:
            for tile, p_tile in ((2, 0.9), (4, 0.1)):
                b2 = board.copy()
                b2[i, j] = tile
                if depth <= 1:
                    total += p_cell * p_tile * evaluate_board(b2)
                else:
                    total += p_cell * p_tile * self._max_node(b2, depth - 1)
        self._cache[key] = total
        return total

    def _max_node(self, board: np.ndarray, depth: int) -> float:
        key = (board.tobytes(), depth, "m")
        if key in self._cache:
            return self._cache[key]
        actions = legal_actions(board)
        if not actions:
            val = evaluate_board(board)
            self._cache[key] = val
            return val
        best = -1e18
        for a in actions:
            new_board, gained, _, moved = move_board(board, a)
            if not moved:
                continue
            best = max(best, gained + self._chance(new_board, depth))
        if best < -1e17:
            best = evaluate_board(board)
        self._cache[key] = best
        return best
