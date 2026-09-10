"""Random legal-move baseline."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from twenty48.env.game2048 import ACTIONS, legal_actions


class RandomAgent:
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> int:
        if info and "legal_actions" in info and info["legal_actions"]:
            choices = info["legal_actions"]
        elif info and "board" in info:
            choices = legal_actions(info["board"])
        else:
            choices = list(ACTIONS)
        if not choices:
            return int(self._rng.integers(0, 4))
        return int(self._rng.choice(choices))
