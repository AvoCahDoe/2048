"""Shared agent protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Agent(Protocol):
    name: str

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> int:
        ...

    def reset(self) -> None:
        ...


def get_agent(name: str, **kwargs: Any) -> Any:
    from twenty48.agents.dqn_agent import DQNAgent
    from twenty48.agents.expectimax import ExpectimaxAgent
    from twenty48.agents.heuristic import HeuristicAgent
    from twenty48.agents.random_agent import RandomAgent

    key = name.lower().replace("-", "_")
    if key in ("random",):
        return RandomAgent(**kwargs)
    if key in ("heuristic", "snake", "greedy"):
        return HeuristicAgent(**kwargs)
    if key in ("expectimax", "expecti"):
        return ExpectimaxAgent(**kwargs)
    if key in ("dqn",):
        return DQNAgent(**kwargs)
    raise ValueError(f"Unknown agent: {name}")
