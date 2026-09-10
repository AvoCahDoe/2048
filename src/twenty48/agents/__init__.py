from twenty48.agents.heuristic import HeuristicAgent
from twenty48.agents.random_agent import RandomAgent
from twenty48.agents.expectimax import ExpectimaxAgent


def get_agent(name: str, **kwargs):
    key = name.lower().replace("-", "_")
    if key in ("random",):
        return RandomAgent(**kwargs)
    if key in ("heuristic", "snake", "greedy"):
        return HeuristicAgent(**kwargs)
    if key in ("expectimax", "expecti"):
        return ExpectimaxAgent(**kwargs)
    if key in ("dqn",):
        from twenty48.agents.dqn_agent import DQNAgent

        return DQNAgent(**kwargs)
    raise ValueError(f"Unknown agent: {name}")


__all__ = [
    "get_agent",
    "RandomAgent",
    "HeuristicAgent",
    "ExpectimaxAgent",
]
