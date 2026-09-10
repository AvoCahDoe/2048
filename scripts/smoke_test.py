"""Smoke-test env + agents after install."""

from twenty48.agents import get_agent
from twenty48.env import Game2048Env
from twenty48.sim import run_episode


def main() -> None:
    env = Game2048Env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 4)
    agent = get_agent("heuristic", seed=0)
    r = run_episode(agent, seed=1)
    print(f"Smoke OK — heuristic score={r.score} max_tile={r.max_tile} steps={r.steps}")


if __name__ == "__main__":
    main()
