"""Monte Carlo batch rollouts and bakeoff comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from twenty48.agents import get_agent
from twenty48.env.game2048 import Game2048Env


@dataclass
class EpisodeResult:
    agent: str
    episode: int
    seed: int
    score: int
    max_tile: int
    steps: int
    won: bool
    first_512: Optional[int]
    first_1024: Optional[int]
    first_2048: Optional[int]
    actions: str  # comma-separated action ids


def run_episode(
    agent: Any,
    seed: int,
    max_steps: int = 10_000,
    log_actions: bool = True,
) -> EpisodeResult:
    env = Game2048Env()
    obs, info = env.reset(seed=seed)
    agent.reset()
    first_512 = first_1024 = first_2048 = None
    actions: list[int] = []

    for step in range(max_steps):
        action = agent.act(obs, info)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        mt = info["max_tile"]
        if first_512 is None and mt >= 512:
            first_512 = step + 1
        if first_1024 is None and mt >= 1024:
            first_1024 = step + 1
        if first_2048 is None and mt >= 2048:
            first_2048 = step + 1
        if terminated or truncated:
            break

    return EpisodeResult(
        agent=getattr(agent, "name", type(agent).__name__),
        episode=0,
        seed=seed,
        score=int(info["score"]),
        max_tile=int(info["max_tile"]),
        steps=int(info["steps"]),
        won=bool(info["won"]),
        first_512=first_512,
        first_1024=first_1024,
        first_2048=first_2048,
        actions=",".join(map(str, actions)) if log_actions else "",
    )


def run_monte_carlo(
    agent_name: str,
    n_episodes: int = 100,
    seed: int = 0,
    out_dir: str | Path | None = None,
    agent_kwargs: Optional[dict[str, Any]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    agent_kwargs = agent_kwargs or {}
    agent = get_agent(agent_name, seed=seed, **agent_kwargs)
    results: list[EpisodeResult] = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_episodes)

    iterator = range(n_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc=f"MC:{agent_name}")

    for i in iterator:
        ep_seed = int(seeds[i])
        # Fresh agent RNG per episode for stochastic agents, but same type
        if hasattr(agent, "_rng"):
            agent._rng = np.random.default_rng(ep_seed)
        result = run_episode(agent, seed=ep_seed)
        result.episode = i
        results.append(result)

    df = pd.DataFrame([asdict(r) for r in results])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{agent_name}_episodes.csv", index=False)
        summary = summarize_results(df)
        (out_dir / f"{agent_name}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return df


def summarize_results(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "agent": str(df["agent"].iloc[0]) if len(df) else "unknown",
        "n_episodes": int(len(df)),
        "mean_score": float(df["score"].mean()),
        "median_score": float(df["score"].median()),
        "std_score": float(df["score"].std(ddof=0)),
        "mean_max_tile": float(df["max_tile"].mean()),
        "max_max_tile": int(df["max_tile"].max()),
        "win_rate": float(df["won"].mean()),
        "mean_steps": float(df["steps"].mean()),
        "tile_hist": {str(k): int(v) for k, v in df["max_tile"].value_counts().items()},
    }


def bakeoff(
    agents: list[str],
    n_episodes: int = 50,
    seed: int = 42,
    out_dir: str | Path = "experiments/runs/bakeoff",
    agent_kwargs_map: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_kwargs_map = agent_kwargs_map or {}
    frames = []
    summaries = {}
    for name in agents:
        kwargs = agent_kwargs_map.get(name, {})
        # Expectimax default shallow for MC speed
        if name == "expectimax" and "depth" not in kwargs:
            kwargs = {**kwargs, "depth": 1}
        df = run_monte_carlo(
            name,
            n_episodes=n_episodes,
            seed=seed,
            out_dir=out_dir,
            agent_kwargs=kwargs,
        )
        frames.append(df)
        summaries[name] = summarize_results(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "bakeoff_all.csv", index=False)
    (out_dir / "bakeoff_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return combined
