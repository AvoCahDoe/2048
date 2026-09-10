"""CLI entry points for simulation, training, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twenty48.agents.dqn_agent import DQNAgent
from twenty48.sim import bakeoff, run_monte_carlo, summarize_results
from twenty48.train import load_config, train_dqn


def sim_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Monte Carlo 2048 simulations")
    p.add_argument("--agent", default="heuristic", help="random|heuristic|expectimax|dqn")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=1, help="expectimax depth")
    p.add_argument("--out", default="experiments/runs/sim")
    p.add_argument("--bakeoff", action="store_true", help="compare multiple agents")
    p.add_argument(
        "--agents",
        default="random,heuristic,expectimax",
        help="comma-separated for bakeoff",
    )
    p.add_argument("--checkpoint", default=None, help="DQN checkpoint path")
    args = p.parse_args(argv)

    if args.bakeoff:
        names = [a.strip() for a in args.agents.split(",") if a.strip()]
        kwargs_map = {"expectimax": {"depth": args.depth}}
        df = bakeoff(names, n_episodes=args.episodes, seed=args.seed, out_dir=args.out, agent_kwargs_map=kwargs_map)
        print(df.groupby("agent")[["score", "max_tile", "won"]].agg(["mean", "max"]))
        return

    kwargs: dict = {}
    if args.agent == "expectimax":
        kwargs["depth"] = args.depth
    if args.agent == "dqn" and args.checkpoint:
        agent = DQNAgent(train_mode=False)
        agent.load(args.checkpoint)
        # Use run via custom path: save temp by wrapping get_agent isn't enough
        from twenty48.sim import run_episode
        import numpy as np
        import pandas as pd
        from dataclasses import asdict

        rng = np.random.default_rng(args.seed)
        rows = []
        for i in range(args.episodes):
            r = run_episode(agent, seed=int(rng.integers(0, 2**31 - 1)))
            r.episode = i
            rows.append(asdict(r))
        df = pd.DataFrame(rows)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "dqn_episodes.csv", index=False)
        summary = summarize_results(df)
        (out / "dqn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    df = run_monte_carlo(args.agent, n_episodes=args.episodes, seed=args.seed, out_dir=args.out, agent_kwargs=kwargs)
    print(json.dumps(summarize_results(df), indent=2))


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Double DQN on 2048")
    p.add_argument("--config", default="experiments/configs/dqn_default.yaml")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = train_dqn(cfg, run_dir=args.out)
    print(f"Training complete: {run_dir}")


def eval_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a DQN checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/runs/eval")
    args = p.parse_args(argv)
    sim_main(
        [
            "--agent",
            "dqn",
            "--checkpoint",
            args.checkpoint,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--out",
            args.out,
        ]
    )


if __name__ == "__main__":
    sim_main()
