"""DQN training loop with metrics and checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from tqdm import tqdm

from twenty48.agents.dqn_agent import DQNAgent
from twenty48.env.game2048 import Game2048Env
from twenty48.sim import run_episode, summarize_results
import pandas as pd


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_dqn(config: dict[str, Any], run_dir: str | Path | None = None) -> Path:
    run_name = config.get("name", f"dqn_{int(time.time())}")
    root = Path(run_dir or f"experiments/runs/{run_name}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    seed = int(config.get("seed", 0))
    episodes = int(config.get("episodes", 2000))
    max_steps = int(config.get("max_steps", 5000))
    eval_every = int(config.get("eval_every", 200))
    eval_episodes = int(config.get("eval_episodes", 20))
    checkpoint_every = int(config.get("checkpoint_every", 500))

    env = Game2048Env(max_tile_bonus=float(config.get("max_tile_bonus", 0.0)))
    agent = DQNAgent(
        hidden=int(config.get("hidden", 128)),
        lr=float(config.get("lr", 1e-3)),
        gamma=float(config.get("gamma", 0.99)),
        epsilon_start=float(config.get("epsilon_start", 1.0)),
        epsilon_end=float(config.get("epsilon_end", 0.05)),
        epsilon_decay=float(config.get("epsilon_decay", 0.995)),
        batch_size=int(config.get("batch_size", 64)),
        target_update=int(config.get("target_update", 200)),
        buffer_size=int(config.get("buffer_size", 50_000)),
        seed=seed,
        train_mode=True,
    )

    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = np.random.default_rng(seed)
    best_eval = -1.0

    for ep in tqdm(range(episodes), desc=f"train:{run_name}"):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=ep_seed)
        agent.reset()
        ep_reward = 0.0
        losses: list[float] = []

        for t in range(max_steps):
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Normalize reward for stability
            scaled = float(reward) / 10.0
            agent.remember(obs, action, scaled, next_obs, done)
            if t % 4 == 0 or done:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            ep_reward += float(reward)
            obs = next_obs
            if done:
                break

        agent.decay_epsilon()
        row = {
            "episode": int(ep),
            "score": int(info["score"]),
            "max_tile": int(info["max_tile"]),
            "steps": int(info["steps"]),
            "ep_reward": float(ep_reward),
            "epsilon": float(agent.epsilon),
            "loss": float(np.mean(losses)) if losses else None,
            "won": bool(info["won"]),
            "buffer": int(len(agent.buffer)),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if (ep + 1) % checkpoint_every == 0:
            agent.save(root / "checkpoints" / f"ep{ep + 1}.pt")

        if (ep + 1) % eval_every == 0:
            eval_df = _eval_agent(agent, eval_episodes, seed + ep)
            mean_score = float(eval_df["score"].mean())
            summary = summarize_results(eval_df)
            summary["episode"] = ep + 1
            with (root / "eval.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
            if mean_score > best_eval:
                best_eval = mean_score
                agent.save(root / "checkpoints" / "best.pt")

    agent.save(root / "checkpoints" / "final.pt")
    # Final offline eval
    agent.train_mode = False
    agent.epsilon = 0.0
    final_df = _eval_agent(agent, max(eval_episodes, 30), seed + 999)
    final_summary = summarize_results(final_df)
    (root / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    final_df.to_csv(root / "final_eval.csv", index=False)
    return root


def _eval_agent(agent: DQNAgent, n: int, seed: int) -> pd.DataFrame:
    was_train = agent.train_mode
    eps = agent.epsilon
    agent.train_mode = False
    agent.epsilon = 0.0
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        r = run_episode(agent, seed=int(rng.integers(0, 2**31 - 1)), log_actions=False)
        r.episode = i
        rows.append(r.__dict__)
    agent.train_mode = was_train
    agent.epsilon = eps
    return pd.DataFrame(rows)


def run_ablations(configs_dir: str | Path, out_root: str | Path) -> list[Path]:
    configs_dir = Path(configs_dir)
    out_root = Path(out_root)
    paths = []
    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        cfg = load_config(cfg_path)
        run_dir = out_root / cfg.get("name", cfg_path.stem)
        paths.append(train_dqn(cfg, run_dir=run_dir))
    return paths
