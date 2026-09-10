"""Generate static PNG plots for the portfolio README (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    bakeoff = ROOT / "experiments/runs/bakeoff/bakeoff_all.csv"
    if bakeoff.exists():
        df = pd.read_csv(bakeoff)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        agents = sorted(df["agent"].unique())
        data = [df.loc[df["agent"] == a, "score"] for a in agents]
        axes[0].boxplot(data, tick_labels=agents)
        axes[0].set_title("Score by agent")
        axes[0].set_ylabel("Score")
        for a in agents:
            tiles = sorted(df.loc[df["agent"] == a, "max_tile"])
            cdf = [i / len(tiles) for i in range(1, len(tiles) + 1)]
            axes[1].plot(tiles, cdf, marker="o", label=a)
        axes[1].set_title("Max-tile CDF")
        axes[1].set_xlabel("Max tile")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(OUT / "bakeoff.png", dpi=140)
        plt.close(fig)
        print("Wrote", OUT / "bakeoff.png")

    metrics = ROOT / "experiments/runs/dqn_default/metrics.jsonl"
    if metrics.exists():
        m = pd.read_json(metrics, lines=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        ep = m["episode"]
        axes[0, 0].plot(ep, m["score"], alpha=0.3)
        axes[0, 0].plot(ep, m["score"].rolling(30, min_periods=1).mean())
        axes[0, 0].set_title("Episode score")
        axes[0, 1].plot(ep, m["max_tile"].rolling(30, min_periods=1).mean())
        axes[0, 1].set_title("Max tile (MA30)")
        axes[1, 0].plot(ep, m["epsilon"])
        axes[1, 0].set_title("Epsilon")
        if m["loss"].notna().any():
            axes[1, 1].plot(ep, m["loss"])
            axes[1, 1].set_title("TD loss")
        fig.suptitle("DQN default learning curves")
        fig.tight_layout()
        fig.savefig(OUT / "dqn_learning.png", dpi=140)
        plt.close(fig)
        print("Wrote", OUT / "dqn_learning.png")

    # Ablation bar chart
    summaries = []
    for name in ("dqn_default", "dqn_lr_low", "dqn_tile_bonus", "dqn_wide"):
        p = ROOT / "experiments/runs" / name / "summary.json"
        if p.exists():
            import json

            s = json.loads(p.read_text(encoding="utf-8"))
            summaries.append((name, s["mean_score"], s["mean_max_tile"]))
    if summaries:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [s[0] for s in summaries]
        scores = [s[1] for s in summaries]
        ax.bar(names, scores, color="#7c3aed")
        ax.set_ylabel("Mean eval score")
        ax.set_title("DQN ablations")
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(OUT / "dqn_ablations.png", dpi=140)
        plt.close(fig)
        print("Wrote", OUT / "dqn_ablations.png")


if __name__ == "__main__":
    main()
