"""Streamlit dashboard for 2048 research experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from twenty48.viz import (
    load_metrics_jsonl,
    plot_ablation_grid,
    plot_learning_curves,
    plot_max_tile_cdf,
    plot_move_direction_hist,
    plot_score_distributions,
    plot_tile_emergence,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"

st.set_page_config(page_title="2048 Research Lab", layout="wide")
st.title("2048 Research Lab")
st.caption("Expectimax · Monte Carlo · Double DQN — interactive experiment viewer")


def discover_runs() -> list[Path]:
    if not RUNS.exists():
        return []
    return sorted([p for p in RUNS.iterdir() if p.is_dir()])


tabs = st.tabs(["Bakeoff", "DQN Training", "Ablations", "About"])

with tabs[0]:
    st.subheader("Agent bakeoff")
    bakeoff_csv = RUNS / "bakeoff" / "bakeoff_all.csv"
    alt = st.text_input("Bakeoff CSV path", value=str(bakeoff_csv))
    path = Path(alt)
    if path.exists():
        df = pd.read_csv(path)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_score_distributions(df), use_container_width=True)
            st.plotly_chart(plot_max_tile_cdf(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_tile_emergence(df), use_container_width=True)
            agents = sorted(df["agent"].unique())
            agent = st.selectbox("Direction hist agent", agents)
            st.plotly_chart(plot_move_direction_hist(df, agent), use_container_width=True)
        st.dataframe(
            df.groupby("agent")[["score", "max_tile", "steps", "won"]]
            .agg(["mean", "median", "max"])
            .round(2)
        )
    else:
        st.info("Run a bakeoff first: `uv run twenty48-sim --bakeoff --episodes 50`")

with tabs[1]:
    st.subheader("DQN learning curves")
    runs = discover_runs()
    dqn_runs = [r for r in runs if (r / "metrics.jsonl").exists()]
    if not dqn_runs:
        st.info("Train first: `uv run twenty48-train --config experiments/configs/dqn_default.yaml`")
    else:
        labels = {r.name: r for r in dqn_runs}
        choice = st.selectbox("Run", list(labels.keys()))
        metrics = load_metrics_jsonl(labels[choice] / "metrics.jsonl")
        window = st.slider("Moving average window", 10, 200, 50)
        st.plotly_chart(plot_learning_curves(metrics, window), use_container_width=True)
        summary_path = labels[choice] / "summary.json"
        if summary_path.exists():
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))

with tabs[2]:
    st.subheader("Ablation comparison")
    summaries = {}
    for r in discover_runs():
        sp = r / "summary.json"
        if sp.exists():
            summaries[r.name] = json.loads(sp.read_text(encoding="utf-8"))
    if len(summaries) < 2:
        st.info("Need ≥2 completed training runs with summary.json")
    else:
        st.plotly_chart(plot_ablation_grid(summaries), use_container_width=True)
        st.dataframe(pd.DataFrame([{"run": k, **v} for k, v in summaries.items()]))

with tabs[3]:
    st.markdown(
        """
        ### Methods
        - **Heuristic**: empty cells, snake weights, smoothness, monotonicity
        - **Expectimax**: shallow probabilistic search over tile spawns
        - **Double DQN**: CNN Q-network, replay buffer, target net, ε-greedy

        ### Reproduce
        ```bash
        uv sync --extra dev
        uv run twenty48-sim --bakeoff --episodes 50
        uv run twenty48-train --config experiments/configs/dqn_default.yaml
        uv run streamlit run scripts/dashboard.py
        ```
        """
    )
