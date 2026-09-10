"""Visualization helpers for notebooks and Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def moving_average(series: pd.Series, window: int = 50) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_learning_curves(metrics: pd.DataFrame, window: int = 50) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Episode score", "Max tile", "Epsilon", "TD loss"),
    )
    ep = metrics["episode"]
    fig.add_trace(
        go.Scatter(x=ep, y=metrics["score"], name="score", opacity=0.3, line=dict(width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ep,
            y=moving_average(metrics["score"], window),
            name=f"score MA{window}",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=ep, y=moving_average(metrics["max_tile"], window), name="max_tile MA"),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=ep, y=metrics["epsilon"], name="epsilon"), row=2, col=1)
    if "loss" in metrics.columns:
        loss = metrics["loss"].dropna()
        fig.add_trace(
            go.Scatter(x=loss.index.map(lambda i: metrics.loc[i, "episode"]), y=loss, name="loss"),
            row=2,
            col=2,
        )
    fig.update_layout(height=700, title_text="DQN learning curves", template="plotly_white")
    return fig


def plot_score_distributions(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="agent",
        y="score",
        color="agent",
        box=True,
        points="outliers",
        title="Score distributions by agent",
        template="plotly_white",
    )
    return fig


def plot_max_tile_cdf(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for agent, g in df.groupby("agent"):
        tiles = np.sort(g["max_tile"].values)
        cdf = np.arange(1, len(tiles) + 1) / len(tiles)
        fig.add_trace(go.Scatter(x=tiles, y=cdf, mode="lines+markers", name=str(agent)))
    fig.update_layout(
        title="Max-tile CDF (survival of achievement)",
        xaxis_title="Max tile",
        yaxis_title="CDF",
        template="plotly_white",
    )
    return fig


def plot_move_direction_hist(df: pd.DataFrame, agent: Optional[str] = None) -> go.Figure:
    names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    subset = df if agent is None else df[df["agent"] == agent]
    counts = {k: 0 for k in names}
    for acts in subset["actions"].dropna():
        if not acts:
            continue
        for a in str(acts).split(","):
            if a == "":
                continue
            counts[int(a)] = counts.get(int(a), 0) + 1
    fig = px.bar(
        x=[names[k] for k in sorted(counts)],
        y=[counts[k] for k in sorted(counts)],
        labels={"x": "Direction", "y": "Count"},
        title=f"Move direction histogram{f' ({agent})' if agent else ''}",
        template="plotly_white",
        color=[names[k] for k in sorted(counts)],
    )
    return fig


def plot_tile_emergence(df: pd.DataFrame) -> go.Figure:
    records = []
    for _, row in df.iterrows():
        for tile, col in ((512, "first_512"), (1024, "first_1024"), (2048, "first_2048")):
            val = row.get(col)
            if pd.notna(val):
                records.append({"agent": row["agent"], "tile": tile, "step": val})
    if not records:
        fig = go.Figure()
        fig.update_layout(title="No tile emergence events logged")
        return fig
    edf = pd.DataFrame(records)
    fig = px.box(
        edf,
        x="tile",
        y="step",
        color="agent",
        title="Tile emergence timeline (first step reaching tile)",
        template="plotly_white",
    )
    return fig


def plot_ablation_grid(summaries: dict[str, dict]) -> go.Figure:
    rows = [{"run": k, **v} for k, v in summaries.items()]
    adf = pd.DataFrame(rows)
    fig = px.bar(
        adf,
        x="run",
        y="mean_score",
        color="run",
        title="Ablation: mean eval score",
        template="plotly_white",
    )
    return fig


def plot_policy_heatmap_from_q(q_grid: np.ndarray) -> go.Figure:
    """q_grid: (4,4) preferred action ids or values for creative viz."""
    fig = px.imshow(
        q_grid,
        color_continuous_scale="Viridis",
        title="Board value / policy heatmap",
        template="plotly_white",
    )
    return fig
