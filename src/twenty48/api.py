"""FastAPI for interactive 2048 simulations and experiment results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from twenty48.agents import get_agent
from twenty48.env.game2048 import (
    ACTION_NAMES,
    Game2048Env,
    encode_board,
    legal_actions,
    move_board,
)
from twenty48.sim import run_episode, summarize_results

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"

try:
    from twenty48.agents.dqn_agent import DQNAgent

    HAS_DQN = True
except Exception:  # torch missing on slim deploys
    DQNAgent = None  # type: ignore[misc, assignment]
    HAS_DQN = False

app = FastAPI(
    title="2048 Research Lab API",
    description="Interactive simulation + experiment artifacts",
    version="0.1.0",
)

_cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors if _cors != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AgentName = Literal["random", "heuristic", "expectimax", "dqn"]


class SimulateRequest(BaseModel):
    agent: AgentName = "heuristic"
    episodes: int = Field(default=5, ge=1, le=50)
    seed: int = 42
    depth: int = Field(default=1, ge=1, le=2)
    checkpoint: Optional[str] = None


class StepRequest(BaseModel):
    board: list[list[int]]
    score: int = 0
    agent: AgentName = "heuristic"
    depth: int = 1
    checkpoint: Optional[str] = None
    seed: Optional[int] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None


def _repo_runs() -> Path:
    return RUNS


def _dqn_checkpoint_ready() -> bool:
    if not HAS_DQN:
        return False
    return (_repo_runs() / "dqn_default" / "checkpoints" / "best.pt").exists() or (
        _repo_runs() / "dqn_default" / "checkpoints" / "final.pt"
    ).exists()


def _load_agent(name: str, depth: int = 1, checkpoint: str | None = None, seed: int | None = None):
    if name == "dqn":
        if not HAS_DQN:
            raise HTTPException(
                503,
                "DQN runtime not installed on this deploy (torch omitted). Use heuristic or expectimax.",
            )
        agent = DQNAgent(train_mode=False, seed=seed)
        ckpt = Path(checkpoint) if checkpoint else _repo_runs() / "dqn_default" / "checkpoints" / "best.pt"
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            alt = _repo_runs() / "dqn_default" / "checkpoints" / "final.pt"
            if alt.exists():
                ckpt = alt
            else:
                raise HTTPException(404, f"DQN checkpoint not found: {ckpt}")
        agent.load(ckpt)
        agent.epsilon = 0.0
        return agent
    kwargs: dict[str, Any] = {"seed": seed}
    if name == "expectimax":
        kwargs["depth"] = depth
    return get_agent(name, **kwargs)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "dqn": HAS_DQN and _dqn_checkpoint_ready()}


@app.get("/api/agents")
def list_agents() -> dict[str, Any]:
    return {
        "agents": [
            {"id": "random", "label": "Random", "blurb": "Uniform legal moves — baseline."},
            {
                "id": "heuristic",
                "label": "Heuristic",
                "blurb": "Snake weights, empties, smoothness, monotonicity.",
            },
            {
                "id": "expectimax",
                "label": "Expectimax",
                "blurb": "Shallow probabilistic search over tile spawns.",
            },
            {
                "id": "dqn",
                "label": "Double DQN",
                "blurb": "Learned Q-policy from checkpoint.",
                "ready": _dqn_checkpoint_ready(),
            },
        ]
    }


@app.post("/api/sim/reset")
def reset_game(body: ResetRequest) -> dict[str, Any]:
    env = Game2048Env()
    seed = body.seed if body.seed is not None else int(np.random.randint(0, 2**31 - 1))
    obs, info = env.reset(seed=seed)
    return {
        "seed": seed,
        "board": info["board"].tolist(),
        "score": info["score"],
        "max_tile": info["max_tile"],
        "legal_actions": info["legal_actions"],
        "done": False,
        "won": info["won"],
        "obs": obs.tolist(),
    }


@app.post("/api/sim/step")
def agent_step(body: StepRequest) -> dict[str, Any]:
    board = np.array(body.board, dtype=np.int32)
    if board.shape != (4, 4):
        raise HTTPException(400, "board must be 4x4")
    agent = _load_agent(body.agent, depth=body.depth, checkpoint=body.checkpoint, seed=body.seed)
    info = {
        "board": board,
        "legal_actions": legal_actions(board),
        "score": body.score,
    }
    if not info["legal_actions"]:
        return {
            "board": board.tolist(),
            "score": body.score,
            "action": None,
            "action_name": None,
            "done": True,
            "won": bool(board.max() >= 2048),
            "max_tile": int(board.max()),
            "moved": False,
        }

    obs = encode_board(board)
    action = int(agent.act(obs, info))
    new_board, gained, _, moved = move_board(board, action)
    if not moved:
        action = info["legal_actions"][0]
        new_board, gained, _, moved = move_board(board, action)

    rng = np.random.default_rng(body.seed)
    empties = list(zip(*np.where(new_board == 0)))
    if empties:
        i, j = empties[int(rng.integers(0, len(empties)))]
        new_board[i, j] = 2 if rng.random() < 0.9 else 4

    new_score = body.score + int(gained)
    from twenty48.env.game2048 import has_moves

    done = not has_moves(new_board)
    return {
        "board": new_board.tolist(),
        "score": new_score,
        "action": action,
        "action_name": ACTION_NAMES[action],
        "gained": int(gained),
        "done": done,
        "won": bool(new_board.max() >= 2048),
        "max_tile": int(new_board.max()),
        "moved": True,
        "legal_actions": legal_actions(new_board),
    }


@app.post("/api/sim/run")
def run_simulation(body: SimulateRequest) -> dict[str, Any]:
    agent = _load_agent(body.agent, depth=body.depth, checkpoint=body.checkpoint, seed=body.seed)
    rng = np.random.default_rng(body.seed)
    rows = []
    for i in range(body.episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        if hasattr(agent, "_rng"):
            agent._rng = np.random.default_rng(ep_seed)
        r = run_episode(agent, seed=ep_seed, log_actions=True)
        r.episode = i
        rows.append(
            {
                "episode": i,
                "seed": r.seed,
                "score": r.score,
                "max_tile": r.max_tile,
                "steps": r.steps,
                "won": r.won,
                "first_512": r.first_512,
                "first_1024": r.first_1024,
                "first_2048": r.first_2048,
            }
        )
    df = pd.DataFrame(rows)
    summary = summarize_results(df.assign(agent=body.agent))
    return {"agent": body.agent, "summary": summary, "episodes": rows}


@app.get("/api/results/bakeoff")
def results_bakeoff() -> dict[str, Any]:
    summary_path = _repo_runs() / "bakeoff" / "bakeoff_summary.json"
    csv_path = _repo_runs() / "bakeoff" / "bakeoff_all.csv"
    if not summary_path.exists():
        raise HTTPException(404, "Bakeoff not found. Run: uv run twenty48-sim --bakeoff")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        episodes = df[["agent", "episode", "score", "max_tile", "steps", "won"]].to_dict(
            orient="records"
        )
    return {"summary": summary, "episodes": episodes}


@app.get("/api/results/dqn")
def results_dqn(run: str = "dqn_default") -> dict[str, Any]:
    run_dir = _repo_runs() / run
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    if not metrics_path.exists():
        raise HTTPException(404, f"No metrics for run '{run}'")
    metrics = pd.read_json(metrics_path, lines=True)
    if len(metrics) > 400:
        metrics = metrics.iloc[:: max(1, len(metrics) // 400)]
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "run": run,
        "summary": summary,
        "metrics": metrics.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/results/ablations")
def results_ablations() -> dict[str, Any]:
    runs = []
    for p in sorted(_repo_runs().iterdir()) if _repo_runs().exists() else []:
        if not p.is_dir() or not p.name.startswith("dqn"):
            continue
        sp = p / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            runs.append({"run": p.name, **s})
    return {"runs": runs}


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("twenty48.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
