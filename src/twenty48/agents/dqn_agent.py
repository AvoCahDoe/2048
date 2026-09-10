"""Double DQN agent for 2048."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from twenty48.env.game2048 import legal_actions


class DQNNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64 * 2 * 2, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 4) -> (B, 1, 4, 4)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.view(-1, 1, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update: int = 200,
        buffer_size: int = 50_000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        train_mode: bool = False,
    ):
        self.hidden = hidden
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_mode = train_mode
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.policy = DQNNet(hidden).to(self.device)
        self.target = DQNNet(hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0
        self.last_loss: Optional[float] = None

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> int:
        legal = None
        if info and "legal_actions" in info:
            legal = info["legal_actions"]
        elif info and "board" in info:
            legal = legal_actions(info["board"])

        if self.train_mode and self._rng.random() < self.epsilon:
            if legal:
                return int(self._rng.choice(legal))
            return int(self._rng.integers(0, 4))

        with torch.no_grad():
            q = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()
        if legal:
            mask = np.full(4, -1e9)
            for a in legal:
                mask[a] = q[a]
            return int(np.argmax(mask))
        return int(np.argmax(q))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.last_loss = float(loss.item())
        return self.last_loss

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "hidden": self.hidden,
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load(self, path: str | Path, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        hidden = ckpt.get("hidden", self.hidden)
        if hidden != self.hidden:
            self.hidden = hidden
            self.policy = DQNNet(hidden).to(self.device)
            self.target = DQNNet(hidden).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.learn_steps = ckpt.get("learn_steps", 0)
        self.train_mode = False
