"""从 Episode 列表构造训练 batch（torch 张量，LeRobot 风格字段名）。

batch 字段约定见 robo_int.models.base.BaseVLA.compute_loss 的 docstring。
支持动作块：若模型 chunk_size>1，每个时间步展开成 (T-chunk+1) 个训练样本。
"""

from __future__ import annotations

import numpy as np
import torch

from robo_int.data.episodes import Episode


class EpisodeDataset:
    """打平 episode 转移，按 chunk 展开成训练样本索引。"""

    def __init__(self, episodes: list[Episode], chunk_size: int = 1):
        self.chunk_size = chunk_size
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        for ep in episodes:
            T = len(ep.actions)
            for t in range(T - chunk_size + 1):
                self.states.append(ep.states[t])
                self.actions.append(ep.actions[t : t + chunk_size].reshape(-1))
        if not self.states:
            raise ValueError("没有可用训练样本（episode 为空或长度不足）")
        self.states = np.stack(self.states).astype(np.float32)
        self.actions = np.stack(self.actions).astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def sample_batch(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self), size=batch_size)
        return {
            "observation.state": torch.as_tensor(self.states[idx]),
            "action": torch.as_tensor(self.actions[idx]),
            "task": [""] * batch_size,
        }
