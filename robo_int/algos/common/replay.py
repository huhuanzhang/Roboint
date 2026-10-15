"""极简 numpy 回放缓冲（在线 RL 用）。

真机数据量级（万级 transition）下 numpy 环形缓冲足够；
后续要跨进程/持久化时替换为 data/ 的 EpisodeStore 并保持接口。
"""

from __future__ import annotations

import numpy as np


class SimpleReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        seed: int = 0,
        feature_dim: int | None = None,
    ):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        # residual 模式的 VLA 原动作（critic 上下文）；direct 模式恒零占位，
        # 回放 schema 统一（RLT.observe 负责）
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        # 可选：冻结 VLA 特征（observe 时一次前向算好缓存，update 不再重跑 VLA）；
        # 不提供特征时列为 None，sample() 不含对应键（mock 路径行为不变）
        self.features = (
            np.zeros((capacity, feature_dim), dtype=np.float32) if feature_dim is not None else None
        )
        self.next_features = (
            np.zeros((capacity, feature_dim), dtype=np.float32) if feature_dim is not None else None
        )
        self._ptr = 0
        self._size = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        base_action=None,
        feature=None,
        next_feature=None,
    ) -> None:
        i = self._ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.base_actions[i] = base_action if base_action is not None else 0.0
        if self.features is not None:
            if feature is None or next_feature is None:
                raise ValueError("回放已启用特征列，add 必须提供 feature/next_feature")
            self.features[i] = feature
            self.next_features[i] = next_feature
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self._size, size=batch_size)
        batch = {
            "observation.state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_observation.state": self.next_states[idx],
            "done": self.dones[idx],
            "base_action": self.base_actions[idx],
        }
        if self.features is not None:
            batch["observation.feature"] = self.features[idx]
            batch["next_observation.feature"] = self.next_features[idx]
        return batch
