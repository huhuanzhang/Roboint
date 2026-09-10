"""MockVLA：小型 MLP「基模」，接口与 π0.5/GR00T 完全一致。

用途：在没有 GPU/大模型权重的环境下，把 SFT / RLT / 评测等
上层流程全部跑通。挂件（arch/）与算法（algos/）也优先在它上面做单元验证。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from robo_int.models.base import BaseVLA


class MockVLA(BaseVLA):
    """obs.state -> MLP -> 动作块 (chunk_size, action_dim)，支持 dropout。"""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_size: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.feature_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, chunk_size * action_dim)
        # 输出裕量：头输出范围略大于任务安全限幅（0.15），避免 tanh 饱和导致
        # 满幅动作梯度消失——真实 VLA 同样是「头输出裕量 + 环境侧安全裁剪」
        self.action_limit = 0.2

    def _embed(self, observation: dict) -> torch.Tensor:
        state = torch.as_tensor(np.asarray(observation["state"]), dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.backbone(state)

    def forward_actions(self, observation: dict, instruction: str | None = None):
        features = self._embed(observation)
        actions = self.action_head(features).reshape(-1, self.chunk_size, self.action_dim)
        return torch.tanh(actions) * self.action_limit

    def compute_loss(self, batch: dict) -> dict:
        obs = {"state": batch["observation.state"].cpu().numpy()}
        pred = self.forward_actions(obs).reshape(batch["action"].shape[0], -1)
        # 展平约定与 BaseVLA.compute_loss 的 batch 文档一致
        target = batch["action"]
        loss = nn.functional.mse_loss(pred, target.to(pred.dtype))
        return {"loss": loss}

    def extract_features(self, observation: dict, instruction: str | None = None) -> torch.Tensor:
        return self._embed(observation)
