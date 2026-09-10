"""SAC 标准网络件：tanh 高斯 actor（带输出裕量）、集成 critic。

输出裕量（action_margin > 任务限幅）：避免 tanh 饱和区梯度消失——
与 MockVLA 的教训一致，这里作为所有连续控制 actor 的默认行为。
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], activation=nn.ReLU) -> nn.Sequential:
    """线性-激活堆叠；最后一层不接激活（回归头标准做法）。"""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """高斯策略 + tanh 压缩。输出范围 (±action_margin)，由环境侧再裁剪到限幅。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, action_margin: float = 1.0):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 2 * action_dim])
        self.action_dim = action_dim
        self.action_margin = action_margin
        # log_std 范围夹紧（标准做法，防熵爆炸/塌缩）
        self.log_std_min, self.log_std_max = -5.0, 2.0

    def forward(self, obs: torch.Tensor):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = torch.distributions.Normal(mu, log_std.exp())
        # rsample + tanh：可微采样；atanh 变换取 log_prob（含修正项）。
        # 修正项必须作用在归一化 tanh(raw) 上：a = margin·tanh(u) 时
        # log|da/du| = log(margin·(1 - tanh(u)^2))——直接写 1 - a^2 会让
        # 修正项退化为近似常数（tanh 饱和惩罚消失），曾导致 SAC 发散
        raw = dist.rsample()
        pre = torch.tanh(raw)
        action = pre * self.action_margin
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(
            self.action_margin * (1 - pre.pow(2)) + 1e-6
        ).sum(-1)
        return action, log_prob, torch.tanh(mu) * self.action_margin


class EnsembleCritic(nn.Module):
    """N 个独立 Q 网络（SAC 双 Q / RLPD 集成 Q 共用）。"""

    def __init__(self, obs_dim: int, action_dim: int, num_qs: int = 2, hidden: int = 256):
        super().__init__()
        self.qs = nn.ModuleList(
            [mlp([obs_dim + action_dim, hidden, hidden, 1]) for _ in range(num_qs)]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]
