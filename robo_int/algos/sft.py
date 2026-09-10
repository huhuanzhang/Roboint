"""SFT / 行为克隆算法（离线更新型，完整实现）。

正则化选项（weight_decay / grad_clip / EMA）由 TrainLoop 与 OptimizerConfig
注入，这里只负责「算 loss -> 反传」。
"""

from __future__ import annotations

import torch

from robo_int.algos.base import BaseAlgo
from robo_int.infra.config import OptimizerConfig, TrainConfig
from robo_int.models.base import BaseVLA


def build_optimizer(model: torch.nn.Module, cfg: OptimizerConfig, weight_decay: float = 0.0):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.betas[0])
    raise ValueError(f"未知优化器 {cfg.optimizer}")


class SFT(BaseAlgo):
    """监督微调：model.compute_loss(batch) -> backward。"""

    def __init__(self, model: BaseVLA, train_cfg: TrainConfig):
        self.model = model
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        model.to(self.device)
        self.optimizer = build_optimizer(
            model,
            train_cfg.optimizer,
            weight_decay=train_cfg.regularization.weight_decay,
        )

    def update(self, batch: dict) -> dict:
        self.model.train()
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        losses = self.model.compute_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if self.cfg.regularization.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.regularization.grad_clip_norm
            )
        self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}
