"""通用训练循环：只认 BaseAlgo + 配置，不关心具体算法。

- 批数据由 batch_sampler 提供（torch 张量 dict，字段约定见 BaseVLA.compute_loss）
- 正则化选项（grad_clip/EMA）在这里统一生效；dropout 在模型构造时注入
- 指标写 jsonl（可选 wandb），断点保存
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterator

import torch

from robo_int.algos.base import BaseAlgo
from robo_int.infra.config import TrainConfig
from robo_int.utils.logging import Logger


class EMA:
    """参数指数滑动平均（regularization.ema_decay 开启时使用）。"""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(list(model.parameters()))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


class TrainLoop:
    def __init__(
        self,
        algo: BaseAlgo,
        cfg: TrainConfig,
        batch_sampler: Callable[[int], dict] | Iterator[dict],
    ):
        self.algo = algo
        self.cfg = cfg
        self.batch_sampler = batch_sampler
        self.device = torch.device(cfg.device)
        self.logger = Logger(cfg.run_dir)
        self.ema = (
            EMA(algo.model, cfg.regularization.ema_decay)
            if cfg.regularization.ema_decay is not None
            else None
        )

    def _next_batch(self, batch_size: int) -> dict:
        if callable(self.batch_sampler):
            return self.batch_sampler(batch_size)
        return next(self.batch_sampler)

    def run(self) -> Path:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        ckpt_dir = Path(cfg.run_dir) / "checkpoints"
        final_path = ckpt_dir / "final.ckpt"
        running: dict[str, float] = {}

        for step in range(1, cfg.total_steps + 1):
            batch = self._next_batch(cfg.batch_size)
            metrics = self.algo.update(batch)

            if self.ema is not None:
                self.ema.update(self.algo.model)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v)

            if step % cfg.log_every == 0:
                avg = {f"{k}_avg": v / cfg.log_every for k, v in running.items()}
                self.logger.log(step, avg)
                running = {}

            if step % cfg.checkpoint_every == 0:
                p = ckpt_dir / f"step_{step}.ckpt"
                self.algo.model.save_checkpoint(p)

        self.algo.model.save_checkpoint(final_path)
        self.logger.log(cfg.total_steps, {"done": 1.0})
        self.logger.finish()
        return final_path
