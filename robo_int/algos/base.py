"""RL/训练算法统一接口。

SFT / RLT / DSRL / GRPO / 自研算法都实现 BaseAlgo，训练循环（infra.train_loop）
只认这个接口——这就是「算法即插即用」的落点。

算法分两类：
- 离线更新型（SFT/DSRL 离线阶段）：只需要 update(batch)；
- 在线交互型（RLT/SARM2 自提升）：还需要 act(observation) 供 rollout 调用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAlgo(ABC):
    """训练算法统一接口。"""

    @abstractmethod
    def update(self, batch: dict) -> dict:
        """执行一次参数更新，返回监控指标 dict（至少含 'loss'）。"""

    def act(self, observation: dict) -> np.ndarray:
        """在线交互时的动作选择（离线算法可不实现）。"""
        raise NotImplementedError(f"{type(self).__name__} 是离线更新型算法，不支持在线 act()。")

    def observe(self, state, action, reward, next_state, done) -> None:
        """在线交互：单条 transition 入算法内部缓冲（在线算法实现）。"""
        raise NotImplementedError(f"{type(self).__name__} 是离线更新型算法，不支持在线 observe()。")

    @property
    def ready(self) -> bool:
        """是否具备 update 条件（如回放缓冲已过预热期）；默认恒真。"""
        return True

    # ------------------------------------------------------------------ 存取

    def save_state(self, path: str | Path) -> None:
        """保存算法状态（含 optimizer/replay 等训练态，区别于模型权重）。"""

    def load_state(self, path: str | Path) -> None:
        """恢复算法状态。"""
