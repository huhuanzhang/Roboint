"""SARM2 -> BaseRewardProvider 适配：真实 episode 的逐步 value / stage / reward。

reward 语义（如实标注）：SARM2 值解码器输出 steps_to_go = 1 − progress ∈ (0,1)。
逐步 reward 取相邻帧的进度增量::

    r_t = V(s_t) − V(s_{t+1})    （steps_to_go 的下降量 = 势能差/进度增量，potential-based）

这是从值模型导出的稠密进度信号，不是论文里独立的 reward 头（SARM2 本体只预测
steps_to_go；官方 SPIRAL 同样由 value 导出奖励）。窗口不足的头部帧为 NaN，不伪造。
"""

from __future__ import annotations

import numpy as np
import torch

from robo_int.algos.common.reward import BaseRewardProvider
from robo_int.algos.sarm2.sarm2 import SARM2


class Sarm2RewardProvider(BaseRewardProvider):
    """用 SARM2（阶段估计器 + MoE 值解码器）给 episode 逐步打信号。"""

    name = "sarm2"

    def __init__(self, algo: SARM2, default_task_id: int = 0):
        self.algo = algo
        self.default_task_id = default_task_id
        # 未训练的模型输出也是「真实模型输出」，但不是可用 RL 奖励——版本里如实标注
        self.version = (
            f"updates={algo._total_updates}" if algo._total_updates > 0 else "untrained"
        )

    def compute_episode(self, states: np.ndarray, *, task_id: int = 0, info: dict | None = None) -> dict:
        states = np.asarray(states, dtype=np.float32)
        F = states.shape[0]
        g, N = self.algo.frame_gap, self.algo.n_obs_steps
        need = (N - 1) * g + 1

        value = np.full(F, np.nan, dtype=np.float32)
        stage = np.full((F, self.algo.num_primitives), np.nan, dtype=np.float32)
        if F >= need:
            ts = np.arange(need - 1, F)
            windows = np.stack(
                [states[t - (N - 1) * g : t + 1 : g] for t in ts]
            )  # (M, N, S)
            value[ts] = self.algo.predict_value(states, task_id=task_id)[ts]
            with torch.no_grad():
                self.algo.stage_estimator.eval()
                logits, _ = self.algo.stage_estimator(torch.as_tensor(windows))
                stage[ts] = torch.softmax(logits, dim=-1).numpy()
        reward = value[:-1] - value[1:]  # 进度增量；任一端 NaN 则该步 NaN
        return {"reward": reward.astype(np.float32), "value": value, "stage": stage}

    def stage_labels(self, stage: np.ndarray) -> np.ndarray:
        """逐帧阶段 argmax（NaN 行给 -1），便于人读。"""
        labels = np.full(stage.shape[0], -1, dtype=np.int64)
        valid = ~np.isnan(stage).all(axis=1)
        labels[valid] = np.asarray(stage[valid]).argmax(axis=1)
        return labels
