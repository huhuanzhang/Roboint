"""step reward 统一入口（reward provider 契约）。

动机：延后打分（StepResult.reward=None）的闭环需要「episode/transition ->
reward_t」的统一来源，RLT / DSRL / 后续算法不得各自实现 reward 逻辑。
本契约放在 algos/common（与 replay/nets 同级），不依赖任何具体算法或本体。

契约要点：
- compute_episode 是主接口：多数奖励模型本质是时序的（需要窗口/序列上下文，
  如 SARM2 的 6 帧窗口），单步无状态接口表达不了；
- compute 单步接口按需实现（无状态奖励 / 内部自维护窗口的 provider）；
- provenance() 必须可追溯 reward 来源（模型/版本/是否占位/是否训练过），
  下游写入 Episode.meta.extra 或 transition 元数据——不留只有浮点无法溯源的 reward。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRewardProvider(ABC):
    """episode -> 逐步 reward 信号的统一抽象。"""

    name: str = "reward_provider"
    version: str | None = None
    reward_is_placeholder: bool = False  # True = functional test / 占位，禁止进训练

    @abstractmethod
    def compute_episode(self, states: np.ndarray, *, task_id: int = 0, info: dict | None = None) -> dict:
        """整条状态序列 -> 逐步信号。

        Args:
            states: (F, S) 逐帧状态（真机常为 [冻结视觉嵌入; 本体状态] 拼接）。
        返回（至少含 reward，其余按 provider 能力可选，缺失键不出现）::
            {
              "reward": (F-1,) float32,   # transition 对齐：r_t 作用于 s_t -> s_{t+1}
              "value":  (F,)   float32,   # 逐帧值/进度估计（如 steps_to_go）
              "stage":  (F, K) float32,   # 逐帧阶段/原语分布（无阶段头则不返回）
              ... 无法计算的头部位置为 NaN（窗口不足），不伪造
            }
        """

    def compute(self, observation, action, next_observation, info: dict | None = None) -> float:
        """单步接口（无状态奖励实现；序列型模型请用 compute_episode）。"""
        raise NotImplementedError(
            f"{type(self).__name__} 是序列型奖励模型，单步 compute 不可用，请走 compute_episode"
        )

    def provenance(self) -> dict:
        """reward 来源元数据（写入 episode/transition，保证可追溯）。"""
        return {
            "reward_source": self.name,
            "reward_model": type(self).__name__,
            "reward_model_version": self.version,
            "reward_is_placeholder": self.reward_is_placeholder,
        }
