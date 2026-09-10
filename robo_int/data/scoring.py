"""数采打分：给每条 episode 一个 [0,1] 质量分。

规则打分器可组合；SARM2 风格的学习型打分器（视觉成功判定）后续实现
BaseDataScorer 即可接入清洗管线，接口不变。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robo_int.data.episodes import Episode


class BaseDataScorer(ABC):
    """打分器接口：单条 episode -> [0,1]，1 表示质量最好。"""

    name: str = "scorer"

    @abstractmethod
    def score(self, episode: Episode) -> float:
        ...


class SuccessScorer(BaseDataScorer):
    """按任务成功 flag 打分（success 为 None 时给中性分 0.5）。"""

    name = "success"

    def score(self, episode: Episode) -> float:
        if episode.meta.success is None:
            return 0.5
        return 1.0 if episode.meta.success else 0.0


class SmoothnessScorer(BaseDataScorer):
    """动作平滑度：动作二阶差分（jerk）越小分越高。

    抖动大的示教往往意味着操作者犹豫或接近失控，对 BC 有害。
    """
    name = "smoothness"

    def __init__(self, action_limit: float = 1.0):
        self.action_limit = action_limit

    def score(self, episode: Episode) -> float:
        actions = episode.actions
        if len(actions) < 3:
            return 0.5
        jerk = np.diff(actions, n=2, axis=0)
        typical = float(np.mean(np.abs(jerk))) / max(self.action_limit, 1e-8)
        # typical ~0 表示完全平滑；0.3 以上认为抖动严重
        return float(np.clip(1.0 - typical / 0.3, 0.0, 1.0))


class MotionScorer(BaseDataScorer):
    """有效运动量：全程近乎静止的 episode 对学习无益。"""

    name = "motion"

    def score(self, episode: Episode) -> float:
        if len(episode.states) < 3:  # 单步 episode 无法评估运动量，给中性分
            return 0.5
        motion = float(np.mean(np.abs(np.diff(episode.states[:-1], axis=0))))
        # 状态平均每步位移 > 0.01 认为运动充分；< 0.001 近乎静止
        return float(np.clip((motion - 0.001) / (0.01 - 0.001), 0.0, 1.0))


class CompositeScorer(BaseDataScorer):
    """加权组合多个打分器。weights 缺省时均匀加权。"""

    name = "composite"

    def __init__(self, scorers: list[BaseDataScorer], weights: list[float] | None = None):
        self.scorers = scorers
        self.weights = weights if weights is not None else [1.0] * len(scorers)
        assert len(self.weights) == len(self.scorers)

    def score(self, episode: Episode) -> float:
        total = sum(w * s.score(episode) for w, s in zip(self.weights, self.scorers))
        return float(total / sum(self.weights))

    def breakdown(self, episode: Episode) -> dict[str, float]:
        return {s.name: s.score(episode) for s in self.scorers}


def default_scorer(action_limit: float) -> CompositeScorer:
    """默认打分组合：成功 0.5 + 平滑 0.3 + 运动量 0.2。"""
    return CompositeScorer(
        scorers=[SuccessScorer(), SmoothnessScorer(action_limit), MotionScorer()],
        weights=[0.5, 0.3, 0.2],
    )
