"""Sarm2Scorer：把训练好的 SARM2 奖励模型接到数据打分管线（BaseDataScorer 接口）。

打分语义：episode 末帧的预测进度 1 − r̂_T（r̂ 为 steps_to_go），
即「模型认为这条数据把任务推进到了多少」。可直接喂给 data.cleaning 做
阈值清洗，论文 §III-C 数采打分用法的 mock 落点。
"""

from __future__ import annotations

import numpy as np

from robo_int.algos.sarm2.sarm2 import SARM2
from robo_int.data.episodes import Episode
from robo_int.data.scoring import BaseDataScorer


class Sarm2Scorer(BaseDataScorer):
    """用 SARM2 值解码器给 episode 打 [0,1] 质量分。

    Args:
        algo: 训练好的 SARM2（至少完成 value phase）。
        default_task_id: episode 未标注 task_id 时使用的缺省任务索引。
    """

    name = "sarm2"

    def __init__(self, algo: SARM2, default_task_id: int = 0):
        self.algo = algo
        self.default_task_id = default_task_id

    def score(self, episode: Episode) -> float:
        task_id = int(episode.meta.extra.get("task_id", self.default_task_id))
        r_hat = self.algo.predict_value(episode.states[:-1], task_id=task_id)
        r_last = r_hat[-1]
        if np.isnan(r_last):  # episode 比窗口还短：按 0 处理（数据太短本身就是低质量信号）
            return 0.0
        return float(np.clip(1.0 - r_last, 0.0, 1.0))
