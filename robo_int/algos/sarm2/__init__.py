"""SARM2：stage-aware 奖励模型（数采打分 + RL 奖励双用途）。

实现说明与论文/官方代码差异表见 README.md。
"""

from robo_int.algos.sarm2.sarm2 import (
    SARM2,
    MMoEValueDecoder,
    StageEstimator,
    build_hierarchy,
    build_stage_windows,
    build_value_windows,
    collapse_score,
)
from robo_int.algos.sarm2.scorer import Sarm2Scorer

__all__ = [
    "SARM2",
    "StageEstimator",
    "MMoEValueDecoder",
    "Sarm2Scorer",
    "build_hierarchy",
    "build_stage_windows",
    "build_value_windows",
    "collapse_score",
]
