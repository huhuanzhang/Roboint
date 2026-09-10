"""数采清洗：依据分数与规则决定 episode 去留。

只改 index（kept/drop_reason 字段），不删除物理数据——可追溯可回滚。
"""

from __future__ import annotations

import numpy as np

from robo_int.data.episodes import Episode, EpisodeStore
from robo_int.data.scoring import BaseDataScorer


def score_all(store: EpisodeStore, scorer: BaseDataScorer, rewrite: bool = True) -> list:
    """给索引里所有 episode 打分（读取物理数据计算规则分）。"""
    metas = store.load_index()
    for meta in metas:
        if meta.score is None:
            episode = store.load(meta.episode_id)
            meta.score = scorer.score(episode)
    if rewrite:
        store.rewrite_index(metas)
    return metas


def clean(
    store: EpisodeStore,
    min_score: float = 0.3,
    max_steps_percentile: float = 99.0,
    rewrite: bool = True,
) -> list:
    """清洗规则：

    1. 总分低于 min_score 的丢弃；
    2. 步数超过 percentile 的超长 episode 丢弃（往往是卡死/发呆）；
    3. 动作含 NaN/Inf 的丢弃。
    """
    metas = store.load_index()
    if not metas:
        return metas

    step_limit = float(np.percentile([m.num_steps for m in metas], max_steps_percentile))

    for meta in metas:
        meta.kept = True
        meta.drop_reason = None
        if meta.score is None:
            episode = store.load(meta.episode_id)
            meta.score = 0.0  # 无分数者按 0 处理，交由人工复核
        if not np.isfinite(meta.score):
            meta.kept = False
            meta.drop_reason = "nan_score"
            continue
        if meta.score < min_score:
            meta.kept = False
            meta.drop_reason = f"score={meta.score:.3f}<{min_score}"
            continue
        if meta.num_steps > step_limit:
            meta.kept = False
            meta.drop_reason = f"steps={meta.num_steps}>{step_limit:.0f}"
            continue
        episode = store.load(meta.episode_id)
        if not np.all(np.isfinite(episode.actions)):
            meta.kept = False
            meta.drop_reason = "non_finite_action"

    if rewrite:
        store.rewrite_index(metas)
    return metas


def load_kept(store: EpisodeStore) -> list[Episode]:
    """读取清洗后保留的全部 episode（训练用）。"""
    return [store.load(m.episode_id) for m in store.load_index() if m.kept]
