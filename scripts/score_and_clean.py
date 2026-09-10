"""打分 + 清洗入口：给 episode 打质量分并决定去留。

用法：
    python scripts/score_and_clean.py data.min_score=0.3
"""

import hydra
from omegaconf import DictConfig

from robo_int.data.cleaning import clean, score_all
from robo_int.data.episodes import EpisodeStore
from robo_int.data.scoring import default_scorer
from robo_int.infra.factory import build_task


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    store = EpisodeStore(cfg.data.data_dir)
    # 打分器的动作量纲来自任务元信息
    task = build_task(cfg)
    action_limit = task.get_info().get("action_limit", cfg.task.action_limit)
    scorer = default_scorer(action_limit=action_limit)

    metas = score_all(store, scorer)
    metas = clean(store, min_score=cfg.data.min_score,
                  max_steps_percentile=cfg.data.max_steps_percentile)

    total = len(metas)
    kept = [m for m in metas if m.kept]
    dropped = [m for m in metas if not m.kept]
    print(f"清洗完成：共 {total} 条，保留 {len(kept)} 条，丢弃 {len(dropped)} 条")
    for m in dropped:
        print(f"  丢弃 {m.episode_id}: {m.drop_reason}")


if __name__ == "__main__":
    main()
