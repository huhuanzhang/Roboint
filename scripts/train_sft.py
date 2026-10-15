"""SFT 训练入口：EpisodeStore(清洗后) -> SFT -> 评测。

用法（默认全 mock，CPU 可跑）：
    python scripts/train_sft.py
    python scripts/train_sft.py infra.regularization.dropout=0.1 infra.total_steps=1000
"""

import hydra
from omegaconf import DictConfig

from robo_int.algos.sft import SFT
from robo_int.data.cleaning import load_kept
from robo_int.data.dataset import EpisodeDataset
from robo_int.data.episodes import EpisodeStore
from robo_int.infra.factory import (
    build_algo,
    build_model,
    build_rollout_cfg,
    build_task,
    model_policy_fn,
)
from robo_int.infra.rollout import evaluate
from robo_int.infra.train_loop import TrainLoop


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # 1. 数据
    store = EpisodeStore(cfg.data.data_dir)
    episodes = load_kept(store)
    dataset = EpisodeDataset(episodes, chunk_size=cfg.model.chunk_size)
    print(f"训练样本：{len(dataset)}（来自 {len(episodes)} 条 episode）")

    # 2. 模型 + 算法 + 训练循环（dropout 等正则选项经 factory 从 infra 注入）
    model = build_model(cfg)
    algo: SFT = build_algo(cfg, model)
    loop = TrainLoop(algo, algo.cfg, batch_sampler=dataset.sample_batch)

    # 3. 训练
    final_ckpt = loop.run()
    print(f"训练完成，最终权重：{final_ckpt}")

    # 4. 评测（训练末验证，不落盘；数采保存走 teleop_collect / eval.save_episodes=true）
    task = build_task(cfg)
    rollout_cfg = build_rollout_cfg(cfg)
    rollout_cfg.save_episodes = False
    metrics = evaluate(task, model_policy_fn(model), rollout_cfg)
    print(f"评测：{metrics}")


if __name__ == "__main__":
    main()
