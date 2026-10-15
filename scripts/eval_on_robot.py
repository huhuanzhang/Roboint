"""评测入口：加载权重，在任务（真机或 mock）上跑 N 个 episode 出成功率。

用法：
    python scripts/eval_on_robot.py eval.ckpt=outputs/run/checkpoints/final.ckpt eval.num_episodes=20
    # server 托管权重的基模（如 pi05_server）：ckpt 置空，策略自管权重
    python scripts/eval_on_robot.py robot=piper_dual_arm model=pi05_server task=fold_clothes \
        eval.ckpt=null eval.num_episodes=1
"""

import hydra
from omegaconf import DictConfig

from robo_int.data.episodes import EpisodeStore
from robo_int.infra.factory import build_model, build_task, model_policy_fn
from robo_int.infra.rollout import evaluate


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    task = build_task(cfg)
    model = build_model(cfg)
    if cfg.eval.ckpt:  # null = 权重由模型自管（server 托管 / 免 checkpoint 模型）
        model.load_checkpoint(cfg.eval.ckpt)

    from robo_int.infra.config import RolloutConfig

    rollout_cfg = RolloutConfig(
        num_episodes=cfg.eval.num_episodes,
        seed=cfg.eval.seed,
        save_episodes=bool(cfg.eval.get("save_episodes", False)),
    )
    store = EpisodeStore(cfg.data.data_dir) if rollout_cfg.save_episodes else None
    try:
        metrics = evaluate(task, model_policy_fn(model), rollout_cfg, store=store)
    finally:
        task.close()
    print(f"评测结果（{cfg.eval.ckpt}）：{metrics}")
    if store is not None:
        print(f"episode 已保存至 {store.data_dir}")


if __name__ == "__main__":
    main()
