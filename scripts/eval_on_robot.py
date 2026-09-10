"""评测入口：加载权重，在任务（真机或 mock）上跑 N 个 episode 出成功率。

用法：
    python scripts/eval_on_robot.py eval.ckpt=outputs/run/checkpoints/final.ckpt eval.num_episodes=20
"""

import hydra
from omegaconf import DictConfig

from robo_int.infra.factory import build_model, build_task, model_policy_fn
from robo_int.infra.rollout import evaluate


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    task = build_task(cfg)
    model = build_model(cfg)
    model.load_checkpoint(cfg.eval.ckpt)

    from robo_int.infra.config import RolloutConfig

    rollout_cfg = RolloutConfig(num_episodes=cfg.eval.num_episodes, seed=cfg.eval.seed)
    metrics = evaluate(task, model_policy_fn(model), rollout_cfg)
    print(f"评测结果（{cfg.eval.ckpt}）：{metrics}")


if __name__ == "__main__":
    main()
