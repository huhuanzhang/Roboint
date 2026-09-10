"""数采入口：专家/teleop 策略在任务上采集 episode。

mock 任务用内置比例控制器专家；真机任务接入后换 teleop 源
（见 robots/unitree/README.md 第 5 步）。

用法：
    python scripts/teleop_collect.py data.num_episodes=20
"""

import hydra
from omegaconf import DictConfig

from robo_int.data.episodes import EpisodeStore
from robo_int.infra.factory import build_expert, build_rollout_cfg, build_task
from robo_int.infra.rollout import run_and_save


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    task = build_task(cfg)
    policy = build_expert(cfg)
    store = EpisodeStore(cfg.data.data_dir)
    rollout_cfg = build_rollout_cfg(cfg)

    episodes = run_and_save(
        task, policy, store, rollout_cfg, source="expert"
    )
    successes = sum(1 for e in episodes if e.meta.success)
    print(f"采集完成：{len(episodes)} 条 episode，成功 {successes} 条，"
          f"写入 {cfg.data.data_dir}")


if __name__ == "__main__":
    main()
