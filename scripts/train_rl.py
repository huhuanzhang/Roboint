"""RL 训练入口（RLT / DSRL 等在线算法统一入口）。

在线交互循环：algo.act -> task.step -> algo.observe -> algo.update（ready 后）。
只认 BaseAlgo（act/observe/update/save_state）与 BaseTask 接口，
不包含任何算法或本体的专用分支——换算法/换任务只改配置。

与 train_sft.py 的 TrainLoop（批采样式离线更新）互补：
在线算法的数据来自 rollout 本身，不走 batch_sampler。

注意：任务 reward 可为 None（延后打分，如真机叠衣服），此时按 0 处理仅保证
链路可跑；正式 RL 需要任务侧提供在线奖励（规则/奖励模型）。

用法（mock 全链路，CPU 可跑）：
    python scripts/train_rl.py algo=rlt infra.total_steps=2000
真机（dry-run 任务可安全跑通链路，动作不发送）：
    python scripts/train_rl.py algo=rlt robot=piper_dual_arm model=pi05_server task=fold_clothes ...
    # ↑ π0.5 server client 未实现 extract_features，会显式 NotImplementedError
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path

from robo_int.infra.factory import build_algo, build_model, build_task


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    model = build_model(cfg)
    algo = build_algo(cfg, model)
    task = build_task(cfg)

    total_steps = int(cfg.infra.total_steps)
    checkpoint_every = max(1, int(cfg.infra.checkpoint_every))
    ckpt_dir = Path(cfg.infra.run_dir) / "checkpoints"

    observation = task.reset(seed=cfg.infra.seed)
    updates, last_metrics = 0, {}
    print(f"在线 RL：task={type(task).__name__} algo={type(algo).__name__} steps={total_steps}")
    try:
        for step in range(1, total_steps + 1):
            action = algo.act(observation)
            result = task.step(action)
            reward = 0.0 if result.reward is None else float(result.reward)
            algo.observe(
                observation["state"], action, reward,
                result.observation["state"], result.terminated,
            )
            observation = result.observation
            if getattr(algo, "ready", True):
                last_metrics = algo.update()
                if "skipped" not in last_metrics:
                    updates += 1
            if step % checkpoint_every == 0:
                algo.save_state(ckpt_dir / f"step_{step}.pt")
            if result.terminated or result.truncated:
                observation = task.reset()
    finally:
        task.close()
        final_path = ckpt_dir / "final.pt"
        algo.save_state(final_path)

    print(f"完成：interactions={total_steps} updates={updates} checkpoints={final_path}")
    if last_metrics and "skipped" not in last_metrics:
        print(f"末次 update 指标：{last_metrics}")


if __name__ == "__main__":
    main()
