"""RL 训练入口（RLT / DSRL 等在线算法统一入口）。

当前 RLT/DSRL 为 stub；实现后本脚本串起
「rollout（真机/mock）-> buffer -> algo.update -> 权重同步」的在线循环。
"""

import hydra
from omegaconf import DictConfig

from robo_int.infra.factory import build_algo, build_model


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    try:
        model = build_model(cfg)
        algo = build_algo(cfg, model)
    except NotImplementedError as e:
        print(f"[未实现] {e}")
        print("当前可用：algo=sft（离线 SFT，走 scripts/train_sft.py）。")
        return
    raise RuntimeError("在线 RL 循环待 RLT/DSRL 实现后在此接入（见各算法 README）")


if __name__ == "__main__":
    main()
