"""EpisodeStore -> LeRobot 数据集转换入口。

用法：
    python scripts/convert_to_lerobot.py data.data_dir=data/episodes data.lerobot_out=data/lerobot
"""

import hydra
from omegaconf import DictConfig

from robo_int.data.lerobot_io import episodes_to_lerobot


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    summary = episodes_to_lerobot(
        cfg.data.data_dir,
        cfg.data.get("lerobot_out", "data/lerobot"),
        kept_only=bool(cfg.data.get("convert_kept_only", True)),
        fps=int(cfg.data.get("fps", 30)),
    )
    print(f"转换完成：{summary['episodes']} 条 episode / {summary['frames']} 帧 -> {summary['out']}")
    print(f"  features: {summary['features']}")
    print(f"  dry_run={summary['dry_run']} training_eligible={summary['training_eligible']}")


if __name__ == "__main__":
    main()
