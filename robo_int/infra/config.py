"""配置 schema（OmegaConf structured config）。

训练正则化选项（dropout / weight_decay / grad_clip / ema 等）集中在这里，
是 infra 的一等公民——所有算法共享同一套开关，避免散落在各算法代码里。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegularizationConfig:
    """训练正则化选项（用户点名的基础设施能力）。"""

    dropout: float = 0.0          # 基模 dropout（由模型构造时注入）
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    label_smoothing: float = 0.0  # 动作回归时退化为 huber 平滑系数
    ema_decay: float | None = None  # 权重 EMA（None 关闭）


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    optimizer: str = "adamw"      # adamw | adam | sgd
    betas: tuple = (0.9, 0.999)
    scheduler: str | None = None  # None | cosine | linear_warmup_cosine
    warmup_steps: int = 0


@dataclass
class TrainConfig:
    """训练循环配置（infra/train.yaml 对应）。"""

    total_steps: int = 2000
    batch_size: int = 64
    log_every: int = 20
    checkpoint_every: int = 500
    seed: int = 0
    device: str = "cpu"           # linux 上换 cuda
    run_dir: str = "outputs/run"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


@dataclass
class RolloutConfig:
    """采集/评测配置（scripts/teleop_collect.py、eval_on_robot.py 共用）。"""

    num_episodes: int = 10
    seed: int | None = None
    control_hz: float | None = None   # None = 用任务默认
    save_episodes: bool = True        # 是否写入 EpisodeStore
    max_episode_steps: int | None = None


@dataclass
class CleanConfig:
    """打分/清洗配置（scripts/score_and_clean.py 对应）。"""

    min_score: float = 0.3
    max_steps_percentile: float = 99.0
