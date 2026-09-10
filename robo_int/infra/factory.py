"""组装工厂：按配置实例化 task / robot / model / algo。

显式按 name 映射（而不是 hydra _target_ 反射），原因：
- dropout 注入这类「infra 选项覆盖模型默认」的装配逻辑写在明处；
- 基模接入各带重依赖，显式分支配合惰性 import 最干净。
"""

from __future__ import annotations

from omegaconf import DictConfig

from robo_int.algos.base import BaseAlgo
from robo_int.algos.sft import SFT
from robo_int.infra.config import RolloutConfig, TrainConfig
from robo_int.models.base import BaseVLA
from robo_int.models.mock import MockVLA
from robo_int.tasks.base import BaseTask, PolicyFn
from robo_int.tasks.mock_reach import MockReachExpert, MockReachTask


def build_task(cfg: DictConfig) -> BaseTask:
    name = cfg.task.name
    if name == "mock_reach":
        return MockReachTask(
            max_episode_steps=cfg.task.max_episode_steps,
            success_radius=cfg.task.success_radius,
            action_limit=cfg.task.action_limit,
        )
    if name == "mock_pick_place":
        from robo_int.tasks.mock_pick_place import MockPickPlaceTask

        return MockPickPlaceTask(
            task_id=cfg.task.get("task_id", 0),
            max_episode_steps=cfg.task.max_episode_steps,
            action_limit=cfg.task.action_limit,
        )
    raise KeyError(f"未知任务 {name}；可选：mock_reach / mock_pick_place（真机任务随硬件接入补齐）")


def build_expert(cfg: DictConfig) -> PolicyFn:
    """数采「专家」策略：mock 任务用比例控制器；真机任务换 teleop/示教回放。"""
    name = cfg.task.name
    if name == "mock_reach":
        expert = MockReachExpert(gain=cfg.task.expert_gain, action_limit=cfg.task.action_limit)
        return expert.act
    if name == "mock_pick_place":
        from robo_int.tasks.mock_pick_place import MockPickPlaceExpert

        expert = MockPickPlaceExpert(
            gain=cfg.task.expert_gain,
            noise=cfg.task.get("expert_noise", 0.02),
            action_limit=cfg.task.action_limit,
        )
        return expert.act
    raise KeyError(f"任务 {name} 暂无内置专家（真机走 teleop，见 robots/unitree/README.md）")


def build_model(cfg: DictConfig) -> BaseVLA:
    name = cfg.model.name
    dropout = float(cfg.infra.regularization.dropout)  # infra 选项注入模型构造
    if name == "mock":
        return MockVLA(
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            chunk_size=cfg.model.chunk_size,
            hidden_dim=cfg.model.hidden_dim,
            dropout=dropout,
        )
    if name == "pi05":
        from robo_int.models.pi05 import Pi05VLA  # 触发带指引的 NotImplementedError
        return Pi05VLA()
    if name == "pi05_server":
        # 推理-only：复用 smolvla_piper_control_gui 的 policy server（见 models/pi05_server.py）
        from robo_int.models.pi05_server import Pi05ServerPolicy

        return Pi05ServerPolicy(
            host=cfg.model.get("host", "127.0.0.1"),
            port=int(cfg.model.get("port", 8011)),
            prompt=cfg.model.get("prompt", "fold the clothes neatly"),
            actions_per_inference=int(cfg.model.get("actions_per_inference", 8)),
        )
    if name == "gr00t_n17":
        from robo_int.models.gr00t_n17 import Gr00tN17VLA
        return Gr00tN17VLA()
    if name == "fast_wam":
        from robo_int.models.fast_wam import FastWAM
        return FastWAM()
    raise KeyError(f"未知基模 {name}；可选：mock / pi05 / gr00t_n17 / fast_wam")


def build_algo(cfg: DictConfig, model: BaseVLA) -> BaseAlgo:
    name = cfg.algo.name
    if name == "sft":
        train_cfg = TrainConfig(
            total_steps=cfg.infra.total_steps,
            batch_size=cfg.infra.batch_size,
            log_every=cfg.infra.log_every,
            checkpoint_every=cfg.infra.checkpoint_every,
            seed=cfg.infra.seed,
            device=cfg.infra.device,
            run_dir=cfg.infra.run_dir,
            optimizer=cfg.infra.optimizer,
            regularization=cfg.infra.regularization,
        )
        return SFT(model, train_cfg)
    if name == "rlt":
        from robo_int.algos.rlt import RLT

        return RLT(
            model,
            action_limit=cfg.task.action_limit,
            action_mode=cfg.algo.get("action_mode", "direct"),
            token_dim=cfg.algo.get("token_dim", 64),
            utd=cfg.algo.get("utd", 2),
            start_steps=cfg.algo.get("start_steps", 300),
            seed=cfg.infra.seed,
        )
    if name == "dsrl":
        from robo_int.algos.dsrl import DSRL
        return DSRL()
    if name == "sarm2":
        from robo_int.algos.sarm2 import SARM2

        # 奖励模型不消费 VLA（model 参数仅保持 factory 签名一致）
        return SARM2(
            primitives=list(cfg.algo.primitives),
            groups=list(cfg.algo.groups),
            state_dim=cfg.algo.state_dim,
            num_tasks=cfg.algo.num_tasks,
            d_model=cfg.algo.d_model,
            stage_layers=cfg.algo.stage_layers,
            value_layers=cfg.algo.value_layers,
            n_heads=cfg.algo.n_heads,
            dropout=cfg.algo.dropout,
            n_obs_steps=cfg.algo.n_obs_steps,
            frame_gap=cfg.algo.frame_gap,
            max_rewind_steps=cfg.algo.max_rewind_steps,
            num_experts=cfg.algo.num_experts,
            top_k=cfg.algo.top_k,
            gate_hidden=cfg.algo.gate_hidden,
            expert_hidden=cfg.algo.expert_hidden,
            lambda_balance=cfg.algo.lambda_balance,
            lambda_entropy=cfg.algo.lambda_entropy,
            lambda_importance=cfg.algo.lambda_importance,
            alpha_auxiliary=cfg.algo.alpha_auxiliary,
            finetune_stage=cfg.algo.finetune_stage,
            stage_loss_weight=cfg.algo.stage_loss_weight,
            lr=cfg.algo.lr,
            weight_decay=cfg.algo.weight_decay,
            total_steps=cfg.algo.total_steps,
            seed=cfg.infra.seed,
        )
    raise KeyError(f"未知算法 {name}；可选：sft / rlt / dsrl / sarm2")


def model_policy_fn(model: BaseVLA) -> PolicyFn:
    """把 BaseVLA 适配成 PolicyFn（rollout/评测用）。

    动作块取第一步执行（receding horizon，真机 VLA 的标准用法）。
    """

    def policy(observation: dict):
        chunk = model.predict_actions(observation)
        return chunk[0]

    return policy


def build_rollout_cfg(cfg: DictConfig) -> RolloutConfig:
    return RolloutConfig(
        num_episodes=cfg.data.num_episodes,
        seed=cfg.data.seed,
        save_episodes=cfg.data.save_episodes,
    )
