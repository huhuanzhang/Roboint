"""Rollout runner：策略/专家在任务上跑 episode，产出 Episode 数据。

统一服务于：
- 数采（专家/teleop 策略 -> EpisodeStore）
- 评测（训练后的 VLA -> 成功率统计）
- 后续 RL 在线交互（RLT 的 act() 也是 PolicyFn）

动作块支持：若 policy 返回 (chunk, action_dim)，逐步执行并记录。
记录约定：actions = 安全链后实际执行命令（StepResult.info["command"]，未提供时
等于 action），policy_actions = 策略原始输出——safety/RTC/smoothing 会让两者不同。
"""

from __future__ import annotations

import time

import numpy as np

from robo_int.data.episodes import Episode, EpisodeMeta, EpisodeStore, new_episode_id
from robo_int.infra.config import RolloutConfig
from robo_int.tasks.base import BaseTask, PolicyFn


def run_episode(task: BaseTask, policy: PolicyFn, seed: int | None = None) -> Episode:
    """跑一个 episode 并记录全部转移。

    policy 可返回单步动作 (D,) 或动作块 (T, D)：块走 BaseTask.execute_chunk
    逐步执行（VLA 的标准用法），单步行为与历史版本完全一致。
    步数预算按实际执行步数计（块可能被 max_episode_steps 截断或提前终止）。
    """
    observation = task.reset(seed=seed)

    states = [observation["state"]]
    images = {cam: [frame] for cam, frame in observation.get("images", {}).items()}
    timestamps = [float(observation.get("timestamp", time.time()))]
    actions, policy_actions, rewards = [], [], []
    terminated = truncated = False

    steps = 0
    while steps < task.max_episode_steps:
        output = np.asarray(policy(observation), dtype=np.float32)
        chunk = output if output.ndim == 2 else output.reshape(1, -1)
        chunk = chunk[: task.max_episode_steps - steps]  # 不超出步数预算
        result = task.execute_chunk(chunk)
        observation = result.observation  # 策略下一拍看到新观测

        executed = result.info.get("executed_actions", [chunk[0]])
        commands = result.info.get("executed_commands", executed)
        intermediates = result.info.get("intermediate_observations", [result.observation])
        for policy_action, command, obs_i in zip(executed, commands, intermediates):
            states.append(obs_i["state"])
            timestamps.append(float(obs_i.get("timestamp", time.time())))
            for cam, frame in obs_i.get("images", {}).items():
                images.setdefault(cam, []).append(frame)
            policy_actions.append(np.asarray(policy_action, dtype=np.float32))
            actions.append(np.asarray(command, dtype=np.float32))
        steps += len(executed)
        rewards.extend(
            reward if reward is not None else np.nan
            for reward in result.info.get("rewards", [result.reward])
        )
        terminated, truncated = result.terminated, result.truncated

        if result.terminated or result.truncated:
            break

    task_info = task.get_info()
    success = task.success()
    return Episode(
        meta=EpisodeMeta(
            episode_id="pending",
            task=type(task).__name__,
            robot=str(task_info.get("robot_type", "unknown")),
            source="policy",
            success=success,
            reward_sum=float(np.nansum(rewards)) if rewards else None,
            instruction=str(observation.get("instruction", "") or "") or None,
            extra={
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                # dry-run 数据不可作为训练数据（真机命令未发送）
                "dry_run": bool(task_info.get("dry_run", False)),
                "training_eligible": not bool(task_info.get("dry_run", False)),
            },
        ),
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        images={cam: np.stack(frames) for cam, frames in images.items()},
        policy_actions=np.stack(policy_actions).astype(np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def run_and_save(
    task: BaseTask,
    policy: PolicyFn,
    store: EpisodeStore,
    cfg: RolloutConfig,
    source: str = "expert",
    policy_version: str | None = None,
) -> list[Episode]:
    """跑 cfg.num_episodes 个 episode 并写入 EpisodeStore。"""
    episodes = []
    existing = store.load_index()
    for i in range(cfg.num_episodes):
        seed = None if cfg.seed is None else cfg.seed + i
        episode = run_episode(task, policy, seed=seed)
        episode.meta.episode_id = new_episode_id(existing)
        episode.meta.source = source
        episode.meta.policy_version = policy_version
        if cfg.save_episodes:
            store.save(episode)
            existing.append(episode.meta)
        episodes.append(episode)
    return episodes


def evaluate(
    task: BaseTask,
    policy: PolicyFn,
    cfg: RolloutConfig,
    store: EpisodeStore | None = None,
) -> dict:
    """评测：成功率 + 平均回报 + 平均步数。

    cfg.save_episodes=True 且给出 store 时顺带保存 episode（评测即数采）。
    """
    successes, returns, lengths = [], [], []
    existing = store.load_index() if (store is not None and cfg.save_episodes) else []
    for i in range(cfg.num_episodes):
        seed = None if cfg.seed is None else cfg.seed + i
        episode = run_episode(task, policy, seed=seed)
        successes.append(bool(episode.meta.success))
        returns.append(episode.meta.reward_sum or 0.0)
        lengths.append(episode.meta.num_steps)
        if cfg.save_episodes:
            if store is None:
                print("Warning: save_episodes=True 但未传入 EpisodeStore，跳过保存", flush=True)
            else:
                episode.meta.episode_id = new_episode_id(existing)
                store.save(episode)
                existing.append(episode.meta)
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
    }
