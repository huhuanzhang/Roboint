"""Rollout runner：策略/专家在任务上跑 episode，产出 Episode 数据。

统一服务于：
- 数采（专家/teleop 策略 -> EpisodeStore）
- 评测（训练后的 VLA -> 成功率统计）
- 后续 RL 在线交互（RLT 的 act() 也是 PolicyFn）

动作块支持：若 policy 返回 (chunk, action_dim)，逐步执行并记录。
"""

from __future__ import annotations

import numpy as np

from robo_int.data.episodes import Episode, EpisodeMeta, EpisodeStore, new_episode_id
from robo_int.infra.config import RolloutConfig
from robo_int.tasks.base import BaseTask, PolicyFn


def run_episode(task: BaseTask, policy: PolicyFn, seed: int | None = None) -> Episode:
    """跑一个 episode 并记录全部转移。"""
    observation = task.reset(seed=seed)

    states = [observation["state"]]
    images = {cam: [frame] for cam, frame in observation.get("images", {}).items()}
    actions, rewards = [], []

    for _ in range(task.max_episode_steps):
        action = np.asarray(policy(observation), dtype=np.float32).reshape(-1)
        result = task.step(action)
        observation = result.observation  # 策略下一拍看到新观测

        states.append(observation["state"])
        for cam, frame in result.observation.get("images", {}).items():
            images.setdefault(cam, []).append(frame)
        actions.append(action)
        rewards.append(result.reward if result.reward is not None else np.nan)

        if result.terminated or result.truncated:
            break

    success = task.success()
    return Episode(
        meta=EpisodeMeta(
            episode_id="pending",
            task=type(task).__name__,
            robot="unknown",
            source="policy",
            success=success,
            reward_sum=float(np.nansum(rewards)) if rewards else None,
        ),
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        images={cam: np.stack(frames) for cam, frames in images.items()},
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
        episode.meta.robot = "unknown"
        episode.meta.source = source
        episode.meta.policy_version = policy_version
        if cfg.save_episodes:
            store.save(episode)
            existing.append(episode.meta)
        episodes.append(episode)
    return episodes


def evaluate(task: BaseTask, policy: PolicyFn, cfg: RolloutConfig) -> dict:
    """评测：成功率 + 平均回报 + 平均步数。不写盘。"""
    successes, returns, lengths = [], [], []
    for i in range(cfg.num_episodes):
        seed = None if cfg.seed is None else cfg.seed + i
        episode = run_episode(task, policy, seed=seed)
        successes.append(bool(episode.meta.success))
        returns.append(episode.meta.reward_sum or 0.0)
        lengths.append(episode.meta.num_steps)
    return {
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
    }
