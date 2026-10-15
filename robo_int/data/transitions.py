"""Episode -> transition 批数据（离线 RL / 函数性验证用）。

约定：
- action 用安全链后的执行命令（Episode.actions，与真机行为一致）；
  策略原始输出保留在 policy_actions，调用方按算法需要自取。
- reward 为 NaN（延后打分）时以 reward_fill 占位并置 placeholder_reward=True，
  调用方必须显式知晓——占位 reward 的数据不可作为正式 RL 训练数据。
- done 只在末步取 meta.extra["terminated"]（gym 惯例：truncated 不置 done，
  value bootstrap 继续跨 episode）。
"""

from __future__ import annotations

import numpy as np

from robo_int.data.episodes import Episode


def episode_to_transitions(episode: Episode, reward_fill: float = 0.0) -> dict:
    """把一个 Episode 展平成 (T, …) 的 transition 批。

    返回::
        {
          "observation.state":       (T, state_dim),
          "action":                  (T, action_dim)  执行命令（安全链后）,
          "policy_action":           (T, action_dim)  策略原始输出（或 None）,
          "reward":                  (T,),
          "next_observation.state":  (T, state_dim),
          "done":                    (T,) float,
          "num_steps": int,
          "placeholder_reward": bool,   # reward 是否为占位值
          "dry_run": bool,              # 源 episode 是否 dry-run（不可作训练数据）
          "training_eligible": bool,
        }
    """
    t = episode.meta.num_steps
    if t == 0:
        raise ValueError("空 episode 没有 transition")
    states = np.asarray(episode.states, dtype=np.float32)
    actions = np.asarray(episode.actions, dtype=np.float32)
    if states.shape[0] != t + 1:
        raise ValueError(f"states 应为 T+1 帧：{states.shape[0]} != {t + 1}")

    rewards = np.asarray(
        episode.rewards if episode.rewards is not None else np.full(t, np.nan, dtype=np.float32),
        dtype=np.float32,
    )
    placeholder = bool(np.isnan(rewards).any())
    if placeholder:
        rewards = np.where(np.isnan(rewards), np.float32(reward_fill), rewards)

    done = np.zeros(t, dtype=np.float32)
    if episode.meta.extra.get("terminated"):
        done[-1] = 1.0

    return {
        "observation.state": states[:-1],
        "action": actions,
        "policy_action": (
            np.asarray(episode.policy_actions, dtype=np.float32)
            if episode.policy_actions is not None
            else None
        ),
        "reward": rewards,
        "next_observation.state": states[1:],
        "done": done,
        "num_steps": t,
        "placeholder_reward": placeholder,
        "dry_run": bool(episode.meta.extra.get("dry_run", False)),
        "training_eligible": bool(episode.meta.extra.get("training_eligible", True)),
    }
