"""Mock 任务：2D 点到点 reaching（纯 numpy，无物理引擎依赖）。

用于端到端验证「数采 -> 打分清洗 -> SFT -> 评测」全链路。
附带一个比例控制器「专家」，扮演 teleop 示教者。
"""

from __future__ import annotations

import numpy as np

from robo_int.tasks.base import BaseTask, ResetMode, StepResult


class MockReachTask(BaseTask):
    """把一个 2 轴「末端」在桌面上移动到随机目标点。

    - 状态空间: pos(2) + goal(2)，动作: delta pos(2)
    - 奖励: -距离（稠密），成功: 距离 < success_radius
    - episode 上限 max_episode_steps 步
    """

    reset_mode = ResetMode.SCRIPTED

    def __init__(
        self,
        max_episode_steps: int = 50,
        success_radius: float = 0.08,
        action_limit: float = 0.15,
        workspace: float = 1.0,
    ):
        self.max_episode_steps = max_episode_steps
        self.success_radius = success_radius
        self.action_limit = action_limit
        self.workspace = workspace
        self._pos = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------ 任务接口

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self._goal = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self._steps = 0
        return self.get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32), -self.action_limit, self.action_limit)
        self._pos = np.clip(self._pos + action, -self.workspace / 2, self.workspace / 2)
        self._steps += 1
        dist = float(np.linalg.norm(self._pos - self._goal))
        return StepResult(
            observation=self.get_observation(),
            reward=-dist,
            terminated=dist < self.success_radius,
            truncated=self._steps >= self.max_episode_steps,
            info={"distance": dist, "steps": self._steps},
        )

    def get_observation(self) -> dict:
        return {
            "state": np.concatenate([self._pos, self._goal]).astype(np.float32),
            "instruction": "reach the goal",
            "images": {},  # mock 任务无图像；真实任务在此填相机帧
        }

    def success(self) -> bool:
        return float(np.linalg.norm(self._pos - self._goal)) < self.success_radius

    def get_info(self) -> dict:
        return {
            **super().get_info(),
            "action_dim": 2,
            "action_spec": "delta_position_2d",
            "success_radius": self.success_radius,
        }


class MockReachExpert:
    """比例控制器「专家」：MockReachTask 的示教/基准策略。"""

    def __init__(self, gain: float = 0.8, action_limit: float = 0.15):
        self.gain = gain
        self.action_limit = action_limit

    def act(self, observation: dict) -> np.ndarray:
        state = observation["state"]
        pos, goal = state[:2], state[2:4]
        return np.clip(self.gain * (goal - pos), -self.action_limit, self.action_limit).astype(
            np.float32
        )
