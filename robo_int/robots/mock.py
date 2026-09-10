"""Mock 机器人：纯 numpy 的理想执行器，用于无硬件环境跑通全链路。

模拟一个 N 关节位置伺服本体（一阶惯性 + 观测噪声），
接口行为与真机 SDK 客户端一致，任何新组件先过它再上真机。
"""

from __future__ import annotations

import time

import numpy as np

from robo_int.robots.base import BaseRobot, RobotState


class MockRobot(BaseRobot):
    """理想执行器：q_target -> q 实际按一阶惯性追踪，读数带小噪声。"""

    def __init__(self, num_joints: int = 6, control_hz: float = 10.0, noise: float = 0.002):
        self.num_joints = num_joints
        self.control_hz = control_hz
        self.noise = noise
        self._q: np.ndarray | None = None
        self._q_target: np.ndarray | None = None
        self._connected = False

    def connect(self) -> None:
        self._q = np.zeros(self.num_joints, dtype=np.float32)
        self._q_target = self._q.copy()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MockRobot 未 connect() 就被使用")

    def get_state(self) -> RobotState:
        self._assert_connected()
        # 模拟一阶惯性追踪（时间常数 ~2 个控制周期）
        tau = 0.2
        self._q = self._q + tau * (self._q_target - self._q)
        q_measured = self._q + np.random.normal(0.0, self.noise, self.num_joints)
        return RobotState(
            joint_positions=q_measured.astype(np.float32),
            timestamp=time.time(),
        )

    def send_action(self, action: np.ndarray) -> None:
        self._assert_connected()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_joints,):
            raise ValueError(f"动作维度 {action.shape} != ({self.num_joints},)")
        self._q_target = np.clip(action, -np.pi, np.pi)

    def get_info(self) -> dict:
        return {
            "robot_type": "mock_arm",
            "num_joints": self.num_joints,
            "action_dim": self.num_joints,
            "action_spec": "joint_position_target",
            "action_limits_low": -np.pi * np.ones(self.num_joints),
            "action_limits_high": np.pi * np.ones(self.num_joints),
        }
