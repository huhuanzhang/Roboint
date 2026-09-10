"""机器人硬件抽象层。

设计要点：
- Robot 只管「读状态、发动作、使能/安全」，不知道任务语义；
  任务语义（奖励、终止、重置）在 robo_int.tasks.BaseTask 里。
- 数据平面统一用 numpy（真机进程里不引入 torch）。
- 人形 + 灵巧手这类组合本体用 CompositeRobot 拼装，对外仍是一个 BaseRobot。
- 重 SDK（unitree_sdk2、强脑 SDK 等）必须惰性导入，保证无硬件环境可 import 本模块。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotState:
    """机器人本体感知状态（proprioception）。

    所有字段可选——由具体本体的能力决定填哪些，任务层按需读取。
    约定单位：位置米/弧度，速度米每秒/弧度每秒，力/力矩牛顿/牛米。
    """

    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    # 末端位姿 [x, y, z, qw, qx, qy, qz]
    ee_pose: np.ndarray | None = None
    ee_velocity: np.ndarray | None = None
    # 末端六维力/力矩 [fx, fy, fz, tx, ty, tz]
    ee_wrench: np.ndarray | None = None
    # 末端手指关节（灵巧手则为本体全部关节）
    finger_positions: np.ndarray | None = None
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """拼接成策略可用的状态向量（按非 None 字段顺序拼接）。"""
        parts = [
            x
            for x in [
                self.joint_positions,
                self.joint_velocities,
                self.ee_pose,
                self.ee_velocity,
                self.ee_wrench,
                self.finger_positions,
            ]
            if x is not None
        ]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])


class BaseRobot(ABC):
    """单台机器人（人形/机械臂/灵巧手）的统一客户端接口。

    实现方注意：
    - send_action 应做限幅/限速裁剪后再下发，绝不让策略动作绕过安全约束；
    - get_state 应返回最近一次总线读数并带时间戳；
    - control_hz 为该本体策略层的标称控制频率（操作任务 ~10Hz，locomotion 100-500Hz）。
    """

    control_hz: float = 10.0

    @abstractmethod
    def connect(self) -> None:
        """建立连接（网络/串口/SDK 初始化）。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接，安全释放。"""

    @abstractmethod
    def get_state(self) -> RobotState:
        """读取当前本体状态。"""

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """下发一步动作。语义由 get_info()['action_spec'] 描述（如 delta ee pose / 关节位置目标）。"""

    def enable(self) -> None:
        """使能并进入受控状态（默认空实现，SDK 不支持时可跳过）。"""

    def disable(self) -> None:
        """去使能（应先回到安全姿态，具体由实现决定）。"""

    def get_info(self) -> dict:
        """本体元信息：关节数、动作维度、限位、action 语义等。

        返回示例::
            {
              "robot_type": "unitree_g1",
              "num_joints": 29,
              "action_dim": 35,          # 手臂+躯干+灵巧手
              "action_spec": "delta_ee_pose_and_fingers",
              "joint_limits_low": np.ndarray, "joint_limits_high": np.ndarray,
              "action_limits_low": np.ndarray, "action_limits_high": np.ndarray,
            }
        """
        return {"robot_type": type(self).__name__}


class CompositeRobot(BaseRobot):
    """组合本体：例如 Unitree 人形（手臂）+ 强脑 Revo2（灵巧手）。

    状态向量按子本体注册顺序拼接；动作同理切段下发。
    """

    def __init__(self, robots: list[BaseRobot]):
        assert len(robots) > 0, "CompositeRobot 至少包含一个子本体"
        self.robots = robots
        self._action_slices = self._build_slices()

    def _build_slices(self) -> list[slice]:
        slices, offset = [], 0
        for r in self.robots:
            dim = int(r.get_info().get("action_dim", 0))
            slices.append(slice(offset, offset + dim))
            offset += dim
        return slices

    # --- BaseRobot 接口 ---

    def connect(self) -> None:
        for r in self.robots:
            r.connect()

    def disconnect(self) -> None:
        for r in reversed(self.robots):
            r.disconnect()

    def enable(self) -> None:
        for r in self.robots:
            r.enable()

    def disable(self) -> None:
        for r in reversed(self.robots):
            r.disable()

    def get_state(self) -> RobotState:
        """拼接子本体状态向量；各子本体的原始状态放在 extra 里按名索引。"""
        state = RobotState()
        vectors, extra = [], {}
        for i, r in enumerate(self.robots):
            s = r.get_state()
            vectors.append(s.to_vector())
            extra[f"robot_{i}_{type(r).__name__}"] = s
        state.joint_positions = (
            np.concatenate(vectors) if vectors else np.zeros(0, dtype=np.float32)
        )
        state.extra = extra
        return state

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        for r, sl in zip(self.robots, self._action_slices):
            r.send_action(action[sl])

    def get_info(self) -> dict:
        infos = [r.get_info() for r in self.robots]
        return {
            "robot_type": "composite:" + "+".join(i.get("robot_type", "?") for i in infos),
            "num_joints": sum(int(i.get("num_joints", 0)) for i in infos),
            "action_dim": sum(int(i.get("action_dim", 0)) for i in infos),
            "sub_robots": infos,
        }
