"""Piper 双臂机器人接入（复用 lerobot-piper-2 的 PiperRobot，不重新实现）。

装配方式照抄 smolvla_piper_control_gui/client/openpi_piper_dual_arm_jax_rtc_client.py
的 build_dual_arm_robot_config()：
- 左臂 can0、右臂 can1，各 6 关节(rad) + 夹爪(m)；
- RealSense 相机按 serial 指定，640x480@30fps；
- inference_time + passive_recording 模式：connect() 只连 CAN 读数与相机，
  不使能电机（与 GUI client 的 dry-run 行为一致）。

14D state / action 顺序：左臂 7D 在前、右臂 7D 在后（robot order = left-right）。
重依赖（lerobot、pyrealsense2）惰性导入，保证无硬件环境可 import 本模块。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from robo_int.robots.base import BaseRobot, RobotState

# 与 GUI client 一致的默认路径
DEFAULT_LEROBOT_PIPER_ROOT = Path("/home/ps/Documents/lerobot-piper-2")

# GUI config.json 中真实在用的四路相机（checkpoint 只要求其中三路，见 model 配置）
DEFAULT_CAMERAS = {
    "head": 254622073267,
    "left_wrist": 244222071617,
    "right_wrist": 317622070857,
    "front_view": 254622079402,
}


class PiperDualArmRobot(BaseRobot):
    """双臂 Piper（Agibot Piper x2）：state/action 均为 14D 绝对关节目标。

    capture_observation() 额外返回相机帧（BaseRobot 接口只约定本体感知，
    VLA 推理需要视觉，故以独立方法提供，语义与 GUI client 的同名调用一致）。
    """

    def __init__(
        self,
        cameras: dict[str, int] | None = None,
        left_can_name: str = "can0",
        right_can_name: str = "can1",
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        gripper_effort: int = 1000,
        control_hz: float = 30.0,
        lerobot_piper_root: str | Path = DEFAULT_LEROBOT_PIPER_ROOT,
        cameras_only: bool = False,
    ):
        self.camera_serials = dict(cameras) if cameras else dict(DEFAULT_CAMERAS)
        self.left_can_name = left_can_name
        self.right_can_name = right_can_name
        self.fps = fps
        self.width = width
        self.height = height
        self.gripper_effort = gripper_effort
        self.control_hz = control_hz
        self.lerobot_piper_root = Path(lerobot_piper_root)
        # 诊断模式：臂控未上电时只连相机，state 用零向量（server validate_inference 同做法）
        self.cameras_only = cameras_only
        self._robot = None
        self._cameras = {}
        self._connected = False

    # ------------------------------------------------------------------ 装配

    def _ensure_lerobot_on_path(self) -> None:
        if not self.lerobot_piper_root.is_dir():
            raise FileNotFoundError(
                f"lerobot-piper-2 不存在：{self.lerobot_piper_root}"
            )
        if str(self.lerobot_piper_root) not in sys.path:
            sys.path.insert(0, str(self.lerobot_piper_root))

    def _build_config(self):
        """照抄 GUI client 的 build_dual_arm_robot_config（被动记录模式）。"""
        from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
        from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig
        from lerobot.common.robot_devices.robots.configs import PiperRobotConfig

        default_config = PiperRobotConfig()
        left_default = default_config.follower_arms["left"]
        right_default = default_config.follower_arms["right"]
        cameras = {
            key: IntelRealSenseCameraConfig(
                serial_number=int(serial),
                fps=self.fps,
                width=self.width,
                height=self.height,
                force_hardware_reset=False,
                rotation=None,
            )
            for key, serial in self.camera_serials.items()
        }
        return PiperRobotConfig(
            inference_time=True,
            passive_recording=True,
            leader_arms={},
            follower_arms={
                "left": PiperMotorsBusConfig(
                    can_name=self.left_can_name,
                    motors=dict(left_default.motors),
                    gripper_effort=self.gripper_effort,
                ),
                "right": PiperMotorsBusConfig(
                    can_name=self.right_can_name,
                    motors=dict(right_default.motors),
                    gripper_effort=self.gripper_effort,
                ),
            },
            follower_arm={},
            cameras=cameras,
        )

    # ------------------------------------------------------------------ BaseRobot 接口

    def connect(self) -> None:
        if self._connected:
            return
        self._ensure_lerobot_on_path()
        if self.cameras_only:
            from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
            from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

            self._cameras = {
                key: IntelRealSenseCamera(
                    IntelRealSenseCameraConfig(
                        serial_number=int(serial),
                        fps=self.fps,
                        width=self.width,
                        height=self.height,
                        force_hardware_reset=False,
                        rotation=None,
                    )
                )
                for key, serial in self.camera_serials.items()
            }
            for camera in self._cameras.values():
                camera.connect()
            self._connected = True
            return
        from lerobot.common.robot_devices.robots.piper import PiperRobot

        self._robot = PiperRobot(self._build_config())
        self._robot.connect()  # passive_recording 模式：不使能电机
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self.cameras_only:
            for camera in self._cameras.values():
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
            self._cameras = {}
            self._connected = False
            return
        # 与 GUI client 的 disconnect_robot_without_arm_motion 一致：只停相机，不动臂
        if getattr(self._robot, "teleop", None) is not None:
            self._robot.teleop.stop()
        for camera in self._robot.cameras.values():
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        self._robot.is_connected = False
        self._connected = False

    def enable(self) -> None:
        """使能双臂电机（live 模式专用；dry-run 不调用）。"""
        self._assert_connected()
        for name, arm in self._robot.follower_arms.items():
            print(f"Enabling {name} arm on {arm.can_name}...")
            if not arm.connect(enable=True):
                raise RuntimeError(f"Failed to enable {name} arm on {arm.can_name}.")

    def disable(self) -> None:
        """回安全姿态并去使能（沿用 GUI client 的 disable_robot_arms）。"""
        self._assert_connected()
        for name, arm in self._robot.follower_arms.items():
            print(f"Moving {name} arm on {arm.can_name} to its safe disconnect pose.")
            arm.safe_disconnect()
        time.sleep(5.0)
        for name, arm in self._robot.follower_arms.items():
            print(f"Disabling {name} arm on {arm.can_name}.")
            arm.connect(enable=False)

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PiperDualArmRobot 未 connect() 就被使用")

    def _to_numpy(self, value) -> np.ndarray:
        import torch

        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def capture_observation(self) -> dict:
        """一拍完整观测：state 14D + 各相机 RGB (H,W,3) uint8。"""
        self._assert_connected()
        if self.cameras_only:
            images = {name: np.array(cam.read(), copy=True) for name, cam in self._cameras.items()}
            return {"state": np.zeros(14, dtype=np.float32), "images": images, "timestamp": time.time()}
        obs = self._robot.capture_observation()
        state = self._to_numpy(obs["observation.state"]).astype(np.float32, copy=True)
        images = {}
        for name in self._robot.cameras:
            key = f"observation.images.{name}"
            if key not in obs:
                raise KeyError(f"PiperRobot 观测缺少相机 {key}")
            images[name] = np.array(obs[key], copy=True)
        return {"state": state, "images": images, "timestamp": time.time()}

    def get_state(self) -> RobotState:
        state = self.capture_observation()["state"]
        return RobotState(joint_positions=state, timestamp=time.time())

    def send_action(self, action: np.ndarray) -> None:
        """下发 14D 绝对关节目标（左臂在前）。live 模式才允许调用。"""
        import torch

        self._assert_connected()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError(f"Piper 双臂动作必须 14D，收到 {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("Piper 动作包含 NaN/Inf，拒绝下发")
        self._robot.send_action(torch.from_numpy(action))

    def get_info(self) -> dict:
        return {
            "robot_type": "piper_dual_arm",
            "num_joints": 14,
            "action_dim": 14,
            "action_spec": "absolute_piper14_left_then_right (6 joints rad + gripper m per arm)",
            "cameras": dict(self.camera_serials),
            "left_can": self.left_can_name,
            "right_can": self.right_can_name,
        }
