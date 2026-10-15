"""Piper 双臂动作安全链：照抄 GUI client 的下发顺序，复用 lerobot-piper-2 的函数。

顺序（openpi_piper_dual_arm_jax_rtc_client.py 主循环）：
    scale_can0_action -> 夹爪单独 scale -> clamp_can0_action
    -> 夹爪单独 clamp -> 平滑（关节/夹爪可不同 alpha）-> deadband

参数默认值来自 GUI config.json 的 profile（max_abs_delta=0.01 rad/步、
gripper_max_abs_delta=0.003 m/步、smoothing alpha=1.0 即关闭），不自行发明。
"""

from __future__ import annotations

import numpy as np

# 双臂布局下两把夹爪在 14D 向量中的下标（每臂 6 关节 + 1 夹爪）
DUAL_ARM_GRIPPER_INDICES = (6, 13)


class PiperDualArmSafetyChain:
    """模型绝对动作 -> 安全下发动作（全部相对最新反馈状态限幅）。"""

    def __init__(
        self,
        action_scale: float = 1.0,
        max_abs_delta: float = 0.01,
        gripper_max_abs_delta: float = 0.003,
        action_smoothing_alpha: float = 1.0,
        gripper_smoothing_alpha: float = 1.0,
    ):
        self.action_scale = float(action_scale)
        # None = 显式关闭对应限幅（实验台选项；其余护栏不变）
        self.max_abs_delta = None if max_abs_delta is None else float(max_abs_delta)
        self.gripper_max_abs_delta = None if gripper_max_abs_delta is None else float(gripper_max_abs_delta)
        self.action_smoothing_alpha = float(action_smoothing_alpha)
        self.gripper_smoothing_alpha = float(gripper_smoothing_alpha)
        self._previous_action: np.ndarray | None = None

    @staticmethod
    def _import_piper_helpers():
        """scale/clamp 直接复用 lerobot-piper-2 中的同名已验证实现。"""
        try:
            from piper_scripts.utils.openpi_piper_single_arm_can0_inference import (
                clamp_can0_action,
                scale_can0_action,
            )
        except ImportError as exc:  # pragma: no cover — 依赖路径注入由 robot 层负责
            raise ImportError(
                "需要先把 lerobot-piper-2 根目录加入 sys.path（见 PiperDualArmRobot）"
            ) from exc
        return scale_can0_action, clamp_can0_action

    def _apply_gripper_scale(self, action, model_action, state):
        scale = self.action_scale  # GUI 无单独夹爪 scale 时同 action_scale
        values = np.array(action, dtype=np.float32, copy=True)
        for idx in DUAL_ARM_GRIPPER_INDICES:
            values[idx] = state[idx] + scale * (model_action[idx] - state[idx])
        return values

    def _apply_gripper_clamp(self, action, state):
        values = np.array(action, dtype=np.float32, copy=True)
        for idx in DUAL_ARM_GRIPPER_INDICES:
            values[idx] = np.clip(
                values[idx], state[idx] - self.gripper_max_abs_delta, state[idx] + self.gripper_max_abs_delta
            )
        return values

    def _smooth(self, action):
        if self._previous_action is None:
            return action
        alpha, grip_alpha = self.action_smoothing_alpha, self.gripper_smoothing_alpha
        if alpha >= 1.0 and grip_alpha >= 1.0:
            return action
        values = alpha * np.asarray(action, dtype=np.float32) + (1.0 - alpha) * self._previous_action
        for idx in DUAL_ARM_GRIPPER_INDICES:
            values[idx] = grip_alpha * action[idx] + (1.0 - grip_alpha) * self._previous_action[idx]
        return values.astype(np.float32)

    def __call__(self, model_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        model_action = np.asarray(model_action, dtype=np.float32)
        current_state = np.asarray(current_state, dtype=np.float32)
        if model_action.shape != (14,) or current_state.shape != (14,):
            raise ValueError(f"期望 14D 动作/状态，收到 {model_action.shape} / {current_state.shape}")
        if not np.isfinite(model_action).all() or not np.isfinite(current_state).all():
            raise ValueError("动作或状态含 NaN/Inf")

        scale_can0_action, clamp_can0_action = self._import_piper_helpers()
        scaled = scale_can0_action(model_action, current_state, self.action_scale)
        scaled = self._apply_gripper_scale(scaled, model_action, current_state)
        if self.max_abs_delta is None:
            clamped = scaled
        else:
            clamped = clamp_can0_action(scaled, current_state, self.max_abs_delta)
        if self.gripper_max_abs_delta is not None:
            clamped = self._apply_gripper_clamp(clamped, current_state)
        final = self._smooth(clamped)
        final = np.asarray(final, dtype=np.float32)
        self._previous_action = final.copy()
        return final

    def reset(self) -> None:
        self._previous_action = None
