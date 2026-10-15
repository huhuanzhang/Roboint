"""真实机器人叠衣服任务：本体无关的 VLA rollout 编排。

任务不做任何 Piper / π0.5 特判——只依赖 BaseRobot 的通用接口
（capture_observation / apply_safety / send_action），语言指令来自任务配置，
推理由上层 BaseVLA（如 Pi05ServerPolicy）完成。

dry_run=True（默认）时：读取真实 state/相机、执行安全链变换并记录最终命令，
但绝不调用 send_action；成功判定延后（人工/奖励模型，见 tasks/README 路线）。
"""

from __future__ import annotations

import time

import numpy as np

from robo_int.robots.base import BaseRobot
from robo_int.tasks.base import BaseTask, ResetMode, StepResult


class FoldClothesTask(BaseTask):
    """叠衣服：人工摆放重置 + 延后成功判定的真机 VLA 任务。"""

    def __init__(
        self,
        robot: BaseRobot,
        max_episode_steps: int = 24,
        dry_run: bool = True,
        instruction: str = "fold the clothes neatly",
        camera_warmup_frames: int = 0,
    ):
        self.robot = robot
        self.max_episode_steps = max_episode_steps
        self.dry_run = dry_run
        self.instruction = instruction
        self.camera_warmup_frames = camera_warmup_frames
        self.reset_mode = ResetMode.MANUAL
        self._steps = 0
        self._state: np.ndarray | None = None
        # dry-run 审计：安全链后的最终命令 & 是否发生过真实下发
        self.command_log: list[np.ndarray] = []
        self.command_sent_steps = 0

    def reset(self, seed: int | None = None) -> dict:
        self.robot.connect()  # 幂等本体：已连接时直接返回
        if self.camera_warmup_frames:
            for _ in range(self.camera_warmup_frames):
                self.robot.capture_observation()
        self._steps = 0
        self.command_log.clear()
        self.command_sent_steps = 0
        return self.get_observation()

    def get_observation(self) -> dict:
        obs = self.robot.capture_observation()
        self._state = np.asarray(obs["state"], dtype=np.float32)
        return {
            "images": obs.get("images", {}),
            "state": self._state.copy(),
            "instruction": self.instruction,
            "timestamp": float(obs.get("timestamp", time.time())),
        }

    def get_info(self) -> dict:
        info = super().get_info()
        info.update(
            {
                "robot_type": self.robot.get_info().get("robot_type", "unknown"),
                "control_hz": self.robot.control_hz,
                "action_limit": float(self.robot.get_info().get("action_limit", 0.01)),
                "dry_run": self.dry_run,
            }
        )
        return info

    def step(self, action: np.ndarray) -> StepResult:
        if self._state is None:
            raise RuntimeError("step 前必须 reset()")
        action = np.asarray(action, dtype=np.float32)
        # 安全链：scale -> clamp -> smooth（相对最新反馈状态），dry-run 只记录不发送
        command = self.robot.apply_safety(action, self._state)
        self.command_log.append(command)
        if not self.dry_run:
            # send_action 内部对同一反馈状态重复同一限幅（scale=1/clamp 下幂等）
            self.robot.send_action(action)
            self.command_sent_steps += 1
        self._steps += 1
        return StepResult(
            observation=self.get_observation(),
            reward=None,  # 叠衣服成功判定延后
            terminated=False,
            truncated=self._steps >= self.max_episode_steps,
            info={"dry_run": self.dry_run, "command": command},
        )

    def close(self) -> None:
        print(
            f"FoldClothesTask 收尾：steps={self._steps} commands={len(self.command_log)} "
            f"command_sent={self.command_sent_steps} dry_run={self.dry_run}",
            flush=True,
        )
        try:
            self.robot.disconnect()  # 与 GUI client 一致：只停相机，不动臂
        except Exception as exc:  # noqa: BLE001 — 清理失败不应掩盖评测结果
            print(f"Warning: FoldClothesTask 关闭本体失败: {exc}")
