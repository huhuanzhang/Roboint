"""任务环境抽象。

真机任务的 gym-like 接口，但面向真机时序做了三处关键扩展：
1. execute_chunk: VLA 输出的是动作块（action chunk），一次下发多步、逐步观测；
2. 重置语义多样化：scripted（自动）/ teleop（人操）/ manual（人摆场景）；
3. 奖励与成功判定分离：奖励可以延后（人工/SARM2 奖励模型离线打分）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

# 策略函数：观测 dict -> 单步动作。rollout runner / teleop 都适配到这个签名。
PolicyFn = Callable[[dict], np.ndarray]


class ResetMode(str, Enum):
    SCRIPTED = "scripted"  # 程序化重置（如固定插销位姿）
    TELEOP = "teleop"      # 人操机器人回到初始位
    MANUAL = "manual"      # 人手动摆放场景/物体


@dataclass
class StepResult:
    """step() 的返回，字段命名对齐 gymnasium 便于迁移。"""

    observation: dict
    reward: float | None = None  # None = 延后打分（人工/奖励模型）
    terminated: bool = False     # 任务成功/失败等终止
    truncated: bool = False      # 超时截断
    info: dict = field(default_factory=dict)


class BaseTask(ABC):
    """真机任务环境。生命周期：reset -> (step | execute_chunk)* -> 结束。"""

    reset_mode: ResetMode = ResetMode.SCRIPTED
    max_episode_steps: int = 100

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict:
        """重置任务，返回初始观测。可能阻塞等待人工确认（manual/teleop 模式）。"""

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """执行单步动作并返回 StepResult。"""

    @abstractmethod
    def get_observation(self) -> dict:
        """当前观测。约定结构::

            {
              "images": {"cam_wrist": np.ndarray(H,W,3) uint8, ...},
              "state": np.ndarray,          # 本体状态向量
              "instruction": str,           # 语言指令
            }
        """

    def execute_chunk(self, chunk: np.ndarray, record: bool = True) -> StepResult:
        """执行一个动作块（VLA 的标准输出）。

        默认实现：逐步执行，块末聚合一次观测；中间观测可存 info['intermediate']。
        灵巧手/接触任务可以重写此方法做更细的聚合（如取每步力反馈）。
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        result = None
        intermediates = []
        for action in chunk:
            result = self.step(action)
            if record:
                intermediates.append(result.observation)
            if result.terminated or result.truncated:
                break
        assert result is not None, "动作块不能为空"
        result.info["intermediate_observations"] = intermediates
        return result

    def success(self) -> bool:
        """当前 episode 是否成功（真机上常由规则/人工/奖励模型判定）。"""
        return False

    def get_info(self) -> dict:
        """任务元信息：动作语义、维度、控制频率等。"""
        return {
            "task_name": type(self).__name__,
            "reset_mode": self.reset_mode.value,
            "max_episode_steps": self.max_episode_steps,
        }
