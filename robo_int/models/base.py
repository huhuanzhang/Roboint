"""VLA 基模统一接口。

π0.5 / GR00T N1.7 / Fast-WAM / Mock 都实现这一层。三个关键能力：
1. predict_actions   —— rollout 推理（真机/采集用，输入观测 dict 输出动作块）
2. compute_loss      —— SFT/BC 训练（接收 LeRobot 风格的 batch）
3. extract_features  —— 输出内部压缩表征（RLT 的 RL token 就挂在这里）

注意：基模的官方 SDK/权重加载放各自子包（models/pi05/ 等）并惰性导入，
本文件只定义接口，不 import 任何重依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BaseVLA(nn.Module, ABC):
    """所有 VLA 基模的统一封装。"""

    # 子类在 __init__ 里赋值：动作块长度、动作维度、表征维度
    chunk_size: int = 1
    action_dim: int = 0
    feature_dim: int = 0

    # ------------------------------------------------------------------ 推理

    @torch.no_grad()
    def predict_actions(self, observation: dict, instruction: str | None = None) -> np.ndarray:
        """ rollout 推理入口（子类一般只需实现 forward_actions）。

        返回 shape=(chunk_size, action_dim) 的动作块，float32。
        """
        self.eval()
        chunk = self.forward_actions(observation, instruction)
        return np.asarray(chunk, dtype=np.float32).reshape(self.chunk_size, self.action_dim)

    @abstractmethod
    def forward_actions(self, observation: dict, instruction: str | None = None):
        """根据观测产出动作块。可以是 flow-matching 采样 / 自回归解码 / 直接回归。"""

    # ------------------------------------------------------------------ 训练

    @abstractmethod
    def compute_loss(self, batch: dict) -> dict:
        """SFT/BC 损失。

        batch 约定（LeRobot 风格的 torch 张量）::

            {
              "observation.state":  Tensor(B, state_dim),
              "observation.images.*": Tensor(B, C, H, W),   # 0-1 float
              "task":               list[str],              # 语言指令
              "action":             Tensor(B, chunk_size*action_dim),  # 展平的目标动作块
            }

        返回 {"loss": 标量, ...其它监控项}。dropout 等正则由 infra 注入（见 infra/config.py）。
        """

    # ------------------------------------------------------------------ 表征（RLT 等算法用）

    def extract_features(self, observation: dict, instruction: str | None = None) -> torch.Tensor:
        """输出内部压缩表征（如 VLM backbone 的 pooled embedding），shape=(B?, feature_dim)。

        RLT（RL Token, arXiv:2604.23073）把这里作为状态输入接 actor-critic。
        基模若未实现则抛 NotImplementedError，算法侧应捕获并提示。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 extract_features；"
            "RLT/DSRL 等需要表征的算法无法在此基模上运行。"
        )

    # ------------------------------------------------------------------ 存取

    def save_checkpoint(self, path: str | Path) -> None:
        """保存权重（默认只存 state_dict，子类可扩展存配置/optimizer）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_state_dict(payload["state_dict"])
