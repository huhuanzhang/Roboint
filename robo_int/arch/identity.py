"""A 类（external/runtime）架构挂件：包裹 BaseVLA，对外接口不变。

与 arch/base.py 的 B 类挂件（BaseArchModule.attach/detach，进入模型内部替换
组件/挂 forward hook）互补：

- A 类挂件作用在 BaseVLA 的边界上（观测进、动作出），任何 BaseVLA 实现都能挂，
  包括 Pi05ServerPolicy 这类「真网络在远端 server」的模型；
- B 类挂件（attention 变体 / LoRA / MEM / TTT）需要模型内部的接入点
  （attention 模块、权重、forward/backward），server 边界外的 client 无法实现。

挂载语义：包一层 wrapper 即 attach，去掉 wrapper 即 detach（factory.apply_arch）。
"""

from __future__ import annotations

import numpy as np
import torch

from robo_int.models.base import BaseVLA


class VLAWrapper(BaseVLA):
    """A 类挂件基类：观测/动作变换钩子默认全透传。

    子类按需重写：
    - transform_observation：观测进模型前（history 堆叠、prompt 改写、MEM 的
      外部记忆拼接等 runtime 侧能力）；
    - transform_actions：动作块出模型后（注意：安全链在 task/robot 层，不在此处）。

    契约：不得改变 predict_actions / compute_loss / extract_features 的签名与
    输出结构（与 BaseArchModule 的约束一致）。
    """

    name: str = "vla_wrapper"

    def __init__(self, model: BaseVLA):
        super().__init__()
        self.model = model
        self.chunk_size = model.chunk_size
        self.action_dim = model.action_dim
        self.feature_dim = model.feature_dim

    def transform_observation(self, observation: dict, instruction: str | None):
        return observation, instruction

    def transform_actions(self, chunk):
        return chunk

    def forward_actions(self, observation: dict, instruction: str | None = None):
        # 浅拷贝保护调用方：rollout 后续还要用原观测做记录
        observation, instruction = self.transform_observation(dict(observation), instruction)
        chunk = self.model.forward_actions(observation, instruction)
        return self.transform_actions(chunk)

    # ---------------------------------------------------------------- 训练/表征：委托内层

    def compute_loss(self, batch: dict) -> dict:
        return self.model.compute_loss(batch)

    def extract_features(self, observation: dict, instruction: str | None = None) -> torch.Tensor:
        return self.model.extract_features(observation, instruction)

    def load_checkpoint(self, path) -> None:
        self.model.load_checkpoint(path)

    def save_checkpoint(self, path) -> None:
        self.model.save_checkpoint(path)


class IdentityArch(VLAWrapper):
    """恒等挂件：什么都不改。

    用途：验证 A 类挂件挂载机制本身——包上它之后，输出必须与不挂完全一致
    （数值 bit 级相等，不只是 shape）。它是所有 runtime 挂件的 A/B 对照基线。
    """

    name = "identity"

    def transform_observation(self, observation: dict, instruction: str | None):
        return observation, instruction

    def transform_actions(self, chunk):
        return chunk
