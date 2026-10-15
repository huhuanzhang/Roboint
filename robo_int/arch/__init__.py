"""架构改进挂件：attention 变体（接口已实现）/ LoRA 工具 / MEM（最小真实实现，见 mem/memory.py）/
TTT / in-context / 轨迹 prompt（后三为 stub，见各自 README）。

两类挂件：
- A 类 external/runtime（VLAWrapper 子类）：包在 BaseVLA 边界上，factory.apply_arch 挂载；
- B 类 internal/model（BaseArchModule 子类）：进入模型内部（attention 钩子/权重/forward），
  需要基模暴露接入点，server 边界外的 client 无法实现。
"""

from robo_int.arch.attention.base import (
    AttentionVariantChain,
    BaseAttentionVariant,
    GazeBiasVariant,
    HookContext,
)
from robo_int.arch.base import BaseArchModule
from robo_int.arch.identity import IdentityArch, VLAWrapper

__all__ = [
    "BaseArchModule",
    "BaseAttentionVariant",
    "AttentionVariantChain",
    "GazeBiasVariant",
    "HookContext",
    "IdentityArch",
    "VLAWrapper",
]
