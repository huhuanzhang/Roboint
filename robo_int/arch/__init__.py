"""架构改进挂件：attention 变体（接口已实现）/ LoRA 工具 / MEM / TTT / in-context / 轨迹 prompt（后四为 stub，见各自 README）。"""

from robo_int.arch.attention.base import (
    AttentionVariantChain,
    BaseAttentionVariant,
    GazeBiasVariant,
    HookContext,
)
from robo_int.arch.base import BaseArchModule

__all__ = [
    "BaseArchModule",
    "BaseAttentionVariant",
    "AttentionVariantChain",
    "GazeBiasVariant",
    "HookContext",
]
