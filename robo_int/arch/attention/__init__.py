"""action-query→vision 注意力变体（通用接口见 base.py，实验矩阵见 README.md）。"""

from robo_int.arch.attention.base import (
    AttentionVariantChain,
    BaseAttentionVariant,
    GazeBiasVariant,
    HookContext,
)

__all__ = ["BaseAttentionVariant", "AttentionVariantChain", "GazeBiasVariant", "HookContext"]
