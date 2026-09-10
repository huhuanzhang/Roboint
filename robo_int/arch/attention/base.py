"""attention 变体通用接口（用户点名的基础设施）。

设计目标：让 AtVLA / GazeVLA / PelicanVLA 式的注意力机制改进都落在同一套钩子上，
换变体不改基模代码；VLA-Trace 式诊断作为独立工具（diagnostics.py，TODO）。

五个钩子在「action query 交叉注意力视觉特征」的标准通路上的位置：

    视觉特征 ──► key_value_transform ──► K,V ─┐
                                              ├─► attention(+bias) ─► post_attention ─► 动作头
    action query ──► query_transform ──► Q ───┤
              └──► pre_attention（动作前生成/注意，Pelican 式）▲
                                  attention_bias ◄── gaze 先验/矫正 mask

HookContext 是跨挂件的上下文总线：MEM 挂件写入 memory token，
attention 变体读取——挂件之间即插即组合，互不直接 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from robo_int.arch.base import BaseArchModule


@dataclass
class HookContext:
    """attention 钩子的统一上下文（跨挂件共享的数据总线）。"""

    observation: dict = field(default_factory=dict)   # 当前观测（images/state）
    instruction: str | None = None                    # 语言指令
    memory: torch.Tensor | None = None                # MEM 等记忆挂件写入的记忆 token
    uncertainty: torch.Tensor | None = None           # 动作块不确定性估计（AtVLA 式门控用）
    layer_name: str = ""                              # 当前挂载的注意力层标识
    extra: dict = field(default_factory=dict)         # 变体私有数据


class BaseAttentionVariant(nn.Module, BaseArchModule):
    """attention 变体基类：五个钩子实现任意子集，未实现的自动跳过。

    继承 nn.Module：变体可带可学习参数（如 gaze 先验、register token 投影），
    由训练循环统一优化；attach/detach 给出默认生命周期（记录挂载的基模引用，
    真实接入点在 π0.5/GR00T wrapper 暴露 get_attention_hook_points() 后由子类重写）。

    约定：
    - 所有钩子必须保持形状不变或返回合法的新形状（由变体自己保证，编排器不校验）；
    - 只改信息流，不改 BaseVLA 对外接口（predict_actions/compute_loss 签名不动）；
    - attach/detach 的可逆性沿用 BaseArchModule 契约（A/B 实验正确性依赖）。
    """

    name: str = "attention_variant"

    def __init__(self):
        nn.Module.__init__(self)
        self._attached_model = None

    # ---------------------------------------------------------------- 生命周期（默认实现）

    def attach(self, model) -> None:
        self._attached_model = model

    def detach(self) -> None:
        self._attached_model = None

    # ---------------------------------------------------------------- 钩子（可选实现）

    def pre_attention(self, context: HookContext) -> None:
        """动作生成前的准备阶段（Pelican-VLA 式：先注意/生成未来再动作）。

        适合做未来帧生成、显式注意目标选择等重计算；只写 context，无返回值。
        """
        return None

    def query_transform(
        self, queries: torch.Tensor, context: HookContext
    ) -> torch.Tensor:
        """变换 action query（GazeVLA 式：gaze/intention embedding 注入 query）。"""
        return queries

    def key_value_transform(
        self, keys: torch.Tensor, values: torch.Tensor, context: HookContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """变换视觉 key/value（AtVLA 式：register token 注入、注意力引导裁剪增强）。"""
        return keys, values

    def attention_bias(self, bias: torch.Tensor | None, context: HookContext) -> torch.Tensor | None:
        """给 attention logits 加偏置（gaze 先验图、attention 矫正 mask）。"""
        return bias

    def post_attention(
        self, output: torch.Tensor, context: HookContext
    ) -> torch.Tensor:
        """注意力输出后处理（AtVLA 式：不确定性门控的局部精化）。"""
        return output

    # ---------------------------------------------------------------- 组合工具

    def apply_cross_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        context: HookContext,
        attn: nn.Module | None = None,
    ) -> torch.Tensor:
        """串行编排一次「Q/K/V 钩子 -> 注意力 -> 输出钩子」的完整通路。

        attn 为 None 时用缩放点积注意力的最小实现（bias 经 additive 注入）；
        基模接入后传真实的 cross-attention 模块（权重结构保持基模原生，
        钩子只在外围变换）。
        """
        self.pre_attention(context)
        queries = self.query_transform(queries, context)
        keys, values = self.key_value_transform(keys, values, context)

        if attn is not None:
            # 真实 attn 模块不暴露 logits 接口，attention_bias 无法注入。
            # 显式拒绝而非静默丢弃（GazeBiasVariant 等依赖 bias 的变体
            # 接 π0.5 时会在此处炸出，直到基模 wrapper 暴露 bias 钩子）
            bias = self.attention_bias(None, context)
            if bias is not None:
                raise NotImplementedError(
                    "传入真实 attn 模块时 attention_bias 无法注入（基模 attention "
                    "不暴露 logits）；请改用 query_transform/key_value_transform "
                    "实现引导，或等基模 wrapper 暴露 bias 钩子后扩展本通路"
                )
            output = attn(queries, keys, values)
        else:
            # 最小缩放点积注意力（不含掩码；bias 经 additive 注入）
            d = queries.shape[-1]
            scores = queries @ keys.transpose(-2, -1) / (d**0.5)
            bias = self.attention_bias(None, context)
            if bias is not None:
                scores = scores + bias
            output = torch.softmax(scores, dim=-1) @ values

        return self.post_attention(output, context)


class AttentionVariantChain(BaseArchModule):
    """多个 attention 变体的串行组合（顺序敏感，按注册顺序依次过钩子）。

    注意：本类**不是** nn.Module（BaseArchModule 契约未强制），parameters()
    为手写去重迭代器——变体参数不进入任何 state_dict，保存/加载时悬空。
    需要持久化时按 self.variants 逐个 save/load 各变体的 state_dict，
    或后续把本类升级为 nn.Module 容器（ModuleList）。
    """

    name = "attention_chain"

    def __init__(self, variants: list[BaseAttentionVariant]):
        assert len(variants) > 0
        self.variants = list(variants)
        self._attached_model = None

    def attach(self, model) -> None:
        self._attached_model = model
        for v in self.variants:
            v.attach(model)

    def detach(self) -> None:
        for v in self.variants:
            v.detach()
        self._attached_model = None

    def parameters(self):
        seen: set[int] = set()
        for v in self.variants:
            for p in v.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def __getitem__(self, name: str) -> BaseAttentionVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"链中没有变体 {name}；现有：{[v.name for v in self.variants]}")


# ---------------------------------------------------------------- 便捷实现：gaze-bias 打样

class GazeBiasVariant(BaseAttentionVariant):
    """v1 打样：attention_bias 注入 gaze 先验（GazeVLA 路线的最小版）。

    gaze_map: 可学习 (H', W') 先验，插值到 keys 的空间网格后加进 logits。
    仅证明接口闭环，真实 gaze 数据接入后替换 gaze_map 的来源。
    """

    name = "gaze_bias"

    def __init__(self, grid_hw: tuple[int, int] = (8, 8), num_keys: int = 64):
        super().__init__()
        self.grid_hw = grid_hw
        self.gaze_map = nn.Parameter(torch.zeros(1, 1, *grid_hw))
        self.num_keys = num_keys

    def attention_bias(self, bias, context: HookContext):
        expected = self.grid_hw[0] * self.grid_hw[1]
        if self.num_keys != expected:
            # keys 非均匀网格时退化为无偏置（打样版限制，真实接入时按 patch 坐标插值）
            return bias
        flat = self.gaze_map.flatten(2)  # (1,1,HW)
        return flat if bias is None else bias + flat
