"""attention 通用接口测试：钩子编排、变体组合、gaze-bias 打样的数值行为。"""

import torch

from robo_int.arch.attention.base import (
    AttentionVariantChain,
    BaseAttentionVariant,
    GazeBiasVariant,
    HookContext,
)
from robo_int.models.mock import MockVLA


class PassthroughVariant(BaseAttentionVariant):
    """全部钩子走默认实现的空变体：输出应与无挂件一致。"""

    name = "passthrough"


class QueryScaleVariant(BaseAttentionVariant):
    """query 乘 2 的最简干预，验证钩子确实生效且可组合。"""

    name = "query_scale"

    def query_transform(self, queries, context):
        return queries * 2.0


def test_passthrough_matches_plain_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 4, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    context = HookContext(layer_name="test")

    variant = PassthroughVariant()
    out = variant.apply_cross_attention(q, k, v, context)

    d = q.shape[-1]
    ref = torch.softmax(q @ k.transpose(-2, -1) / d**0.5, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-6)


def test_gaze_bias_changes_output_and_is_learnable():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 16), torch.randn(1, 64, 16), torch.randn(1, 64, 16)
    context = HookContext(layer_name="test")

    variant = GazeBiasVariant(grid_hw=(8, 8), num_keys=64)
    with torch.no_grad():
        variant.gaze_map.normal_(0, 0.5)
    out_bias = variant.apply_cross_attention(q, k, v, context)
    out_plain = PassthroughVariant().apply_cross_attention(q, k, v, context)

    assert not torch.allclose(out_bias, out_plain, atol=1e-4)
    assert any(p.requires_grad for p in variant.parameters())


def test_variant_chain_combines_and_indexes():
    chain = AttentionVariantChain([PassthroughVariant(), QueryScaleVariant()])
    model = MockVLA()
    chain.attach(model)  # BaseArchModule 契约：可挂到任意 BaseVLA

    q, k, v = torch.randn(1, 4, 16), torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    context = HookContext(layer_name="test")
    out_chain = chain.variants[-1].apply_cross_attention(q, k, v, context)  # 组合路径由编排器串
    out_single = QueryScaleVariant().apply_cross_attention(q, k, v, context)
    assert torch.allclose(out_chain, out_single, atol=1e-6)

    assert isinstance(chain["query_scale"], QueryScaleVariant)
    chain.detach()  # detach 后可逆（无残留状态即可再次 attach）
    chain.attach(model)


def test_hook_context_carries_memory_for_cross_module():
    """MEM 写 memory -> attention 变体读：跨挂件上下文总线契约。"""
    context = HookContext()
    context.memory = torch.zeros(1, 8, 16)  # 模拟 MEM 挂件写入

    class MemoryReader(BaseAttentionVariant):
        name = "memory_reader"
        saw_memory = False

        def query_transform(self, queries, ctx):
            if ctx.memory is not None and ctx.memory.shape[-1] == queries.shape[-1]:
                MemoryReader.saw_memory = True
            return queries

    MemoryReader().apply_cross_attention(
        torch.randn(1, 4, 16), torch.randn(1, 6, 16), torch.randn(1, 6, 16), context
    )
    assert MemoryReader.saw_memory
