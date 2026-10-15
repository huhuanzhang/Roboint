"""ActionQueryVisionGate 离线单测（fake PaliGemmaWithExpert，零重依赖）。

核心验证：
- gate=0 时 KV 对象原样返回（无任何算术）→ bit 级零效应；
- gate>0 时只有视觉切片的 V 被 (1+g) 缩放，K 与语言段不动；
- 可微路径（layer_gates 反传）；
- 相机映射不可靠时拒绝切相机（不硬猜）。
"""

import pytest
import torch

from robo_int.models.pi05.action_query_vision import ActionQueryVisionGate

SEQ, VISION, LAYERS = 10, 6, 2
CAMERAS = ["head", "left_wrist", "right_wrist"]


class FakePwe:
    """最小 PaliGemmaWithExpert 替身：prefix-only 调用返回固定 KV。"""

    def __init__(self):
        torch.manual_seed(0)
        self.kv = [
            (torch.randn(1, 2, SEQ, 4), torch.randn(1, 2, SEQ, 4)) for _ in range(LAYERS)
        ]
        self.calls = 0

    def forward(self, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, adarms_cond=None):
        self.calls += 1
        return [torch.zeros(1, SEQ, 4), None], tuple(self.kv)


def _gate(pwe, fixed=0.0, cameras=CAMERAS):
    return ActionQueryVisionGate(pwe, num_layers=LAYERS, vision_tokens=VISION,
                                 fixed_gate=fixed, camera_names=cameras)


def test_zero_gate_returns_original_kv_object():
    pwe = FakePwe()
    gate = _gate(pwe)
    out = gate._transform_kv(tuple(pwe.kv))
    assert out is not None
    # 逐层 K/V 均为原对象（无 clone 无算术）
    for (k0, v0), (k1, v1) in zip(pwe.kv, out):
        assert k1 is k0 and v1 is v0
    assert gate.last_stats["transformed"] is False


def test_zero_gate_wrapped_forward_bit_parity():
    pwe = FakePwe()
    gate = _gate(pwe, fixed=0.0)
    gate.attach()
    try:
        outs, kv = pwe.forward(inputs_embeds=[torch.zeros(1, SEQ, 4), None], use_cache=True)
        for (k0, v0), (k1, v1) in zip(pwe.kv, kv):
            assert k1 is k0 and v1 is v0
    finally:
        gate.detach()
    assert pwe.forward.__name__ == "forward"  # detach 恢复（FakePwe 原方法）


def test_positive_gate_scales_only_vision_v():
    pwe = FakePwe()
    gate = _gate(pwe, fixed=0.5)
    out = gate._transform_kv(tuple(pwe.kv))
    for (k0, v0), (k1, v1) in zip(pwe.kv, out):
        assert k1 is k0  # K 不动
        torch.testing.assert_close(v1[:, :, :VISION, :], v0[:, :, :VISION, :] * 1.5)
        torch.testing.assert_close(v1[:, :, VISION:, :], v0[:, :, VISION:, :])  # 语言段不动
    assert gate.last_stats["transformed"] is True


def test_per_camera_mass_and_mapping_guard():
    pwe = FakePwe()
    gate = _gate(pwe, fixed=0.1)
    gate._transform_kv(tuple(pwe.kv))
    mass = gate.last_stats.get("camera_v_mass")
    assert set(mass) == {"head", "left_wrist", "right_wrist", "language"}

    # 映射不可靠（vision_tokens 不能整除相机数）：不硬切，mass 缺失
    gate_bad = ActionQueryVisionGate(pwe, num_layers=LAYERS, vision_tokens=7,
                                     fixed_gate=0.1, camera_names=CAMERAS)
    gate_bad._transform_kv(tuple(pwe.kv))
    assert "camera_v_mass" not in gate_bad.last_stats


def test_layer_gates_differentiable_through_transform():
    pwe = FakePwe()
    gate = _gate(pwe, fixed=0.0)
    with torch.no_grad():
        gate.layer_gates.fill_(0.3)  # 非零才走变换路径
    out = gate._transform_kv(tuple(pwe.kv))
    total = sum(v.float().sum() for (_, v) in out)
    total.backward()
    assert gate.layer_gates.grad is not None
    assert float(gate.layer_gates.grad.abs().sum()) > 0


def test_learnable_gates_start_at_zero():
    gate = _gate(FakePwe())
    assert bool((gate.layer_gates == 0).all())
    assert bool((gate.effective_gates() == 0).all())


class FakeLayer:
    def __init__(self, k, v):
        self.keys, self.values = k, v


class FakeCache:
    """transformers DynamicCache 形状：.layers[i].keys/.values；__iter__ 出 3 元组。"""

    def __init__(self, kv):
        self.layers = [FakeLayer(k, v) for k, v in kv]

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, None


def test_dynamic_cache_container_transform_and_zero_gate():
    pwe = FakePwe()
    cache = FakeCache(pwe.kv)
    gate = _gate(pwe, fixed=0.5)
    out = gate._transform_kv(cache)
    assert out is cache  # DynamicCache：原地更新、返回同一对象
    for layer, (k0, v0) in zip(cache.layers, pwe.kv):
        assert layer.keys is k0  # K 不动
        torch.testing.assert_close(layer.values[:, :, :VISION, :], v0[:, :, :VISION, :] * 1.5)
        torch.testing.assert_close(layer.values[:, :, VISION:, :], v0[:, :, VISION:, :])
    assert gate.last_stats["transformed"] is True
    assert gate.last_stats["container"] == "dynamic_cache"

    pwe0 = FakePwe()
    cache0 = FakeCache(pwe0.kv)
    gate0 = _gate(pwe0)
    out0 = gate0._transform_kv(cache0)
    assert out0 is cache0
    for layer, (k0, v0) in zip(cache0.layers, pwe0.kv):
        assert layer.keys is k0 and layer.values is v0  # gate=0：无 clone 无算术
    assert gate0.last_stats["transformed"] is False
