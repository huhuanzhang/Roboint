"""TTT 离线单测（可移植核心，零重依赖；synthetic tensor 只用于离线层，
最终功能验证在 scripts/pi05_ttt_functional_test.py 用真实 Piper 数据）。"""

import pytest
import torch
from torch import nn

from robo_int.arch.ttt import TTTLayer, TTTLinearLayer


class FakePi05Model(nn.Module):
    """带 embed_prefix 的最小替身（π0.5 系接口）。"""

    def __init__(self, d_model=64, seq=12):
        super().__init__()
        self.own = nn.Linear(d_model, d_model)  # 验证 state_dict 不被挂件污染
        self._d_model, self._seq = d_model, seq

    def embed_prefix(self, images, img_masks, tokens, masks):
        torch.manual_seed(7)
        embs = torch.randn(1, self._seq, self._d_model)
        pad = torch.ones(1, self._seq, dtype=torch.bool)
        att = torch.zeros(1, self._seq, dtype=torch.bool)
        return embs, pad, att


def make_ttt(**kw):
    defaults = dict(d_model=64, latent_dim=16, lr=0.1, gate=0.5,
                    chunk_size=4, persist=False)
    defaults.update(kw)
    return TTTLinearLayer(**defaults)


def token_reference(layer, x, lr):
    """逐 token 在线参考递推（显式循环，无 einsum）：查询用更新前 W。"""
    with torch.no_grad():
        h = layer.norm_in(x.float())
        q = torch.nn.functional.normalize(layer.wq(h), dim=-1)
        k = torch.nn.functional.normalize(layer.wk(h), dim=-1)
        v = layer.wv(h)
        w = layer.fast_init.detach().clone().unsqueeze(0)
        b, l, _ = q.shape
        outs = []
        for t in range(l):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            zt = torch.matmul(w, qt.unsqueeze(-1)).squeeze(-1)  # (B, D)
            outs.append(zt)
            err = torch.matmul(w, kt.unsqueeze(-1)).squeeze(-1) - vt  # (B, D)
            w = w - lr * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
        z = torch.cat([o.unsqueeze(1) for o in outs], dim=1)
        z = layer.wo(z)
        delta = layer.gate * layer.norm_out(z)
        return x + delta.to(x.dtype)


def test_constructor_validation():
    with pytest.raises(ValueError):
        TTTLinearLayer(d_model=0)
    with pytest.raises(ValueError):
        TTTLinearLayer(latent_dim=-1)
    with pytest.raises(ValueError):
        TTTLinearLayer(chunk_size=0)
    with pytest.raises(ValueError):
        TTTLinearLayer(lr=0.0)
    with pytest.raises(ValueError):
        TTTLinearLayer(lr=1.5)  # 超出归一化稳定域
    with pytest.raises(ValueError):
        TTTLinearLayer(gate=1.5)
    assert TTTLayer is TTTLinearLayer


def test_zero_effect_returns_original_object():
    x = torch.randn(2, 10, 64)
    layer = make_ttt(gate=0.0)
    layer.active = True
    out, recon = layer.forward_tokens(x)
    assert out is x and recon is None
    layer = make_ttt(gate=0.5)
    layer.active = False
    out, recon = layer.forward_tokens(x)
    assert out is x and recon is None
    assert layer.last_stats["applied"] is False


def test_shapes_and_dtype_preserved():
    layer = make_ttt()
    layer.active = True
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(1, 13, 64, dtype=dtype)  # 13 = 非整块序列
        out, recon = layer.forward_tokens(x)
        assert out.shape == x.shape and out.dtype == dtype
        assert recon is not None and torch.isfinite(recon)


def test_chunk1_matches_token_reference():
    torch.manual_seed(3)
    x = torch.randn(2, 11, 64)
    layer = make_ttt(chunk_size=1, lr=0.03, gate=0.7)
    layer.active = True
    out, _ = layer.forward_tokens(x.clone())
    ref = token_reference(layer, x, lr=0.03)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_multi_chunk_matches_explicit_chunk_reference():
    torch.manual_seed(4)
    x = torch.randn(1, 12, 64)
    chunk, lr = 5, 0.02
    layer = make_ttt(chunk_size=chunk, lr=lr, gate=0.6)
    layer.active = True
    out, _ = layer.forward_tokens(x.clone())
    with torch.no_grad():
        h = layer.norm_in(x.float())
        q = torch.nn.functional.normalize(layer.wq(h), dim=-1)
        k = torch.nn.functional.normalize(layer.wk(h), dim=-1)
        v = layer.wv(h)
        w = layer.fast_init.detach().clone().unsqueeze(0)
        outs = []
        for s in range(0, 12, chunk):
            e = min(s + chunk, 12)
            outs.append(torch.einsum("bdk,bck->bcd", w, q[:, s:e]))
            err = torch.einsum("bdk,bck->bcd", w, k[:, s:e]) - v[:, s:e]
            w = w - lr * torch.einsum("bcd,bck->bdk", err, k[:, s:e]) / (e - s)
        z = layer.wo(torch.cat(outs, dim=1))
        ref = x + (layer.gate * layer.norm_out(z)).to(x.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_delta_rule_error_decreases():
    """k̂ 单位归一下，一步更新后重建误差按 (1 − lr·‖k̂‖²) = (1 − lr) 收缩。"""
    torch.manual_seed(5)
    x = torch.randn(1, 8, 64)
    layer = make_ttt(chunk_size=1, lr=0.5)
    layer.active = True
    with torch.no_grad():
        h = layer.norm_in(x.float())
        k = torch.nn.functional.normalize(layer.wk(h), dim=-1)
        v = layer.wv(h)
        w = layer.fast_init.detach().clone().unsqueeze(0)
        shrunk = 0
        for t in range(x.shape[1]):
            kt, vt = k[:, t], v[:, t]
            err = torch.matmul(w, kt.unsqueeze(-1)).squeeze(-1) - vt
            before = float(err.norm())
            w = w - 0.5 * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            after = float((torch.matmul(w, kt.unsqueeze(-1)).squeeze(-1) - vt).norm())
            if before > 1e-8:
                assert after <= before + 1e-6
                shrunk += int(after < before)
        assert shrunk > 0


def test_persist_mode_and_reset():
    torch.manual_seed(6)
    x = torch.randn(1, 12, 64)
    layer = make_ttt(persist=True)
    layer.active = True
    out1, _ = layer.forward_tokens(x.clone())
    assert layer.fast_state_persisted
    out2, _ = layer.forward_tokens(x.clone())
    assert not torch.equal(out1, out2)  # 快权重复用 -> 同输入输出变化
    assert layer.last_stats["reused_state"] is True
    layer.reset_episode()
    out3, _ = layer.forward_tokens(x.clone())
    assert torch.equal(out1, out3)  # 重置后回到 per-sequence 起点（确定性）
    assert layer.last_stats["reused_state"] is False


def test_non_persist_deterministic_per_request():
    torch.manual_seed(6)
    x = torch.randn(1, 12, 64)
    layer = make_ttt(persist=False)
    layer.active = True
    out1, _ = layer.forward_tokens(x.clone())
    out2, _ = layer.forward_tokens(x.clone())
    assert torch.equal(out1, out2) and not layer.fast_state_persisted


def test_attach_wraps_and_detaches_restores():
    model = FakePi05Model()
    before = model.embed_prefix(None, None, None, None)
    layer = make_ttt()
    layer.attach(model)
    assert "embed_prefix" in model.__dict__
    e, p, a = model.embed_prefix(None, None, None, None)
    assert torch.equal(e, before[0]) and torch.equal(p, before[1])  # inactive 零效应
    layer.active = True
    e2, p2, a2 = model.embed_prefix(None, None, None, None)
    assert not torch.equal(e2, before[0]) and torch.equal(p2, before[1])
    layer.detach()
    assert "embed_prefix" not in model.__dict__
    after = model.embed_prefix(None, None, None, None)
    assert torch.equal(after[0], before[0])


def test_attach_restores_previous_instance_wrap():
    model = FakePi05Model()
    sentinel = lambda *args: (torch.randn(1, 12, 64), None, None)  # 真实包裹必返 tensor 元组
    model.embed_prefix = sentinel  # 模拟 MEM 先挂的实例级包裹
    layer = make_ttt()
    layer.attach(model)
    layer.active = True
    model.embed_prefix(None, None, None, None)  # 生效路径
    layer.detach()
    assert model.embed_prefix is sentinel


def test_attached_model_not_in_state_dict_or_parameters():
    model = FakePi05Model()
    keys_before = set(model.state_dict().keys())
    params_before = {id(p) for p in model.parameters()}
    layer = make_ttt()
    layer.attach(model)
    assert set(model.state_dict().keys()) == keys_before
    assert {id(p) for p in model.parameters()} == params_before
    layer_ids = {id(p) for p in layer.parameters()}
    assert len(layer_ids) > 0 and not (layer_ids & params_before)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(8)
    x = torch.randn(1, 12, 64)
    layer = make_ttt(persist=True)
    layer.active = True
    layer.forward_tokens(x.clone())  # 产生持久快权重
    layer.save(tmp_path / "ttt.pt")
    fresh = make_ttt(persist=True)
    fresh.load(tmp_path / "ttt.pt")
    fresh.active = True
    out_a, _ = layer.forward_tokens(x.clone())
    out_b, _ = fresh.forward_tokens(x.clone())
    torch.testing.assert_close(out_a, out_b)  # 参数 + 快权重同载
    snap = layer.state_dict()
    fresh2 = make_ttt(persist=False)
    fresh2.load_state_dict(snap)
    fresh2.active = True
    out_c, _ = layer.forward_tokens(x.clone())
    out_d, _ = fresh2.forward_tokens(x.clone())
    assert not torch.equal(out_c, out_d)  # 非 persist 不读快权重（per-sequence）


def test_gradients_reach_ttt_params():
    layer = make_ttt()
    layer.active = True
    x = torch.randn(1, 12, 64)
    out, recon = layer.forward_tokens(x)
    assert recon is not None
    (out.float().pow(2).mean() + recon.mean()).backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    grad_norm = sum(p.grad.norm().item() for p in layer.parameters())
    assert grad_norm > 0


def test_last_stats_fields():
    layer = make_ttt()
    layer.active = True
    x = torch.randn(1, 12, 64)
    layer.forward_tokens(x)
    stats = layer.last_stats
    assert stats["applied"] is True
    assert stats["seq_len"] == 12 and stats["chunks"] == 3
    assert stats["out_nan_count"] == 0 and stats["out_inf_count"] == 0
    assert stats["w_delta_norm"] >= 0 and stats["recon_loss"] >= 0
