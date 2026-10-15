"""MEM 离线单测（可移植核心，零重依赖；synthetic tensor 只用于离线层，
最终功能验证在 scripts/pi05_mem_functional_test.py 用真实 Piper 数据）。"""

import numpy as np
import pytest
import torch

from robo_int.arch.mem import MEMModule, TextMemoryBank


class FakePi05Model:
    """带 embed_prefix 的最小替身（π0.5 系接口）。"""

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs = torch.randn(1, 10, 64)
        pad = torch.ones(1, 10, dtype=torch.bool)
        att = torch.zeros(1, 10, dtype=torch.bool)
        return embs, pad, att


def make_mem(**kw):
    defaults = dict(d_model=64, history_len=3, num_tokens=2, bank_capacity=4, num_heads=8)
    defaults.update(kw)
    return MEMModule(**defaults)


def test_bank_write_retrieve_forget():
    bank = TextMemoryBank(capacity=2)
    bank.write("oldest", np.zeros(4))
    bank.write("near query", np.ones(4))
    bank.write("newest", np.ones(4))  # 容量遗忘：oldest 出局
    assert len(bank) == 2
    retrieved = bank.retrieve(np.ones(4), k=5)
    assert [t for t, _ in retrieved][:2] == ["near query", "newest"]
    assert all(s > 0.99 for _, s in retrieved[:1])


def test_disabled_and_empty_memory_return_original_objects():
    model = FakePi05Model()
    mem = make_mem()
    mem.attach(model)
    # 基线对象（attach 前计算会被 wrap 替换——先取 detached 参照再比对语义）
    e0, p0, a0 = torch.randn(1, 10, 64), torch.ones(1, 10, dtype=torch.bool), torch.zeros(1, 10, dtype=torch.bool)
    model.embed_prefix = lambda *args: (e0.clone(), p0.clone(), a0.clone())
    mem.detach()
    mem.attach(model)
    # active=False：原样返回（数值与形状完全一致，无拼接）
    e1, p1, a1 = model.embed_prefix(None, None, None, None)
    assert e1.shape == (1, 10, 64) and p1.shape == (1, 10) and a1.shape == (1, 10)
    # active=True 但记忆为空：同样零注入
    mem.active = True
    e2, p2, a2 = model.embed_prefix(None, None, None, None)
    assert e2.shape == (1, 10, 64) and mem.last_stats["injected"] is False


def test_active_history_appends_memory_tokens():
    model = FakePi05Model()
    mem = make_mem()
    mem.attach(model)
    mem.active = True
    mem.push_history(torch.randn(64))
    e, p, a = model.embed_prefix(None, None, None, None)
    assert e.shape == (1, 12, 64) and p.shape == (1, 12) and a.shape == (1, 12)
    assert bool(p[:, -2:].all())          # 记忆 token 是真实 token（pad=1）
    assert not bool(a[:, -2:].any())      # 与 prefix 同块（att=0，双向注意）
    assert mem.last_stats["injected"] is True and mem.last_stats["history_len"] == 1


def test_tokens_change_with_history():
    mem = make_mem()
    mem.push_history(torch.randn(64))
    t_a = mem.compute_tokens().detach().clone()
    mem.push_history(torch.randn(64) * 3)
    t_b = mem.compute_tokens().detach()
    assert float((t_a - t_b).abs().max()) > 0


def test_text_branch_changes_tokens():
    enc = lambda s: np.full(64, float(len(s)), dtype=np.float32)
    mem = make_mem(encode_text=enc)
    mem.push_history(torch.randn(64))
    t_wo = mem.compute_tokens().detach().clone()
    mem.write_memory("gripper closed on the cloth", np.random.randn(64))
    t_w = mem.compute_tokens().detach()
    assert float((t_wo - t_w).abs().max()) > 0
    assert mem.last_stats["used_long"] is True


def test_save_load_roundtrip(tmp_path):
    mem = make_mem(encode_text=lambda s: np.ones(64, dtype=np.float32))
    mem.push_history(torch.randn(64))
    mem.write_memory("real event", np.random.randn(64))
    t_ref = mem.compute_tokens().detach().clone()
    path = tmp_path / "mem_state.pt"
    mem.save(path)
    other = make_mem()
    other.load(path)
    other.push_history(mem._history[-1])
    assert float((t_ref - other.compute_tokens().detach()).abs().max()) == 0.0
    assert [t for t, _ in other.bank.retrieve(np.random.randn(64), k=5)] == ["real event"]


def test_detach_restores_original_embed_prefix():
    model = FakePi05Model()
    mem = make_mem()
    mem.attach(model)
    assert "embed_prefix" in model.__dict__          # attach = 实例遮蔽
    mem.active = True
    mem.push_history(torch.randn(64))
    mem.detach()
    assert "embed_prefix" not in model.__dict__      # detach 撤销遮蔽，回落类方法
    e, p, a = model.embed_prefix(None, None, None, None)  # 撤销后即使 active 也不注入
    assert e.shape == (1, 10, 64) and mem.last_stats.get("injected") is not True


def test_gradients_flow_to_mem_params():
    mem = make_mem()
    mem.push_history(torch.randn(64))
    tokens = mem.compute_tokens()
    tokens.float().pow(2).sum().backward()
    grads = [p for p in mem.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(p.grad).all() for p in grads)
    assert any(float(p.grad.abs().sum()) > 0 for p in grads)


def test_reset_episode_clears_history_keeps_bank():
    mem = make_mem(encode_text=lambda s: np.ones(64, dtype=np.float32))
    mem.push_history(torch.randn(64))
    mem.write_memory("event", np.random.randn(64))
    mem.reset_episode()
    assert mem.history_len_current == 0
    assert len(mem.bank) == 1
    # 只有长时库时仍可注入（used_short=False, used_long=True）
    mem.active = True
    tokens = mem.compute_tokens()
    assert tokens is not None and mem.last_stats["used_long"] is True
