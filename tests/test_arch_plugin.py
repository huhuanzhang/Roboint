"""A 类架构挂件（external/runtime wrapper）的功能测试。

核心验证：IdentityArch 挂上后输出与不挂 bit 级一致——挂载机制本身零侵入。
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robo_int.arch.identity import IdentityArch, VLAWrapper
from robo_int.infra.factory import apply_arch, build_model
from robo_int.models.base import BaseVLA
from robo_int.models.mock import MockVLA


def _mock_cfg(arch_name):
    return OmegaConf.create(
        {
            "model": {"name": "mock", "state_dim": 4, "action_dim": 2, "chunk_size": 3, "hidden_dim": 16},
            "arch": {"name": arch_name},
            "infra": {"regularization": {"dropout": 0.0}},
        }
    )


def _observation():
    return {"state": np.linspace(-0.3, 0.4, 4, dtype=np.float32), "images": {}}


def test_identity_arch_numerical_parity():
    """恒等挂件：同模型同观测，输出必须 bit 级相等（不只是 shape）。"""
    torch.manual_seed(0)
    model = MockVLA(state_dim=4, action_dim=2, chunk_size=3, hidden_dim=16)
    wrapped = IdentityArch(model)

    observation = _observation()
    raw = model.predict_actions(observation, "do it")
    out = wrapped.predict_actions(dict(observation), "do it")

    assert out.shape == raw.shape == (3, 2)
    np.testing.assert_array_equal(out, raw)  # bit 级
    assert wrapped.chunk_size == model.chunk_size and wrapped.action_dim == model.action_dim


def test_identity_arch_does_not_mutate_observation():
    model = MockVLA(state_dim=4, action_dim=2, chunk_size=3, hidden_dim=16)
    wrapped = IdentityArch(model)
    observation = _observation()
    snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in observation.items()}
    wrapped.predict_actions(observation, "do it")
    for key in observation:
        if isinstance(observation[key], np.ndarray):
            np.testing.assert_array_equal(observation[key], snapshot[key])


def test_factory_arch_switch():
    # arch=none：不包 wrapper，返回原模型
    cfg_none = _mock_cfg("none")
    model_none = build_model(cfg_none)
    assert type(model_none) is MockVLA

    # arch=identity：包 IdentityArch，内层是原模型
    cfg_identity = _mock_cfg("identity")
    model_wrapped = build_model(cfg_identity)
    assert isinstance(model_wrapped, IdentityArch)
    assert isinstance(model_wrapped, VLAWrapper)
    assert isinstance(model_wrapped.model, BaseVLA)

    # 无 arch 组的旧配置：行为同 none（向后兼容）
    cfg_legacy = _mock_cfg("none")
    OmegaConf.set_struct(cfg_legacy, True)
    del cfg_legacy.arch
    assert type(build_model(cfg_legacy)) is MockVLA


def test_factory_unknown_arch_rejected():
    with pytest.raises(KeyError, match="B 类"):
        apply_arch(MockVLA(state_dim=4, action_dim=2), _mock_cfg("lora"))


def test_wrapper_delegates_training_interface():
    model = MockVLA(state_dim=4, action_dim=2, chunk_size=1, hidden_dim=16)
    wrapped = IdentityArch(model)
    batch = {
        "observation.state": torch.zeros(2, 4),
        "action": torch.zeros(2, 2),
        "task": ["a", "b"],
    }
    assert torch.isfinite(wrapped.compute_loss(batch)["loss"])
    # wrapper 是 nn.Module：内层参数纳入 parameters()（训练循环可见）
    inner_ids = {id(p) for p in model.parameters()}
    wrapped_ids = {id(p) for p in wrapped.parameters()}
    assert inner_ids <= wrapped_ids
