"""Piper / π0.5-server adapter 的离线单元测试（不依赖真机与 policy server）。

安全链测试需要 lerobot-piper-2（复用其 scale/clamp 函数），无该依赖时跳过；
policy 测试用假 WebSocket client 验证观测构造与返回解析。
"""

import numpy as np
import pytest

from robo_int.models.pi05_server import Pi05ServerPolicy
from robo_int.robots.piper_safety import DUAL_ARM_GRIPPER_INDICES, PiperDualArmSafetyChain


@pytest.fixture
def safety_chain():
    pytest.importorskip("piper_scripts.utils.openpi_piper_single_arm_can0_inference")
    # 与 GUI config.json profile 相同的安全参数
    return PiperDualArmSafetyChain(
        action_scale=1.0,
        max_abs_delta=0.01,
        gripper_max_abs_delta=0.003,
        action_smoothing_alpha=1.0,
        gripper_smoothing_alpha=1.0,
    )


def test_safety_chain_clamps_joint_and_gripper(safety_chain):
    state = np.zeros(14, dtype=np.float32)
    state[6] = state[13] = 0.05
    # 模型输出远离当前状态：关节 +1 rad、夹爪全开
    model_action = np.ones(14, dtype=np.float32) * 1.0
    command = safety_chain(model_action, state)
    joint_idx = [i for i in range(14) if i not in DUAL_ARM_GRIPPER_INDICES]
    assert np.all(np.abs(command[joint_idx] - state[joint_idx]) <= 0.01 + 1e-6)
    assert np.all(np.abs(command[[6, 13]] - state[[6, 13]]) <= 0.003 + 1e-6)


def test_safety_chain_passes_small_actions(safety_chain):
    state = np.full(14, 0.1, dtype=np.float32)
    delta = np.full(14, 0.002, dtype=np.float32)  # 关节/夹爪均在限幅内
    command = safety_chain(state + delta, state)
    np.testing.assert_allclose(command, state + delta, atol=1e-6)


def test_safety_chain_rejects_nan(safety_chain):
    bad = np.zeros(14, dtype=np.float32)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        safety_chain(bad, np.zeros(14, dtype=np.float32))


class _FakePolicyClient:
    """伪造 openpi-client：记录请求并返回 (8, 14) 动作块。"""

    def __init__(self):
        self.requests = []
        self.response = {
            "actions": np.arange(8 * 14, dtype=np.float32).reshape(8, 14) / 100.0,
            "policy_timing": {"infer_ms": 12.5},
        }

    def infer(self, request):
        self.requests.append(request)
        return dict(self.response)


def test_pi05_server_policy_request_mapping():
    policy = Pi05ServerPolicy(prompt="fold the clothes neatly", actions_per_inference=8)
    policy.required_cameras = ("head", "left_wrist", "right_wrist")
    policy._client = _FakePolicyClient()  # 跳过真实连接

    state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in policy.required_cameras}
    chunk = policy.predict_actions({"state": state, "images": images})

    assert chunk.shape == (8, 14)
    assert policy.last_infer_ms == 12.5
    request = policy._client.requests[0]
    assert np.array_equal(request["active_state"], state)
    assert request["prompt"] == "fold the clothes neatly"
    assert request["actions_per_inference"] == 8
    # 只发送 server 要求的相机，且顺序与 required_cameras 一致
    assert tuple(request["images"].keys()) == policy.required_cameras


def test_pi05_server_policy_validates_observation():
    policy = Pi05ServerPolicy(actions_per_inference=8)
    policy.required_cameras = ("head", "left_wrist", "right_wrist")
    policy._client = _FakePolicyClient()

    with pytest.raises(ValueError, match="state"):
        policy.predict_actions({"state": np.zeros(7, dtype=np.float32), "images": {}})
    with pytest.raises(ValueError, match="相机"):
        policy.predict_actions(
            {"state": np.zeros(14, dtype=np.float32), "images": {"head": np.zeros((480, 640, 3), np.uint8)}}
        )


def test_pi05_connect_uses_installed_client(monkeypatch):
    """openpi_client 已安装时，无需额外提供源码 checkout。"""
    import sys
    from types import ModuleType, SimpleNamespace

    monkeypatch.delenv("OPENPI_CLIENT_SRC", raising=False)

    class InstalledClient:
        def __init__(self, host, port):
            self.host, self.port = host, port

        def get_server_metadata(self):
            return {"required_policy_cameras": ["head"]}

    module = ModuleType("openpi_client")
    module.websocket_client_policy = SimpleNamespace(WebsocketClientPolicy=InstalledClient)
    monkeypatch.setitem(sys.modules, "openpi_client", module)
    policy = Pi05ServerPolicy()

    assert policy.openpi_client_src is None
    assert policy.connect()["required_policy_cameras"] == ["head"]
    assert policy.required_cameras == ("head",)
