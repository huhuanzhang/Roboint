"""Robo_int π0.5 server fork 与 reference 实现的 parity 测试。

三层：
1. 原文件未被修改（md5 基准，复制时刻 2026-09-10）；
2. wire 级 metadata parity：reference(8011) vs fork(8012)，双 server 在线时跑；
3. 数值级 parity：同一进程内加载两份实现，同 observation + 同 noise 采样，
   动作必须一致（绕开 wire 协议不传 noise 的限制）。需 molmoact2 lerobot
   （.venv-smolvla 解释器）+ GPU，用 ROBO_INT_PI05_PARITY=1 显式开启。
"""

from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SERVER = Path(
    os.environ.get("ROBO_INT_PI05_REFERENCE_SERVER",
                   str(ROOT.parent / "smolvla_piper_control_gui" / "smolvla_policy_server.py"))
).expanduser()
FORK_SERVER = ROOT / "robo_int" / "models" / "pi05" / "server.py"
CHECKPOINT = Path(
    os.environ.get("ROBO_INT_PI05_CHECKPOINT", str(ROOT / "checkpoints" / "pi05"))
).expanduser()
# md5 基准；reference 一旦被改动此测试报警。
# 2026-09-10 初版 73dbb54d198ff3fb9a563a57b94de2c4；
# 2026-09-11 起漂移（有意）：RTC 接线 + 注意力捕获（PI05AttentionCapture
# 调用段，smolvla_piper_control_gui 授权功能工作），基准更新如下——
# 守卫语义不变：仍捕捉此后任何未记录的改动。
REFERENCE_MD5 = "de337c368d2d2f5b2d6e3825e67722bf"
REFERENCE_PORT, FORK_PORT = 8011, 8012


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _client(port: int):
    if os.environ.get("OPENPI_CLIENT_SRC"):
        import sys

        sys.path.insert(0, os.path.expanduser(os.environ["OPENPI_CLIENT_SRC"]))
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)


def test_reference_server_untouched():
    """reference 实现保持原样（fork 的 parity 基准不能漂移）。"""
    if not REFERENCE_SERVER.is_file() and not os.environ.get("ROBO_INT_PI05_REFERENCE_SERVER"):
        pytest.skip("本机无 reference server；设置 ROBO_INT_PI05_REFERENCE_SERVER 后验证")
    assert REFERENCE_SERVER.is_file(), f"reference server 不存在：{REFERENCE_SERVER}"
    assert _md5(REFERENCE_SERVER) == REFERENCE_MD5, "reference smolvla_policy_server.py 被修改了！"


@pytest.mark.skipif(
    os.environ.get("ROBO_INT_PI05_WIRE_PARITY") != "1"
    or not (_port_open(REFERENCE_PORT) and _port_open(FORK_PORT)),
    reason="需 ROBO_INT_PI05_WIRE_PARITY=1 且 reference(8011)/fork(8012) 同时在线",
)
def test_wire_metadata_parity():
    ref_meta = _client(REFERENCE_PORT).get_server_metadata()
    fork_meta = _client(FORK_PORT).get_server_metadata()
    stable_keys = [
        "policy_type", "robot_layout", "policy_io_mode", "state_dim", "action_dim",
        "controlled_arm", "action_semantics", "actions_per_inference",
        "full_action_horizon", "rtc_supported", "rtc_enabled", "rtc_mode",
        "rtc_execution_horizon", "rtc_max_guidance_weight",
        "rtc_prefix_attention_schedule", "required_policy_cameras",
        "checkpoint", "normalization",
    ]
    for key in stable_keys:
        assert fork_meta.get(key) == ref_meta.get(key), f"metadata 不一致: {key}"
    # fork 独有字段：向后兼容的可选特征通道
    assert fork_meta.get("feature_available") is True
    assert fork_meta.get("feature_name") == "vlm_prefix_mean"
    assert int(fork_meta.get("feature_dim", 0)) > 0


@pytest.mark.skipif(
    os.environ.get("ROBO_INT_PI05_PARITY") != "1",
    reason="数值级 parity 需显式 ROBO_INT_PI05_PARITY=1（双模型加载，GPU，约 3 分钟）",
)
def test_numerical_action_parity_in_process():
    """同进程加载 reference 与 fork 实现，同 obs + 同 noise -> 动作一致。

    必须在能 import molmoact2 lerobot 的解释器下运行
    （.venv-smolvla/bin/python -m pytest tests/test_pi05_server_parity.py）。
    """
    import importlib.util

    import numpy as np
    import torch

    def load_module(name: str, path: Path):
        import sys

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses 按 __module__ 反查类型，必须先注册
        spec.loader.exec_module(module)
        return module

    ref_mod = load_module("ref_pi05_server", REFERENCE_SERVER)
    fork_mod = load_module("fork_pi05_server", FORK_SERVER)

    common = dict(device="cuda", actions_per_inference=8, policy_io_mode="dual-arm-14d")
    ref_policy = ref_mod.SmolVLAPiperPolicy(CHECKPOINT, rtc_enabled=False, **common)
    fork_policy = fork_mod.SmolVLAPiperPolicy(CHECKPOINT, rtc_enabled=False, **common)

    observation = {
        "active_state": np.linspace(-0.4, 0.5, 14, dtype=np.float32),
        "images": {
            name: (np.arange(480 * 640 * 3, dtype=np.uint8) % 251).reshape(480, 640, 3)
            for name in ref_policy.expected_image_keys
        },
        "prompt": "fold the clothes neatly",
        "actions_per_inference": 8,
        "request_id": "parity",
    }
    torch.manual_seed(20260910)
    noise = ref_policy.policy.model.sample_noise(
        (1, ref_policy.config.chunk_size, ref_policy.config.max_action_dim), "cuda"
    )

    ref_out = ref_policy.infer(dict(observation), ref_policy.new_session(), noise=noise.clone())
    fork_out = fork_policy.infer(dict(observation), fork_policy.new_session(), noise=noise.clone())

    ref_actions = np.asarray(ref_out["full_actions"])
    fork_actions = np.asarray(fork_out["full_actions"])
    assert ref_actions.shape == fork_actions.shape == (50, 14)
    np.testing.assert_allclose(fork_actions, ref_actions, atol=1e-4, rtol=0)
    max_diff = float(np.abs(fork_actions - ref_actions).max())
    print(f"数值 parity：max_abs_diff={max_diff:.3e}（同 checkpoint 同 noise 同代码路径）")
