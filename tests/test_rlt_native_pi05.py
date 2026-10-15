"""真实 π0.5 表征 -> RLT 的功能测试（F5，需 Robo_int fork server 在线）。

前置：fork server（robo_int/models/pi05/server.py）运行于 127.0.0.1:8012，
加载 150000 checkpoint。skipif 端口不通。数据用已录制的真实 Piper dry-run
episode（functional_test 用途，reward 占位、不作训练数据）。
"""

from __future__ import annotations

import copy
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
FORK_PORT = 8012
EPISODE_DIR = ROOT / "data" / "piper_dryrun"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


pytestmark = pytest.mark.skipif(
    not _port_open(FORK_PORT), reason=f"Robo_int fork server 不在线（127.0.0.1:{FORK_PORT}）"
)


@pytest.fixture(scope="module")
def policy():
    import sys

    sys.path.insert(0, str(ROOT))
    from robo_int.models.pi05_server import Pi05ServerPolicy

    p = Pi05ServerPolicy(port=FORK_PORT, prompt="fold the clothes neatly")
    p.connect()
    return p


@pytest.fixture(scope="module")
def frames():
    """真实 Piper episode 的逐帧观测（state + 三路相机）。"""
    from robo_int.data.episodes import EpisodeStore

    store = EpisodeStore(EPISODE_DIR)
    episode = store.load("ep_000001")
    n = episode.meta.num_steps
    return [
        {
            "state": episode.states[t].astype(np.float32),
            "images": {cam: episode.images[cam][t] for cam in ("head", "left_wrist", "right_wrist")},
        }
        for t in range(n + 1)
    ], episode


def test_feature_metadata(policy):
    meta = policy.server_metadata
    assert meta["feature_available"] is True
    assert meta["feature_name"] == "vlm_prefix_mean"
    assert policy.feature_dim == int(meta["feature_dim"]) and policy.feature_dim > 0
    assert meta["feature_dtype"] == "float32"


def test_one_forward_actions_and_features(policy, frames):
    obs, _ = frames
    actions, feature = policy.predict_actions_and_features(obs[0])
    assert actions.shape == (8, 14) and np.isfinite(actions).all()
    assert feature.shape == (policy.feature_dim,) and np.isfinite(feature).all()

    # 特征是 VLM prefix（flow 采样之前）的确定性函数：同 obs 两次请求，
    # 特征一致而动作（flow 噪声随机）不同——证明一次 forward 同源、非事后拼凑
    actions2, feature2 = policy.predict_actions_and_features(obs[0])
    np.testing.assert_allclose(feature2, feature, atol=1e-4)
    assert not np.allclose(actions2, actions), "flow 采样应有随机性（同 noise 种子除外）"


def test_feature_varies_across_observations(policy, frames):
    obs, _ = frames
    feats = np.stack([policy.extract_features(o).numpy() for o in obs[:6]])
    assert feats.shape == (6, policy.feature_dim)
    assert float(feats.std(axis=0).mean()) > 1e-4, "不同观测的特征应有变化"


def test_rlt_with_real_pi05_features(policy, frames):
    """真实 Piper observation -> π0.5 feature -> RLTokenHead -> actor/critic -> update。"""
    from robo_int.algos.rlt import RLT

    obs, episode = frames
    torch.manual_seed(0)
    algo = RLT(
        policy,
        action_limit=0.15,
        token_dim=64,
        hidden_dim=64,
        batch_size=8,
        start_steps=8,
        utd=1,
        seed=0,
    )
    assert algo.token_head.encoder[0].in_features == policy.feature_dim

    snapshot = {
        "token_head": copy.deepcopy(algo.token_head.state_dict()),
        "actor": copy.deepcopy(algo.actor.state_dict()),
        "critic": copy.deepcopy(algo.critic.state_dict()),
    }
    server_param_ids = {id(p) for p in policy.parameters()}
    optimizer_param_ids = {
        id(p)
        for group in algo._optimizers.values()
        for p in group.param_groups[0]["params"]
    }
    assert not (server_param_ids & optimizer_param_ids), "π0.5（client 侧无参数）不得进入优化器"

    t = episode.meta.num_steps
    rewards = np.zeros(t, dtype=np.float32)  # 占位 reward（functional test，非训练数据）
    for i in range(t):
        algo.observe(
            episode.states[i], episode.actions[i], rewards[i],
            episode.states[i + 1], 0.0,
            observation=obs[i], next_observation=obs[i + 1],
        )
    assert algo.ready and algo.buffer.features is not None
    assert algo.buffer.features.shape[-1] == policy.feature_dim

    metrics = [algo.update() for _ in range(3)]
    for m in metrics:
        assert np.isfinite(m["actor_loss"]) and np.isfinite(m["critic_loss"]) and np.isfinite(m["aux_loss"])

    # 三个组件各自都要有参数变化
    for name in ("token_head", "actor", "critic"):
        module = getattr(algo, name)
        assert any(
            not torch.equal(snapshot[name][key], module.state_dict()[key]) for key in snapshot[name]
        ), f"{name} 参数未更新"

    # 更新后 π0.5 行为不受影响（client 无参数；server 端权重只在 load 时变化）
    actions, feature = policy.predict_actions_and_features(obs[0])
    assert actions.shape == (8, 14) and feature.shape == (policy.feature_dim,)
