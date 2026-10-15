"""真实 Piper episode + 真实 π0.5 feature 过 SARM2 奖励管线（需 fork server 8012）。

链路：Episode -> [vlm_prefix_mean(2048); state(14)] = 2062D -> SARM2 ->
value / stage / reward -> RLT replay（只灌不训）-> 来源可追溯的 relabel 副本。

如实结论：SARM2 未训练（无示教数据），value/stage 是真实前向输出但无语义，
reward ≈ 0 常数是预期——本测试只验证管线，不声称 reward 可用。
"""

from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FORK_PORT = 8012
EPISODE_DIR = ROOT / "data" / "piper_dryrun"
RELABEL_DIR = ROOT / "data" / "sarm2_functional_test"
STATE_DIM = 14
FEATURE_DIM = 2048


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


pytestmark = pytest.mark.skipif(
    not (_port_open(FORK_PORT) and EPISODE_DIR.exists()),
    reason="需要 fork server(8012) 在线且 data/piper_dryrun 存在",
)


@pytest.fixture(scope="module")
def episode_and_inputs():
    from robo_int.data.episodes import EpisodeStore
    from robo_int.models.pi05_server import Pi05ServerPolicy

    episode = EpisodeStore(EPISODE_DIR).load("ep_000001")
    policy = Pi05ServerPolicy(port=FORK_PORT, prompt="fold the clothes neatly")
    policy.connect()
    frames = episode.states.shape[0]
    # 逐帧真实 π0.5 特征（VLM prefix 均值池化）+ 本体状态拼接 -> (F, 2062)
    features = []
    for t in range(frames):
        obs = {
            "state": episode.states[t].astype(np.float32),
            "images": {cam: episode.images[cam][t]
                       for cam in ("head", "left_wrist", "right_wrist")},
        }
        features.append(policy.extract_features(obs).numpy())
    inputs = np.concatenate([np.stack(features), episode.states.astype(np.float32)], axis=1)
    return episode, inputs, policy


def test_real_feature_state_concat(episode_and_inputs):
    _, inputs, policy = episode_and_inputs
    assert inputs.shape == (17, FEATURE_DIM + STATE_DIM)
    assert np.isfinite(inputs).all()
    assert float(np.abs(inputs[:, :FEATURE_DIM]).std()) > 0  # 特征逐帧有变化
    # 注意：vlm_prefix_mean 不含 robot state——state 由拼接补足（2062 = 2048 + 14）


def test_sarm2_real_episode_outputs(episode_and_inputs):
    from robo_int.algos.sarm2.reward_provider import Sarm2RewardProvider
    from robo_int.algos.sarm2.sarm2 import SARM2

    episode, inputs, _ = episode_and_inputs
    algo = SARM2(
        primitives=["null", "approach", "grasp", "transport", "release", "retreat"],
        groups=["null", "approach or retreat", "grasp or release", "transport"],
        state_dim=FEATURE_DIM + STATE_DIM,
        num_tasks=1, d_model=32, stage_layers=1, value_layers=1, n_heads=2,
        n_obs_steps=4, frame_gap=1, num_experts=4, expert_hidden=16, gate_hidden=16,
        seed=0,
    )
    provider = Sarm2RewardProvider(algo)
    out = provider.compute_episode(inputs)

    F = inputs.shape[0]
    assert out["reward"].shape == (F - 1,)
    assert out["value"].shape == (F,)
    assert out["stage"].shape == (F, 6)
    # 头部窗口不足为 NaN（如实），之后有限
    assert np.isnan(out["value"][:3]).all()
    valid = out["value"][4:]
    assert np.isfinite(valid).all() and ((valid > 0) & (valid < 1)).all()
    assert np.isfinite(out["reward"][3:]).all()
    # 未训练：reward 近常数（预期，不伪造变化）
    assert float(np.abs(out["reward"][3:]).max()) < 0.2


def test_reward_into_rlt_replay(episode_and_inputs):
    """SARM2 reward -> transition -> RLT replay（只灌不训）。"""
    from robo_int.algos.rlt import RLT
    from robo_int.algos.sarm2.reward_provider import Sarm2RewardProvider
    from robo_int.algos.sarm2.sarm2 import SARM2

    episode, inputs, policy = episode_and_inputs
    algo = SARM2(
        primitives=["null", "approach", "grasp", "transport", "release", "retreat"],
        groups=["null", "approach or retreat", "grasp or release", "transport"],
        state_dim=FEATURE_DIM + STATE_DIM, num_tasks=1,
        d_model=32, stage_layers=1, value_layers=1, n_heads=2,
        n_obs_steps=4, frame_gap=1, num_experts=4, expert_hidden=16, gate_hidden=16, seed=0,
    )
    provider = Sarm2RewardProvider(algo)
    out = provider.compute_episode(inputs)

    # NaN 头帧不进回放（NaN 会污染后续 update）：只灌窗口就绪后的 transition
    valid_t = np.where(np.isfinite(out["reward"]))[0]
    assert len(valid_t) > 0
    rlt = RLT(policy, action_limit=0.15, token_dim=64, hidden_dim=32,
              batch_size=4, start_steps=4, utd=1, seed=0)
    for t in valid_t:
        rlt.observe(
            episode.states[t], episode.actions[t], float(out["reward"][t]),
            episode.states[t + 1], 0.0,
        )
    batch = rlt.buffer.sample(min(8, len(rlt.buffer)))
    rewards = np.asarray(batch["reward"], dtype=np.float32)
    assert np.isfinite(rewards).all()
    # 不是占位 0：未训练 SARM2 的输出是真实前向的小幅非零波动（或近零），来源是模型
    provenance = provider.provenance()
    assert provenance["reward_source"] == "sarm2"
    assert provenance["reward_is_placeholder"] is False
    assert provenance["reward_model_version"] == "untrained"


def test_relabeled_copy_with_provenance(episode_and_inputs):
    """来源可追溯的 relabel 副本：不动物理原数据，provenance 写进 meta.extra。"""
    from robo_int.algos.sarm2.reward_provider import Sarm2RewardProvider
    from robo_int.algos.sarm2.sarm2 import SARM2
    from robo_int.data.episodes import Episode, EpisodeStore

    episode, inputs, _ = episode_and_inputs
    original_md5 = hashlib.md5((EPISODE_DIR / "episodes" / "ep_000001.npz").read_bytes()).hexdigest()

    algo = SARM2(
        primitives=["null", "approach", "grasp", "transport", "release", "retreat"],
        groups=["null", "approach or retreat", "grasp or release", "transport"],
        state_dim=FEATURE_DIM + STATE_DIM, num_tasks=1,
        d_model=32, stage_layers=1, value_layers=1, n_heads=2,
        n_obs_steps=4, frame_gap=1, num_experts=4, expert_hidden=16, gate_hidden=16, seed=0,
    )
    provider = Sarm2RewardProvider(algo)
    out = provider.compute_episode(inputs)

    import dataclasses

    relabeled = Episode(
        meta=dataclasses.replace(episode.meta, episode_id="ep_000001"),
        states=episode.states, actions=episode.actions,
        rewards=out["reward"],  # 头部 NaN 保留（如实）
        images=episode.images, policy_actions=episode.policy_actions,
        timestamps=episode.timestamps,
    )
    relabeled.meta.extra.update(
        provider.provenance(),
        stage_source="sarm2_stage_estimator_untrained",
        not_for_training=True,       # 未训练模型 + dry-run episode：链路验证用
        functional_test=True,
        reward_semantics="progress_delta(steps_to_go)",
    )
    store = EpisodeStore(RELABEL_DIR)
    path = store.save(relabeled)

    reloaded_meta = store.load_index()[0]
    assert reloaded_meta.extra["reward_source"] == "sarm2"
    assert reloaded_meta.extra["reward_model_version"] == "untrained"
    assert reloaded_meta.extra["not_for_training"] is True
    reloaded = store.load("ep_000001")
    np.testing.assert_allclose(reloaded.rewards, out["reward"], equal_nan=True)
    # 原始数据不动
    assert hashlib.md5((EPISODE_DIR / "episodes" / "ep_000001.npz").read_bytes()).hexdigest() == original_md5
    assert path.is_file()
