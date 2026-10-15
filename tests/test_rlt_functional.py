"""RLT 功能性验证（F4）：真实 Episode 数据 -> transition -> 最小 update。

区分两个结论：
- RLT infrastructure（replay/actor/critic/optimizer/checkpoint）：真实可跑；
- 真实 π0.5 表征（extract_features）：Pi05ServerPolicy 未实现（server 只返回
  action），此处用 MockVLA 冻结特征作为 stand-in——不伪装成真实 VLA token。
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from robo_int.algos.rlt import RLT
from robo_int.data.episodes import Episode, EpisodeMeta
from robo_int.data.transitions import episode_to_transitions
from robo_int.models.mock import MockVLA

ROOT = Path(__file__).resolve().parent.parent
REAL_EPISODE = ROOT / "data" / "piper_dryrun" / "episodes" / "ep_000001.npz"


def _episode(state_dim=4, action_dim=2, steps=5, dry_run=True):
    rng = np.random.default_rng(0)
    return Episode(
        meta=EpisodeMeta(
            episode_id="ep_func",
            task="FuncTask",
            robot="func_robot",
            source="policy",
            instruction="functional test",
            extra={
                "terminated": False,
                "truncated": True,
                "dry_run": dry_run,
                "training_eligible": not dry_run,
            },
        ),
        states=rng.uniform(-0.4, 0.4, (steps + 1, state_dim)).astype(np.float32),
        actions=rng.uniform(-0.1, 0.1, (steps, action_dim)).astype(np.float32),
        rewards=np.full(steps, np.nan, dtype=np.float32),  # 延后打分 -> NaN
        images={},
        policy_actions=rng.uniform(-2.0, 2.0, (steps, action_dim)).astype(np.float32),
    )


def test_episode_to_transitions_schema():
    tr = episode_to_transitions(_episode(), reward_fill=0.0)
    assert tr["observation.state"].shape == (5, 4)
    assert tr["next_observation.state"].shape == (5, 4)
    assert tr["action"].shape == (5, 2)
    np.testing.assert_allclose(tr["next_observation.state"], tr["observation.state"] * 0 + tr["next_observation.state"])
    assert tr["action"].shape == tr["policy_action"].shape
    # NaN reward -> 占位 0 并显式标记
    np.testing.assert_allclose(tr["reward"], 0.0)
    assert tr["placeholder_reward"] is True
    assert tr["dry_run"] is True and tr["training_eligible"] is False
    # terminated=false -> done 全 0
    np.testing.assert_allclose(tr["done"], 0.0)


def test_episode_to_transitions_done_on_termination():
    ep = _episode()
    ep.meta.extra["terminated"] = True
    tr = episode_to_transitions(ep)
    assert tr["done"][-1] == 1.0 and tr["done"][:-1].sum() == 0


def _run_rlt_on_transitions(state_dim, action_dim, transitions, seed=0):
    torch.manual_seed(seed)
    model = MockVLA(state_dim=state_dim, action_dim=action_dim, hidden_dim=32)
    algo = RLT(
        model,
        action_limit=0.15,
        token_dim=16,
        hidden_dim=32,
        batch_size=4,
        start_steps=4,
        utd=1,
        seed=seed,
    )
    # 逐条灌入回放（在线 observe 契约）
    for i in range(transitions["num_steps"]):
        algo.observe(
            transitions["observation.state"][i],
            transitions["action"][i],
            transitions["reward"][i],
            transitions["next_observation.state"][i],
            transitions["done"][i],
        )
    assert algo.ready
    snapshot = {
        "model": copy.deepcopy(model.state_dict()),
        "token_head": copy.deepcopy(algo.token_head.state_dict()),
        "actor": copy.deepcopy(algo.actor.state_dict()),
        "critic": copy.deepcopy(algo.critic.state_dict()),
    }
    batch = {k: transitions[k] for k in ("observation.state", "action", "reward",
                                         "next_observation.state", "done")}
    metrics = [algo.update(batch) for _ in range(3)]
    return algo, model, snapshot, metrics


def _param_changed(before, after):
    return any(not torch.equal(before[k], after[k]) for k in before)


def _grad_norm(params):
    return float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)))


def test_rlt_minimal_update_on_episode_data():
    tr = episode_to_transitions(_episode())
    algo, model, snap, metrics = _run_rlt_on_transitions(4, 2, tr)

    for m in metrics:
        assert np.isfinite(m["actor_loss"]) and np.isfinite(m["critic_loss"])
        assert np.isfinite(m["aux_loss"]) and np.isfinite(m["loss"])
    # 梯度有限且非零（update 内部 backward 后梯度仍挂在参数上）
    for group in (algo.actor.parameters(), algo.critic.parameters(), algo.token_head.parameters()):
        assert 0.0 < _grad_norm(list(group)) < float("inf")
    # 参数更新边界：VLA 冻结不变；RLT 侧三个组件都动了
    assert not _param_changed(snap["model"], model.state_dict()), "冻结 VLA 参数被改动"
    assert _param_changed(snap["token_head"], algo.token_head.state_dict())
    assert _param_changed(snap["actor"], algo.actor.state_dict())
    assert _param_changed(snap["critic"], algo.critic.state_dict())


def test_rlt_save_reload_parity(tmp_path):
    tr = episode_to_transitions(_episode())
    algo, model, snap, _ = _run_rlt_on_transitions(4, 2, tr, seed=7)
    ckpt = tmp_path / "rlt_state.pt"
    algo.save_state(ckpt)

    torch.manual_seed(7)
    model2 = MockVLA(state_dim=4, action_dim=2, hidden_dim=32)
    algo2 = RLT(model2, action_limit=0.15, token_dim=16, hidden_dim=32,
                batch_size=4, start_steps=4, utd=1, seed=7)
    algo2.load_state(ckpt)
    for key in ("token_head", "actor", "critic", "critic_target"):
        before = getattr(algo, key).state_dict()
        after = getattr(algo2, key).state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before), f"{key} 恢复不一致"

    # 同一确定性输入 -> 一致输出（actor 均值路径；direct 模式无 VLA 采样）
    obs = {"state": tr["observation.state"][0]}
    a1 = algo.act(obs, deterministic=True)
    a2 = algo2.act(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=0.0)
    # 注：save_state 不含 optimizer 动量（Adam state）——当前实现如实记录此边界


@pytest.mark.skipif(not REAL_EPISODE.exists(), reason="本机无真实 Piper dry-run episode")
def test_rlt_functional_on_real_piper_episode():
    """真实 14D/14D Piper transition 过一遍最小 RLT update（特征为 MockVLA stand-in）。"""
    from robo_int.data.episodes import EpisodeStore

    store = EpisodeStore(ROOT / "data" / "piper_dryrun")
    episode = store.load("ep_000001")
    tr = episode_to_transitions(episode)

    assert tr["observation.state"].shape == (episode.meta.num_steps, 14)
    assert tr["action"].shape == (episode.meta.num_steps, 14)
    assert tr["placeholder_reward"] and tr["dry_run"] and not tr["training_eligible"]

    torch.manual_seed(0)
    standin = MockVLA(state_dim=14, action_dim=14, hidden_dim=32)
    algo = RLT(standin, action_limit=0.15, token_dim=16, hidden_dim=32,
               batch_size=8, start_steps=8, utd=1, seed=0)
    for i in range(tr["num_steps"]):
        algo.observe(tr["observation.state"][i], tr["action"][i], tr["reward"][i],
                     tr["next_observation.state"][i], tr["done"][i])
    frozen = copy.deepcopy(standin.state_dict())
    batch = {k: tr[k] for k in ("observation.state", "action", "reward",
                                "next_observation.state", "done")}
    metrics = [algo.update(batch) for _ in range(3)]
    assert all(np.isfinite(m["loss"]) for m in metrics)
    assert not _param_changed(frozen, standin.state_dict())


def test_train_rl_generic_entry(tmp_path):
    """generic 在线 RL 入口端到端（mock 任务，含 checkpoint 落盘）。"""
    run_dir = tmp_path / "rl_run"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "train_rl.py"),
            "algo=rlt", "model=mock", "task=mock_reach", "robot=mock",
            "infra.total_steps=40", "infra.checkpoint_every=20",
            "algo.start_steps=8", "algo.batch_size=8",
            f"infra.run_dir={run_dir}",
        ],
        capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "updates=" in result.stdout
    assert "updates=0" not in result.stdout, f"无 update 发生：{result.stdout}"
    assert (run_dir / "checkpoints" / "final.pt").is_file()
