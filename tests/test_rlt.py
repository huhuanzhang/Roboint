"""RLT（RL Token）测试：核心断言是「冻结 VLA 特征 + RL token + SAC」
能从零（无示教）在线学会 MockReachTask——这是 README 复现计划第一步。
"""

import numpy as np
import pytest
import torch

from robo_int.algos.rlt import RLT, RLTokenHead
from robo_int.models.mock import MockVLA
from robo_int.tasks.mock_reach import MockReachTask


def make_algo(seed: int = 0, **overrides) -> RLT:
    torch.manual_seed(seed)
    model = MockVLA(state_dim=4, action_dim=2, hidden_dim=128)
    kwargs = dict(action_limit=0.15, seed=seed, hidden_dim=128, batch_size=128)
    kwargs.update(overrides)
    return RLT(model, **kwargs)


def run_training(algo: RLT, task: MockReachTask, total_steps: int, seed: int = 100):
    """在线交互训练循环（rollout.py 是面向策略函数的，RLT 用 act/observe 接口）。"""
    rng = np.random.default_rng(seed)
    episode, steps, returns = 0, 0, []
    while steps < total_steps:
        obs = task.reset(seed=int(rng.integers(1 << 31)))
        ep_ret = 0.0
        for _ in range(task.max_episode_steps):
            action = algo.act(obs)
            res = task.step(action)
            algo.observe(obs["state"], action, res.reward, res.observation["state"], res.terminated)
            obs = res.observation
            ep_ret += res.reward
            steps += 1
            if algo.ready:
                algo.update()
            if res.terminated or res.truncated:
                break
        returns.append(ep_ret)
        episode += 1
    return returns


def evaluate(algo: RLT, task: MockReachTask, num_episodes: int = 10, seed: int = 999):
    successes, returns = 0, []
    for i in range(num_episodes):
        obs = task.reset(seed=seed + i)
        ep_ret = 0.0
        success = False
        for _ in range(task.max_episode_steps):
            action = algo.act(obs, deterministic=True)
            res = task.step(action)
            obs = res.observation
            ep_ret += res.reward
            if res.terminated:
                success = True
                break
        successes += success
        returns.append(ep_ret)
    return successes / num_episodes, float(np.mean(returns))


class TestRLTokenHead:
    def test_shapes(self):
        head = RLTokenHead(feature_dim=128, token_dim=64)
        feats = torch.randn(8, 128)
        token = head.encode(feats)
        assert token.shape == (8, 64)
        assert head.reconstruction_loss(token, feats).dim() == 0

    def test_reconstruction_learns(self):
        """token 重建辅助损失可训练（token 不丢特征信息）。"""
        torch.manual_seed(0)
        head = RLTokenHead(feature_dim=32, token_dim=16)
        data = torch.randn(256, 32)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        first = last = None
        for step in range(200):
            loss = head.reconstruction_loss(head.encode(data), data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        assert last < first * 0.5, f"重建损失应显著下降: {first:.4f} -> {last:.4f}"


class TestRLT:
    def test_model_frozen(self):
        """RLT 构造后基模参数全部冻结（论文核心设定）。"""
        algo = make_algo()
        assert all(not p.requires_grad for p in algo.model.parameters())

    def test_act_shapes_and_limit(self):
        algo = make_algo()
        task = MockReachTask()
        obs = task.reset(seed=0)
        for _ in range(5):
            a = algo.act(obs)  # 预热期随机
            algo.observe(obs["state"], a, -1.0, obs["state"], False)
            obs = {"state": obs["state"] + 0.01}
        assert a.shape == (2,)
        assert np.all(np.abs(a) <= algo.action_limit + 1e-6)

    def test_update_returns_metrics(self):
        algo = make_algo(batch_size=8, start_steps=8)
        task = MockReachTask()
        obs = task.reset(seed=0)
        for _ in range(10):
            res = task.step(algo.act(obs))
            algo.observe(obs["state"], np.zeros(2, np.float32), res.reward, res.observation["state"], res.terminated)
            obs = res.observation
        m = algo.update()
        assert "loss" in m and "aux_loss" in m and "alpha" in m
        assert algo._total_updates == algo.utd

    def test_residual_mode_act(self):
        """residual 模式：动作 = VLA 原动作 + actor 残差，仍受限幅。"""
        algo = make_algo(action_mode="residual")
        task = MockReachTask()
        obs = task.reset(seed=0)
        a = algo.act(obs)
        assert a.shape == (2,)
        assert np.all(np.abs(a) <= algo.action_limit + 1e-6)

    def test_residual_critic_input_dim(self):
        """residual 模式 critic 动作侧输入 = [executed, base]（2A）；direct 保持 A 不变。"""
        token_dim = make_algo().token_head.encoder[0].out_features
        algo_r = make_algo(action_mode="residual")
        algo_d = make_algo()
        in_r = algo_r.critic.qs[0][0].in_features
        in_d = algo_d.critic.qs[0][0].in_features
        assert in_r == token_dim + 2 * algo_r.action_dim
        assert in_d == token_dim + algo_d.action_dim, "direct 模式 critic 输入维度不应改变"

    def test_residual_critic_uses_base(self):
        """同 (token, executed) 下不同 base -> 不同 Q 值（base 真进入 critic 输入）。"""
        algo = make_algo(action_mode="residual")
        token_dim = algo.token_head.encoder[0].out_features
        tok = torch.randn(4, token_dim)
        a = torch.randn(4, algo.action_dim)
        b0 = torch.zeros(4, algo.action_dim)
        b1 = torch.ones(4, algo.action_dim)
        q0 = algo.critic(tok, torch.cat([a, b0], dim=-1))
        q1 = algo.critic(tok, torch.cat([a, b1], dim=-1))
        assert not torch.allclose(q0[0], q1[0]), "critic 应把 base 当输入（否则 Q(token,a) 系统性模糊）"

    def test_residual_base_flows_to_replay(self):
        """act() 采样的 base 经 observe() 缺省参数落入回放（Markov 化闭环）。"""
        algo = make_algo(action_mode="residual", start_steps=4)
        task = MockReachTask()
        obs = task.reset(seed=0)
        base_seen = []
        for _ in range(6):
            a = algo.act(obs)
            assert algo.last_base is not None and algo.last_base.shape == (2,)
            base_seen.append(algo.last_base.copy())
            algo.observe(obs["state"], a, -1.0, obs["state"], False)  # 缺省取 last_base
            obs = {"state": obs["state"] + 0.01}
        batch = algo.buffer.sample(6)
        assert "base_action" in batch and batch["base_action"].shape == (6, 2)
        # 前几条（含预热步）base 已如实记录，非全零
        assert np.abs(batch["base_action"]).sum() > 0

    def test_direct_mode_replay_base_zero(self):
        """direct 模式回放 base 恒零占位（schema 统一，critic 不使用）。"""
        algo = make_algo(start_steps=4)
        task = MockReachTask()
        obs = task.reset(seed=0)
        for _ in range(6):
            a = algo.act(obs)
            algo.observe(obs["state"], a, -1.0, obs["state"], False)
            obs = {"state": obs["state"] + 0.01}
        assert algo.last_base is None
        assert np.abs(algo.buffer.sample(6)["base_action"]).sum() == 0.0

    def test_residual_mode_update_runs(self):
        """residual 模式完整 SAC 步可跑（critic 三处输入均含 base）。"""
        algo = make_algo(action_mode="residual", batch_size=8, start_steps=8)
        task = MockReachTask()
        obs = task.reset(seed=0)
        for _ in range(10):
            a = algo.act(obs)
            res = task.step(a)
            algo.observe(obs["state"], a, res.reward, res.observation["state"], res.terminated)
            obs = res.observation
        m = algo.update()
        assert np.isfinite(m["loss"]) and np.isfinite(m["q_mean"])

    @pytest.mark.slow
    def test_learns_mock_reach_from_scratch(self):
        """核心：无示教数据，冻结 VLA 特征 + RL token + SAC 在线学到 >=80% 成功率。"""
        algo = make_algo(seed=3)
        task = MockReachTask()
        run_training(algo, task, total_steps=3500)
        rate, ret = evaluate(algo, task)
        assert rate >= 0.8, f"成功率 {rate:.0%}（return={ret:.2f}）应 >= 80%"

    @pytest.mark.slow
    def test_improves_over_training(self):
        """训练中 return 单调改善（早段 vs 末段）。"""
        algo = make_algo(seed=5)
        task = MockReachTask()
        returns = run_training(algo, task, total_steps=2500)
        early = float(np.mean(returns[:5]))
        late = float(np.mean(returns[-5:]))
        assert late > early, f"末段 return ({late:.2f}) 应优于早段 ({early:.2f})"

    def test_save_load_state_roundtrip(self, tmp_path):
        algo = make_algo()
        task = MockReachTask()
        obs = task.reset(seed=0)
        for _ in range(10):
            algo.observe(obs["state"], np.zeros(2, np.float32), -1.0, obs["state"], False)
            obs = {"state": obs["state"] + 0.02}
        algo.update = lambda batch=None: {}  # 跳过训练
        algo.save_state(tmp_path / "rlt_state.pt")
        algo2 = make_algo(seed=99)  # 不同初始化
        algo2.load_state(tmp_path / "rlt_state.pt")
        for k in ("token_head", "actor", "critic"):
            a = dict(getattr(algo, k).state_dict())
            b = dict(getattr(algo2, k).state_dict())
            for key in a:
                assert torch.allclose(a[key], b[key]), f"{k}.{key} 加载后不一致"
