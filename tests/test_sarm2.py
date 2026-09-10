"""SARM2（阶段感知奖励模型）测试。

核心断言（对应 README 复现计划第一步）：
1. 阶段估计器在多任务 mock 上学到高精度，且迁移到 held-out 任务；
2. 多门 MoE 值解码器学到显著低于常数基线的 MSE，预测值沿 episode 单调下降；
3. 路由不塌缩（collapse_score 远离 1）；
4. 消融开关（single_gate / use_primitives）可用且可训练；
5. Sarm2Scorer 给完整 demo 打分高于半截 demo。
"""

import numpy as np
import pytest
import torch

from robo_int.algos.sarm2 import (
    SARM2,
    MMoEValueDecoder,
    StageEstimator,
    Sarm2Scorer,
    build_hierarchy,
    build_stage_windows,
    build_value_windows,
    collapse_score,
)
from robo_int.data.episodes import Episode, EpisodeMeta
from robo_int.tasks.mock_pick_place import (
    PRIM_TO_GROUP,
    PRIMITIVES,
    MockPickPlaceExpert,
    MockPickPlaceTask,
    SEMANTIC_GROUPS,
    collect_demo,
)

STATE_DIM, NUM_TASKS = 6, 3


def make_algo(seed: int = 0, **overrides) -> SARM2:
    kwargs = dict(
        primitives=PRIMITIVES,
        groups=SEMANTIC_GROUPS,
        state_dim=STATE_DIM,
        num_tasks=NUM_TASKS,
        seed=seed,
    )
    kwargs.update(overrides)
    return SARM2(**kwargs)


def collect_demos(task_ids, n_per_task, seed0):
    demos = []
    for tid in task_ids:
        for i in range(n_per_task):
            task = MockPickPlaceTask(task_id=tid)
            demos.append(collect_demo(task, MockPickPlaceExpert(), seed=seed0 + tid * 1000 + i))
    return demos


def batch_iter(arrs, batch_size, rng):
    M = len(arrs["state_seq"])
    order = rng.permutation(M)
    for i in range(0, M, batch_size):
        idx = order[i : i + batch_size]
        yield {k: v[idx] for k, v in arrs.items()}


class TestBuildHierarchy:
    def test_parses_semantic_groups(self):
        """"a or b" 分组描述 -> 原语到组的索引表（官方 _build_hierarchy 同款机制）。"""
        mapping = build_hierarchy(PRIMITIVES, SEMANTIC_GROUPS)
        assert mapping.tolist() == PRIM_TO_GROUP

    def test_unmatched_falls_to_null_group(self):
        mapping = build_hierarchy(["foo", "transport"], ["null", "transport"])
        assert mapping.tolist() == [0, 1]


class TestCollapseScore:
    """S_route（论文 Eq.6 / 官方 moe_utils 同公式）：0=均衡 1=塌缩。"""

    def test_uniform_is_zero(self):
        E, k = 10, 2
        assert collapse_score(np.full(E, 1.0 / E), k) == pytest.approx(0.0, abs=1e-9)

    def test_collapsed_is_one(self):
        E, k = 10, 2
        p = np.zeros(E)
        p[:k] = 0.5  # 全部流量集中到 top-k
        assert collapse_score(p, k) == pytest.approx(1.0, abs=1e-6)

    def test_partial_is_between(self):
        E, k = 10, 2
        p = np.full(E, 0.02)
        p[0] = 0.5
        s = collapse_score(p, k)
        assert 0.0 < s < 1.0


class TestStageEstimator:
    def test_shapes(self):
        est = StageEstimator(STATE_DIM, len(PRIMITIVES), len(SEMANTIC_GROUPS), d_model=32, max_window=8)
        prim, grp = est(torch.randn(4, 6, STATE_DIM))
        assert prim.shape == (4, len(PRIMITIVES))
        assert grp.shape == (4, len(SEMANTIC_GROUPS))

    def test_learns_separable_stages(self):
        """合成可分数据：窗口首帧特征决定阶段 -> 快速学到高精度。"""
        torch.manual_seed(0)
        est = StageEstimator(4, num_primitives=3, num_groups=2, d_model=32, n_layers=1, max_window=8)
        rng = np.random.default_rng(0)
        M = 256
        labels = rng.integers(0, 3, M)
        seq = rng.normal(0, 0.1, (M, 6, 4)) + torch.nn.functional.one_hot(
            torch.as_tensor(labels), 4
        ).numpy()[:, None, :]
        opt = torch.optim.Adam(est.parameters(), lr=1e-3)
        for _ in range(150):
            logits, _ = est(torch.as_tensor(seq, dtype=torch.float32))
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = (logits.argmax(-1).numpy() == labels).mean()
        assert acc >= 0.95, f"合成可分阶段 acc={acc:.2f} 应 >= 0.95"


class TestMMoEValueDecoder:
    def _make(self, **over):
        kw = dict(
            state_dim=STATE_DIM,
            num_tasks=NUM_TASKS,
            num_primitives=len(PRIMITIVES),
            num_groups=len(SEMANTIC_GROUPS),
            d_model=32,
            n_layers=1,
        )
        kw.update(over)
        return MMoEValueDecoder(**kw)

    def _forward(self, dec, gate_idx=None, B=3, T=10):
        torch.manual_seed(1)
        state_seq = torch.randn(B, T, STATE_DIM)
        task = torch.zeros(B, dtype=torch.long)
        prim_vec = torch.randn(B, dec.prim_emb.weight.shape[1])
        lengths = torch.full((B,), T)
        g = torch.zeros(B, dtype=torch.long) if gate_idx is None else gate_idx
        return dec(state_seq, task, prim_vec, lengths, g)

    def test_output_range_and_shapes(self):
        dec = self._make()
        r, aux, info = self._forward(dec)
        assert r.shape == (3, 10)
        assert float(r.detach().min()) >= 0.0 and float(r.detach().max()) <= 1.0  # sigmoid
        assert aux.dim() == 0
        assert info["expert_load"].shape == (dec.num_experts,)

    def test_multi_gate_is_active(self):
        """不同 gate_idx 走不同门网络 -> 输出不同（多门真实生效）。"""
        dec = self._make()
        r0, _, _ = self._forward(dec, gate_idx=torch.zeros(3, dtype=torch.long))
        r1, _, _ = self._forward(dec, gate_idx=torch.ones(3, dtype=torch.long))
        assert not torch.allclose(r0, r1)

    def test_gradients_flow_to_experts_and_gates(self):
        dec = self._make()
        r, aux, _ = self._forward(dec, B=4)
        (r.mean() + aux).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.experts.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.gate_nets.parameters())

    def test_padding_positions_masked(self):
        """超长 length 的 pad 位置不参与前向逻辑（输出仍在值域，不产生 NaN）。"""
        dec = self._make()
        B, T = 2, 10
        state_seq = torch.randn(B, T, STATE_DIM)
        prim_vec = torch.randn(B, dec.prim_emb.weight.shape[1])
        lengths = torch.tensor([T, 6])  # 第二个样本有 4 个 pad 位
        r, _, _ = dec(state_seq, torch.zeros(B, dtype=torch.long), prim_vec, lengths, torch.zeros(B, dtype=torch.long))
        assert torch.isfinite(r).all()

    def test_causal_no_future_leakage(self):
        """因果性回归防线：帧 i 的输出对帧 j>i 输入的梯度必须严格为零。

        曾因 token 布局（stack 后 reshape 成 stream-major）与 causal mask
        （repeat_interleave 的 time-major 语义）错位，每帧融合输出都能看到
        全部未来帧，非最后位置的 steps_to_go 回归可直接读未来答案。
        """
        torch.manual_seed(0)
        dec = self._make()
        dec.eval()  # 关 dropout，保证梯度通路确定
        B, T = 1, 10
        state_seq = torch.randn(B, T, STATE_DIM, requires_grad=True)
        prim_vec = torch.randn(B, dec.prim_emb.weight.shape[1])
        lengths = torch.full((B,), T)  # 全有效，排除 pad 掩码干扰
        r, _, _ = dec(state_seq, torch.zeros(B, dtype=torch.long), prim_vec, lengths, torch.zeros(B, dtype=torch.long))
        for i in (2, 5, 7):
            grad = torch.autograd.grad(r[0, i], state_seq, retain_graph=True)[0]
            assert grad[0, i + 1 :].abs().max().item() <= 1e-9, (
                f"帧 {i} 的输出不应依赖未来帧（存在梯度泄漏）"
            )
            assert grad[0, : i + 1].abs().sum().item() > 0, f"帧 {i} 的输出应依赖历史帧"

    def test_same_frame_cross_stream_not_masked(self):
        """同帧流间不屏蔽：帧 i 的输出依赖 e_prim（p 流参与同帧融合）。"""
        torch.manual_seed(0)
        dec = self._make()
        dec.eval()
        B, T = 1, 10
        state_seq = torch.randn(B, T, STATE_DIM)
        prim_vec = torch.randn(B, dec.prim_emb.weight.shape[1], requires_grad=True)
        lengths = torch.full((B,), T)
        r, _, _ = dec(state_seq, torch.zeros(B, dtype=torch.long), prim_vec, lengths, torch.zeros(B, dtype=torch.long))
        grad = torch.autograd.grad(r[0, 3], prim_vec)[0]
        assert grad.abs().sum().item() > 0, "帧 3 的输出应经由同帧 e_prim token 依赖 prim_vec"


class TestWindowBuilders:
    def test_stage_windows(self):
        demos = collect_demos([0], 2, seed0=0)
        w = build_stage_windows(demos, n_obs_steps=6, frame_gap=2)
        assert w["state_seq"].shape[1:] == (6, STATE_DIM)
        assert len(w["primitive"]) == len(w["state_seq"])
        assert set(np.unique(w["primitive"])) <= set(range(len(PRIMITIVES)))

    def test_value_windows_layout(self):
        """序列布局 [anchor, mid×N, rewind×r, pad]，target=steps_to_go 且 pad 位为 0。"""
        demos = collect_demos([0], 3, seed0=0)
        N, R = 6, 3
        w = build_value_windows(demos, n_obs_steps=N, max_rewind_steps=R, rew_prob=1.0, seed=0)
        total = 1 + N + R
        assert w["state_seq"].shape[1:] == (total, STATE_DIM)
        assert w["target"].shape == (len(w["state_seq"]), total)
        assert w["lengths"].min() >= 1 + N and w["lengths"].max() <= total
        # 有效位置 target = 1 − progress ∈ [0,1]，pad 位置为 0
        for i, L in enumerate(w["lengths"]):
            assert (w["target"][i, :L] >= 0).all() and (w["target"][i, :L] <= 1).all()
            assert (w["target"][i, L:] == 0).all()
        # rew_prob=1 时多数窗口带 rewind
        assert (w["lengths"] > 1 + N).mean() > 0.5


class TestSARM2Algo:
    def test_stage_update_metrics(self):
        algo = make_algo()
        demos = collect_demos([0], 2, seed0=0)
        w = build_stage_windows(demos)
        m = algo.update({k: v[:8] for k, v in w.items()})
        assert {"loss", "ce_prim", "ce_group", "prim_acc"} <= set(m)

    def test_value_update_metrics(self):
        algo = make_algo()
        algo.phase = "value"
        demos = collect_demos([0], 2, seed0=0)
        w = build_value_windows(demos)
        m = algo.update({k: v[:8] for k, v in w.items()})
        assert {"loss", "mse", "aux_loss", "gate_entropy", "s_route", "ce_prim"} <= set(m)

    def test_invalid_phase_raises(self):
        algo = make_algo()
        algo.phase = "bogus"
        with pytest.raises(ValueError):
            algo.update({"state_seq": np.zeros((1, 6, STATE_DIM), np.float32), "primitive": np.zeros(1)})

    def test_ablation_flags_run(self):
        for flags in ({"single_gate": True}, {"use_primitives": False}, {"single_gate": True, "use_primitives": False}):
            algo = make_algo(**flags)
            algo.phase = "value"
            demos = collect_demos([0], 1, seed0=0)
            w = build_value_windows(demos)
            m = algo.update({k: v[:4] for k, v in w.items()})
            assert np.isfinite(m["loss"])

    def test_predict_value_preserves_module_modes(self):
        """predict_value 分别记录两个模块训练态：eval 的 decoder 不被误切回 train。"""
        algo = make_algo()
        algo.phase = "value"
        demos = collect_demos([0], 1, seed0=0)
        algo.stage_estimator.train()
        algo.value_decoder.eval()  # 曾被统一 train() 误切回 train（dropout 打开）
        r_hat = algo.predict_value(demos[0]["states"][:-1], task_id=0)
        assert not np.isnan(r_hat).all()
        assert algo.stage_estimator.training, "stage_estimator 原为 train，应恢复"
        assert not algo.value_decoder.training, "value_decoder 原为 eval，应保持"

    def test_save_load_roundtrip(self, tmp_path):
        algo = make_algo()
        algo.phase = "value"  # 触发两个子模型都存在
        demos = collect_demos([0], 1, seed0=0)
        w = build_value_windows(demos)
        algo.update({k: v[:4] for k, v in w.items()})
        algo.save_state(tmp_path / "sarm2.pt")
        algo2 = make_algo(seed=99)
        algo2.load_state(tmp_path / "sarm2.pt")
        for k in ("stage_estimator", "value_decoder"):
            a = dict(getattr(algo, k).state_dict())
            b = dict(getattr(algo2, k).state_dict())
            for key in a:
                assert torch.allclose(a[key], b[key]), f"{k}.{key} 加载后不一致"
        assert algo2.phase == "value"


# ================================================================= 学习测试


def train_full(seed=0, n_per_task=8, stage_steps=500, value_steps=500, **over):
    """完整两阶段训练，返回 (algo, val_demos, heldout_demos)。"""
    torch.manual_seed(seed)
    train = collect_demos([0, 1], n_per_task, seed0=seed * 10_000)
    val = collect_demos([0, 1], 3, seed0=77_000 + seed)
    heldout = collect_demos([2], 3, seed0=99_000 + seed)

    algo = make_algo(seed=seed, **over)
    rng = np.random.default_rng(seed)

    algo.phase = "stage"
    sw = build_stage_windows(train)
    for _ in range(stage_steps):
        algo.update(next(batch_iter(sw, 64, rng)))

    algo.phase = "value"
    vw = build_value_windows(train, seed=seed)
    for _ in range(value_steps):
        algo.update(next(batch_iter(vw, 64, rng)))
    return algo, val, heldout


@torch.no_grad()
def stage_accuracy(algo, demos):
    w = build_stage_windows(demos)
    logits, _ = algo.stage_estimator(torch.as_tensor(w["state_seq"]))
    return float((logits.argmax(-1).numpy() == w["primitive"]).mean())


@torch.no_grad()
def value_mse(algo, demos):
    w = build_value_windows(demos, rew_prob=0.0, seed=7)
    rs = []
    for i in range(0, len(w["state_seq"]), 64):
        b = {k: v[i : i + 64] for k, v in w.items()}
        state_seq = torch.as_tensor(b["state_seq"])
        mid = state_seq[:, 1 : 1 + algo.n_obs_steps]
        logits, _ = algo.stage_estimator(mid)
        probs = torch.softmax(logits, -1)
        prim_vec = algo.value_decoder.compose_prim_embedding(probs)
        gate = algo.prim_to_group[logits.argmax(-1)]
        r, _, _ = algo.value_decoder(
            state_seq, torch.as_tensor(b["task"]), prim_vec, torch.as_tensor(b["lengths"]), gate
        )
        rs.append(r.numpy())
    r = np.concatenate(rs)
    valid = np.arange(r.shape[1])[None, :] < w["lengths"][:, None]
    mse = float((((r - w["target"]) ** 2) * valid).sum() / valid.sum())
    const = float((((0.5 - w["target"]) ** 2) * valid).sum() / valid.sum())  # 常数基线
    return mse, const


@pytest.mark.slow
class TestSARM2Learning:
    def test_stage_learns_and_transfers(self):
        """阶段估计器：训练任务高精度 + held-out 任务正迁移（论文 §IV-A 泛化主张）。"""
        algo, val, heldout = train_full(seed=3, value_steps=0)
        acc_val, acc_held = stage_accuracy(algo, val), stage_accuracy(algo, heldout)
        assert acc_val >= 0.85, f"val acc {acc_val:.2f} 应 >= 0.85"
        assert acc_held >= 0.7, f"held-out acc {acc_held:.2f} 应 >= 0.7（正迁移）"

    def test_value_learns_and_monotone(self):
        """值解码器：MSE 显著低于常数基线；预测值沿 episode 递减（进度语义）。"""
        algo, val, _ = train_full(seed=5)
        mse, const = value_mse(algo, val)
        assert mse < 0.25 * const, f"val MSE {mse:.4f} 应 < 常数基线 {const:.4f} 的 1/4"
        r_hat = algo.predict_value(val[0]["states"][:-1], task_id=val[0]["task_id"])
        tail = r_hat[~np.isnan(r_hat)]
        assert tail[-1] < tail[0], "预测 steps_to_go 应沿 episode 下降"
        assert np.mean(np.diff(tail)) < 0

    def test_routing_not_collapsed(self):
        """路由健康度：训练后 S_route 远离 1（论文 full model 0.10；塌缩=1）。"""
        algo, val, _ = train_full(seed=7)
        w = build_value_windows(val, rew_prob=0.0, seed=7)
        s_routes = []
        for i in range(0, len(w["state_seq"]), 64):
            b = {k: v[i : i + 64] for k, v in w.items()}
            state_seq = torch.as_tensor(b["state_seq"])
            with torch.no_grad():
                mid = state_seq[:, 1 : 1 + algo.n_obs_steps]
                logits, _ = algo.stage_estimator(mid)
                prim_vec = algo.value_decoder.compose_prim_embedding(torch.softmax(logits, -1))
                gate = algo.prim_to_group[logits.argmax(-1)]
                _, _, info = algo.value_decoder(
                    state_seq, torch.as_tensor(b["task"]), prim_vec, torch.as_tensor(b["lengths"]), gate
                )
            s_routes.append(collapse_score(info["avg_gate_probs"], algo.value_decoder.top_k))
        assert float(np.mean(s_routes)) < 0.5, "路由不应塌缩（S_route 应 < 0.5）"

    def test_ablations_trainable(self):
        """w/o MG / w/o SE 消融可训练（论文 Table 1 对应开关；mock 规模不做排序断言）。"""
        for flags in ({"single_gate": True}, {"use_primitives": False}):
            algo, val, _ = train_full(seed=11, **flags)
            mse, const = value_mse(algo, val)
            assert mse < 0.5 * const, f"{flags} 消融也应学到低于常数基线一半的 MSE（{mse:.4f} vs {const:.4f}）"

    def test_scorer_separates_full_from_partial(self):
        """Sarm2Scorer：完整 demo 得分 > 中途放弃的半截 demo（数采打分语义）。"""
        algo, val, _ = train_full(seed=13)

        def to_episode(states, task_id, ep_id):
            T = states.shape[0] - 1
            meta = EpisodeMeta(episode_id=ep_id, task=f"mock_pick_place_{task_id}", robot="mock")
            meta.extra = {"task_id": task_id}
            return Episode(meta=meta, states=states, actions=np.zeros((T, 3), np.float32))

        scorer = Sarm2Scorer(algo)
        full = [scorer.score(to_episode(d["states"], d["task_id"], f"f{i}")) for i, d in enumerate(val)]
        half = []
        for i in range(3):
            task = MockPickPlaceTask(task_id=0)
            expert = MockPickPlaceExpert()
            obs = task.reset(seed=55_000 + i)
            states = [obs["state"]]
            for _ in range(12):  # 只走到 approach 中途
                res = task.step(expert.act(obs))
                states.append(res.observation["state"])
                obs = res.observation
            half.append(
                scorer.score(to_episode(np.stack(states).astype(np.float32), 0, f"h{i}"))
            )
        assert np.mean(full) > np.mean(half) + 0.1, (
            f"完整 demo 均分 {np.mean(full):.3f} 应高于半截 {np.mean(half):.3f} 至少 0.1"
        )
