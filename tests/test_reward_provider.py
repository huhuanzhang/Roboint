"""reward provider 契约 + SARM2 适配的离线功能测试（无真机/无 server）。

如实结论：SARM2 输出是真实模型输出（forward 全链路），但未训练时 value≈常数、
reward≈0 是预期行为——测试只验证形状/有限性/来源可追溯，不伪造变化。
"""

import copy

import numpy as np
import pytest
import torch

from robo_int.algos.common.reward import BaseRewardProvider
from robo_int.algos.sarm2.reward_provider import Sarm2RewardProvider
from robo_int.algos.sarm2.sarm2 import SARM2

PRIMITIVES = ["null", "approach", "grasp", "transport", "release", "retreat"]
GROUPS = ["null", "approach or retreat", "grasp or release", "transport"]


def _sarm2(state_dim=8, seed=0):
    return SARM2(
        primitives=PRIMITIVES, groups=GROUPS, state_dim=state_dim, num_tasks=2,
        d_model=32, stage_layers=1, value_layers=1, n_heads=2,
        n_obs_steps=4, frame_gap=1,
        num_experts=4, expert_hidden=16, gate_hidden=16, seed=seed,
    )


def test_provider_is_base_and_provenance():
    algo = _sarm2()
    provider = Sarm2RewardProvider(algo)
    assert isinstance(provider, BaseRewardProvider)
    prov = provider.provenance()
    assert prov["reward_source"] == "sarm2"
    assert prov["reward_model"] == "Sarm2RewardProvider"
    assert prov["reward_model_version"] == "untrained"
    assert prov["reward_is_placeholder"] is False


def test_single_step_compute_not_applicable():
    provider = Sarm2RewardProvider(_sarm2())
    with pytest.raises(NotImplementedError, match="compute_episode"):
        provider.compute({"state": np.zeros(8)}, np.zeros(2), {"state": np.zeros(8)})


def test_compute_episode_shapes_and_finiteness():
    provider = Sarm2RewardProvider(_sarm2())
    states = np.random.default_rng(0).uniform(-1, 1, (20, 8)).astype(np.float32)
    out = provider.compute_episode(states)
    assert out["reward"].shape == (19,)
    assert out["value"].shape == (20,)
    assert out["stage"].shape == (20, len(PRIMITIVES))
    # frame_gap=1, n_obs=4 -> need=4：前 3 帧窗口不足为 NaN，之后有限
    assert np.isnan(out["value"][:3]).all()
    assert np.isfinite(out["value"][4:]).all()
    assert (out["value"][4:] > 0).all() and (out["value"][4:] < 1).all()  # sigmoid(steps_to_go)
    valid_reward = out["reward"][3:]
    assert np.isfinite(valid_reward).all()
    labels = provider.stage_labels(out["stage"])
    assert labels[:3].tolist() == [-1, -1, -1]
    assert (labels[4:] >= 0).all()


def test_untrained_reward_is_near_constant_reported():
    """未训练模型的诚实预期：value 近常数 -> reward 近 0。不伪造变化。"""
    provider = Sarm2RewardProvider(_sarm2())
    states = np.random.default_rng(1).uniform(-1, 1, (20, 8)).astype(np.float32)
    out = provider.compute_episode(states)
    assert float(np.abs(out["reward"][3:]).max()) < 0.2  # 随机初始化 sigmoid 波动很小


def test_training_path_functional():
    """两阶段 update 的功能测试：forward/loss/backward/step/参数变化。

    标签为占位（primitive=0、progress=t/T），仅验证训练代码链路，不产出可用模型。
    """
    algo = _sarm2()
    rng = np.random.default_rng(2)
    states = rng.uniform(-1, 1, (40, 8)).astype(np.float32)
    # 占位标签（functional test）：原语恒 null，progress 线性
    prims = np.zeros(39, dtype=np.int64)
    progress = np.linspace(0, 1, 39, dtype=np.float32)

    algo.phase = "stage"
    stage_batch = {
        "state_seq": np.stack([states[t - 3 : t + 1] for t in range(3, 39)]),
        "primitive": prims[3:],
    }
    snapshot_stage = copy.deepcopy(algo.stage_estimator.state_dict())
    for _ in range(2):
        metrics = algo.update(stage_batch)
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["ce_prim"])
    assert any(not torch.equal(snapshot_stage[k], algo.stage_estimator.state_dict()[k])
               for k in snapshot_stage)

    algo.phase = "value"
    demo = {"states": states, "primitives": prims, "progress": progress, "task_id": 0}
    from robo_int.algos.sarm2.sarm2 import build_value_windows

    value_batch = build_value_windows([demo], n_obs_steps=4, frame_gap=1, max_rewind_steps=2)
    snapshot_value = copy.deepcopy(algo.value_decoder.state_dict())
    for _ in range(2):
        metrics = algo.update(value_batch)
    for key in ("loss", "mse", "aux_loss", "gate_entropy", "s_route"):
        assert np.isfinite(metrics[key]), f"{key} 非有限"
    assert any(not torch.equal(snapshot_value[k], algo.value_decoder.state_dict()[k])
               for k in snapshot_value)
    # 训练过 -> provenance 版本不再是 untrained
    provider = Sarm2RewardProvider(algo)
    assert provider.provenance()["reward_model_version"] == "updates=4"


def test_no_piper_or_rlt_coupling():
    """契约与 SARM2 适配不得 import 本体/任务/RLT（挂件不是底座的反向约束）。"""
    import ast

    for path in (
        "robo_int/algos/common/reward.py",
        "robo_int/algos/sarm2/reward_provider.py",
    ):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith(("robo_int.robots", "robo_int.tasks", "robo_int.algos.rlt")), (
                    f"{path} 出现 {node.module} 耦合"
                )
