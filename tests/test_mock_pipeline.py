"""mock 全链路端到端测试：数采 -> 打分清洗 -> SFT -> 评测。

这是「能跑优先」的核心保障：任何接口改动如果破坏链路，此测试立刻红。
"""

import pytest

from robo_int.data.cleaning import clean, load_kept, score_all
from robo_int.data.dataset import EpisodeDataset
from robo_int.data.episodes import EpisodeStore
from robo_int.data.scoring import default_scorer
from robo_int.infra.config import RolloutConfig, TrainConfig, OptimizerConfig, RegularizationConfig
from robo_int.infra.factory import build_expert, build_task, model_policy_fn
from robo_int.infra.rollout import evaluate, run_and_save, run_episode
from robo_int.models.mock import MockVLA
from robo_int.algos.sft import SFT
from robo_int.infra.train_loop import TrainLoop


@pytest.fixture()
def task():
    return build_task(_mock_cfg())


def _mock_cfg():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config")


def test_expert_collects_and_store_roundtrips(task, tmp_path):
    expert = build_expert(_mock_cfg())
    store = EpisodeStore(tmp_path)
    episodes = run_and_save(task, expert, store, RolloutConfig(num_episodes=3, seed=0))

    assert len(episodes) == 3
    # 专家策略成功率应该很高（比例控制器在 50 步内到达）
    assert sum(e.meta.success for e in episodes) >= 2

    # 落盘回读一致
    metas = store.load_index()
    assert len(metas) == 3
    reloaded = store.load(metas[0].episode_id)
    assert reloaded.states.shape[1] == 4  # pos2 + goal2
    assert reloaded.actions.shape == (reloaded.meta.num_steps, 2)


def test_scoring_and_cleaning(task, tmp_path):
    expert = build_expert(_mock_cfg())
    store = EpisodeStore(tmp_path)
    run_and_save(task, expert, store, RolloutConfig(num_episodes=4, seed=1))

    scorer = default_scorer(action_limit=0.15)
    metas = score_all(store, scorer)
    assert all(m.score is not None and 0.0 <= m.score <= 1.0 for m in metas)

    metas = clean(store, min_score=0.3)
    assert all(isinstance(m.kept, bool) for m in metas)
    assert len(load_kept(store)) >= 1


def test_dataset_action_alignment():
    """回归测试：episode 里的 (state, action) 必须是专家在同观测下产生的。

    曾有 bug：rollout 循环未更新观测，策略一直看初始观测——
    训练数据配对错乱，SFT 无法收敛。此测试防止该类回归。
    """
    import numpy as np

    task = build_task(_mock_cfg())
    expert = build_expert(_mock_cfg())
    episodes = [run_episode(task, expert, seed=s) for s in range(3)]
    dataset = EpisodeDataset(episodes)
    gain = 0.8
    expected = np.clip(gain * (dataset.states[:, 2:4] - dataset.states[:, 0:2]), -0.15, 0.15)
    assert np.abs(expected - dataset.actions).max() < 1e-5


def test_sft_learns_expert_and_evaluates(task, tmp_path):
    """SFT 若干步后 loss 下降、评测成功率显著高于随机初始化。"""
    expert = build_expert(_mock_cfg())
    store = EpisodeStore(tmp_path / "data")
    episodes = run_and_save(task, expert, store, RolloutConfig(num_episodes=8, seed=2))
    dataset = EpisodeDataset(episodes)

    train_cfg = TrainConfig(
        total_steps=120,
        batch_size=64,
        log_every=1000,
        checkpoint_every=10_000,
        run_dir=str(tmp_path / "run"),
        optimizer=OptimizerConfig(lr=1e-3),
    )
    model = MockVLA(state_dim=4, action_dim=2, hidden_dim=128)
    algo = SFT(model, train_cfg)

    loss_before = float(algo.update(dataset.sample_batch(64))["loss"])
    loop = TrainLoop(algo, train_cfg, batch_sampler=dataset.sample_batch)
    loop.run()
    loss_after = float(algo.update(dataset.sample_batch(64))["loss"])
    assert loss_after < loss_before * 0.2, (loss_before, loss_after)

    metrics = evaluate(task, model_policy_fn(model), RolloutConfig(num_episodes=10, seed=7))
    assert metrics["success_rate"] >= 0.8, metrics
