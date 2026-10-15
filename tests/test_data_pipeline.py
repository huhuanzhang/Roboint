"""数据管线测试：Episode 新字段往返 + policy/executed 区分 + LeRobot 转换。

LeRobot 转换依赖 lerobot 包（本机旧版 fork 可用），缺失时跳过。
"""

import numpy as np
import pytest

from robo_int.data.episodes import Episode, EpisodeMeta, EpisodeStore, new_episode_id
from robo_int.infra.rollout import run_episode
from robo_int.robots.base import BaseRobot
from robo_int.tasks.fold_clothes import FoldClothesTask


class ClampingRobot(BaseRobot):
    """假本体：apply_safety 把动作夹到 state±0.01（模拟 Piper 安全链）。"""

    def __init__(self):
        self.sent = []
        self._state = np.full(14, 0.1, dtype=np.float32)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_state(self):
        from robo_int.robots.base import RobotState

        return RobotState(joint_positions=self._state.copy())

    def send_action(self, action: np.ndarray) -> None:
        self.sent.append(np.asarray(action, dtype=np.float32))

    def capture_observation(self) -> dict:
        return {
            "images": {"head": np.full((4, 4, 3), 7, dtype=np.uint8)},
            "state": self._state.copy(),
            "timestamp": 1_000_000.0,
        }

    def apply_safety(self, action, current_state):
        return np.clip(np.asarray(action, dtype=np.float32),
                       np.asarray(current_state) - 0.01, np.asarray(current_state) + 0.01)


def _fake_chunk_policy(observation):
    # 远超限幅的原始动作 -> 安全链后应被明显压缩
    return np.full((4, 14), 2.0, dtype=np.float32)


def test_rollout_separates_policy_and_executed_actions():
    task = FoldClothesTask(ClampingRobot(), max_episode_steps=8, dry_run=True, camera_warmup_frames=0)
    episode = run_episode(task, _fake_chunk_policy)

    assert episode.policy_actions.shape == (8, 14)
    assert episode.actions.shape == (8, 14)
    # 策略原始输出 vs 安全链后执行命令必须可区分
    np.testing.assert_allclose(episode.policy_actions, 2.0)
    np.testing.assert_allclose(episode.actions, 0.11, atol=1e-6)
    # meta：指令 / dry-run 标记 / 时间戳
    assert episode.meta.instruction == "fold the clothes neatly"
    assert episode.meta.extra["dry_run"] is True
    assert episode.meta.extra["training_eligible"] is False
    assert episode.timestamps.shape == (9,)
    assert episode.timestamps[0] == 1_000_000.0


def test_episode_store_roundtrip(tmp_path):
    store = EpisodeStore(tmp_path)
    episode = Episode(
        meta=EpisodeMeta(episode_id="ep_000001", task="t", robot="r", instruction="do it"),
        states=np.zeros((3, 14), dtype=np.float32),
        actions=np.full((2, 14), 0.1, dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        images={"head": np.zeros((3, 4, 4, 3), dtype=np.uint8)},
        policy_actions=np.full((2, 14), 2.0, dtype=np.float32),
        timestamps=np.linspace(0.0, 0.1, 3),
    )
    store.save(episode)
    loaded = store.load("ep_000001")
    assert loaded.meta.instruction == "do it"
    np.testing.assert_allclose(loaded.policy_actions, 2.0)
    np.testing.assert_allclose(loaded.timestamps, episode.timestamps)
    np.testing.assert_array_equal(loaded.images["head"].shape, (3, 4, 4, 3))


def test_new_episode_id_increments():
    assert new_episode_id([]) == "ep_000001"
    metas = [EpisodeMeta(episode_id=f"ep_{i:06d}", task="t", robot="r") for i in range(1, 3)]
    assert new_episode_id(metas) == "ep_000003"


def test_lerobot_conversion(tmp_path):
    pytest.importorskip("lerobot")
    from robo_int.data.lerobot_io import _import_lerobot_dataset, episodes_to_lerobot

    # 顶层包可导入不代表 dataset 依赖已安装；回读也使用适配器选中的 API。
    try:
        LeRobotDataset, _ = _import_lerobot_dataset()
    except ImportError as exc:
        pytest.skip(f"LeRobot dataset 依赖不可用：{exc}")

    store_dir = tmp_path / "episodes"
    store = EpisodeStore(store_dir)
    episode = Episode(
        meta=EpisodeMeta(
            episode_id="ep_000001",
            task="FoldClothesTask",
            robot="piper_dual_arm",
            source="policy",
            instruction="fold the clothes neatly",
            kept=True,
            extra={"dry_run": True, "training_eligible": False},
        ),
        states=np.random.default_rng(0).uniform(0, 0.5, (5, 14)).astype(np.float32),
        actions=np.full((4, 14), 0.01, dtype=np.float32),
        rewards=None,
        images={
            "head": np.full((5, 8, 8, 3), 10, dtype=np.uint8),
            "left_wrist": np.full((5, 8, 8, 3), 20, dtype=np.uint8),
            "right_wrist": np.full((5, 8, 8, 3), 30, dtype=np.uint8),
        },
        policy_actions=np.full((4, 14), 2.0, dtype=np.float32),
        timestamps=np.linspace(0.0, 0.13, 5),
    )
    store.save(episode)

    out = tmp_path / "lerobot_out"
    summary = episodes_to_lerobot(store_dir, out, fps=30)
    assert summary["episodes"] == 1 and summary["frames"] == 4
    assert summary["dry_run"] is True and summary["training_eligible"] is False
    for cam in ("head", "left_wrist", "right_wrist"):
        assert f"observation.images.{cam}" in summary["features"]

    # 回读验证 schema（root 即数据集目录本身）
    ds = LeRobotDataset(out.name, root=out)
    assert ds.meta.total_episodes == 1
    frame = ds[0]
    assert frame["observation.state"].shape == (14,)
    assert frame["action"].shape == (14,)
    assert frame["task"] == "fold the clothes neatly"
