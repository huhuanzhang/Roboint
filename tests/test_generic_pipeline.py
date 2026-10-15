"""generic pipeline 扩展的离线测试：任务 dry-run 语义 + rollout 动作块消费。

不依赖真机/服务器：FoldClothesTask 用假本体，rollout 用 mock 任务。
"""

import numpy as np

from robo_int.infra.factory import model_policy_fn
from robo_int.infra.rollout import run_episode
from robo_int.models.mock import MockVLA
from robo_int.robots.base import BaseRobot
from robo_int.tasks.base import BaseTask, StepResult
from robo_int.tasks.fold_clothes import FoldClothesTask
from robo_int.tasks.mock_reach import MockReachTask


class FakeRobot(BaseRobot):
    """记录 send_action 次数的假本体，观测恒定。"""

    def __init__(self, state_dim=14):
        self.state_dim = state_dim
        self.sent: list[np.ndarray] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_state(self):
        from robo_int.robots.base import RobotState

        return RobotState(joint_positions=np.zeros(self.state_dim, dtype=np.float32))

    def send_action(self, action: np.ndarray) -> None:
        self.sent.append(np.asarray(action, dtype=np.float32))

    def capture_observation(self) -> dict:
        return {
            "images": {"head": np.zeros((8, 8, 3), dtype=np.uint8)},
            "state": np.linspace(0.1, 0.2, self.state_dim, dtype=np.float32),
        }


def test_build_robot_mock():
    """factory.build_robot 的 mock 分支（真机分支需要硬件，由实机链路验证）。"""
    from omegaconf import OmegaConf

    from robo_int.infra.factory import build_robot
    from robo_int.robots.mock import MockRobot

    cfg = OmegaConf.create({"robot": {"name": "mock", "num_joints": 6, "control_hz": 10.0}})
    robot = build_robot(cfg)
    assert isinstance(robot, MockRobot) and robot.num_joints == 6


def test_fold_clothes_dry_run_never_sends():
    robot = FakeRobot()
    task = FoldClothesTask(robot, max_episode_steps=5, dry_run=True, camera_warmup_frames=0)
    obs = task.reset()
    # 观测 schema：images + state + instruction（BaseTask.get_observation 约定）
    assert set(("images", "state", "instruction")) <= set(obs)
    assert obs["state"].shape == (14,)
    assert obs["instruction"] == "fold the clothes neatly"

    result = task.step(np.ones(14, dtype=np.float32) * 2.0)
    assert result.observation["state"].shape == (14,)
    assert result.reward is None and not result.terminated
    assert len(task.command_log) == 1
    assert robot.sent == []  # dry-run 绝不下发
    assert task.command_sent_steps == 0

    for _ in range(4):
        result = task.step(np.zeros(14, dtype=np.float32))
    assert result.truncated  # 步数预算用尽
    assert len(task.command_log) == 5 and robot.sent == []


def test_rollout_consumes_action_chunks():
    class ChunkedPolicy:
        def __init__(self, chunk_size, action_dim):
            self.chunk_size, self.action_dim = chunk_size, action_dim
            self.calls = 0

        def __call__(self, observation):
            self.calls += 1
            return np.full((self.chunk_size, self.action_dim), 0.05, dtype=np.float32)

    task = MockReachTask(max_episode_steps=10, success_radius=0.001, action_limit=0.5)
    task.reset(seed=0)
    policy = ChunkedPolicy(chunk_size=4, action_dim=2)
    episode = run_episode(task, policy, seed=0)

    # 步数预算按实际执行步数计：10 步 / 每块 4 步 -> 3 次推理、第 3 块截断为 2 步
    assert policy.calls == 3
    assert episode.actions.shape == (10, 2)
    assert episode.states.shape[0] == 11  # reset 观测 + 每步之后
    assert episode.meta.num_steps == 10


def test_rollout_single_step_policy_unchanged():
    """单步（1D）策略行为与历史版本一致。"""

    class SingleStepPolicy:
        def __init__(self):
            self.action_dim = 2

        def __call__(self, observation):
            return np.asarray([0.05, -0.05], dtype=np.float32)

    task = MockReachTask(max_episode_steps=6, success_radius=0.001, action_limit=0.5)
    episode = run_episode(task, SingleStepPolicy(), seed=0)
    assert episode.actions.shape == (6, 2)
    assert episode.states.shape[0] == 7


def test_model_policy_fn_returns_full_chunk():
    model = MockVLA(state_dim=4, action_dim=2, chunk_size=3, hidden_dim=16)
    observation = {"state": np.zeros(4, dtype=np.float32)}
    chunk = model_policy_fn(model)(observation)
    assert chunk.shape == (3, 2)  # (chunk_size, action_dim)，不再丢弃为 chunk[0]
