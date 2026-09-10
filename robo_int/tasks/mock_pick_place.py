"""Mock 任务：2D 多阶段 pick-and-place（纯 numpy，无物理引擎依赖）。

作为 SARM2 论文 10 任务基准（叠短裤/擦白板/收雨伞……）的 mock 替身：
多阶段、带动作基元（action primitive）结构、进度非线性依赖几何而非时间步。
用于验证「阶段估计器 + 多门 MoE 值模型」能从状态序列学出阶段与进度。

- 原语词汇表（PRIMITIVES，模拟论文 21 原语 + null）:
    null -> approach -> grasp -> transport -> release -> retreat
- 语义分组（SEMANTIC_GROUPS，模拟论文把 K+1 原语归成 M+1 组选 MoE 门）:
    null / approach or retreat（自由移动）/ grasp or release（夹爪动作）/ transport
- 任务变体（task_id 0/1/2，模拟论文多任务训练 + held-out 泛化评测）:
    0 抓左下放右上、1 抓右下放左上（训练），2 抓下放上（held-out 迁移测试）
- 每步 info 携带真值标注: primitive（阶段索引）/ progress（[0,1] 任务进度），
  episode 级进度分段非线性（按距离比例插值），使「时间索引预测器」不足以作弊。

阶段歧义性设计（让时序窗口/多门真正有用）：
- approach 与 retreat 同为自由移动，仅凭单帧状态无法区分（须看运动方向）；
- grasp 与 release 同为夹爪开合，须结合位置上下文（物体旁 vs 放置点旁）。
"""

from __future__ import annotations

import numpy as np

from robo_int.tasks.base import BaseTask, ResetMode, StepResult

# ---------------------------------------------------------------- 原语词汇表

PRIMITIVES = ["null", "approach", "grasp", "transport", "release", "retreat"]
PRIM_NULL, PRIM_APPROACH, PRIM_GRASP, PRIM_TRANSPORT, PRIM_RELEASE, PRIM_RETREAT = range(6)

# 语义分组（SARM2 的 class_list：组内原语共享一个 MoE 门）
SEMANTIC_GROUPS = ["null", "approach or retreat", "grasp or release", "transport"]

# 原语 -> 组 索引表（build_hierarchy 同款方式可由 SEMANTIC_GROUPS 推出，这里直接写死便于核对）
PRIM_TO_GROUP = [0, 1, 2, 3, 2, 1]

# 进度分段边界（各原语占据的进度区间；组内按几何距离比例插值 -> 非线性）
PROGRESS_BOUNDS = {
    PRIM_NULL: (0.00, 0.05),
    PRIM_APPROACH: (0.05, 0.35),
    PRIM_GRASP: (0.35, 0.45),
    PRIM_TRANSPORT: (0.45, 0.75),
    PRIM_RELEASE: (0.75, 0.85),
    PRIM_RETREAT: (0.85, 1.00),
}

# 任务变体指令（区域定义见 _sample_region 的 regs）—— 0/1 训练、2 held-out
TASK_VARIANTS = [
    "pick from left and place on the right",
    "pick from right and place on the left",
    "pick from bottom and place on top",
]
HOME = np.array([0.5, 0.5], dtype=np.float32)


class MockPickPlaceTask(BaseTask):
    """2D 平面抓放：手爪 (x,y) + 夹爪开度 + 单物体。

    - 状态: [hand_x, hand_y, gripper, obj_x, obj_y, held]（6 维）
    - 动作: [dx, dy, d_gripper]（3 维，限幅 ±action_limit）
    - 奖励: None（延后打分——正是 SARM2 奖励模型的用武之地）
    - 成功: 物体被放到放置区域内且已松爪
    """

    reset_mode = ResetMode.SCRIPTED

    def __init__(
        self,
        task_id: int = 0,
        max_episode_steps: int = 90,
        action_limit: float = 0.10,
        grasp_radius: float = 0.06,
        place_radius: float = 0.08,
        grasp_threshold: float = 0.5,
        null_steps: int = 3,
        grasp_steps: int = 3,
        release_steps: int = 3,
    ):
        if task_id not in range(len(TASK_VARIANTS)):
            raise ValueError(f"task_id 需在 0..{len(TASK_VARIANTS) - 1}， got {task_id}")
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.action_limit = action_limit
        self.grasp_radius = grasp_radius
        self.place_radius = place_radius
        self.grasp_threshold = grasp_threshold
        self.null_steps, self.grasp_steps, self.release_steps = (
            null_steps,
            grasp_steps,
            release_steps,
        )

        self._hand = np.zeros(2, dtype=np.float32)
        self._gripper = 0.0
        self._obj = np.zeros(2, dtype=np.float32)
        self._held = False
        self._steps = 0
        self._phase = PRIM_NULL
        self._phase_step = 0
        self._phase_start_dist = 1.0  # 阶段起点参考距离（进度插值用）
        self._phase_min_dist = 1.0  # 阶段内最优距离包络（保证进度对噪声单调）
        self._rng = np.random.default_rng()
        self._episode_done = False

    # ------------------------------------------------------------ 任务接口

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # 采样物体/放置点（按任务变体区域）
        self._obj = self._sample_region("object").astype(np.float32)
        self._place = self._sample_region("place").astype(np.float32)
        self._hand = HOME + self._rng.uniform(-0.03, 0.03, size=2).astype(np.float32)
        self._gripper = 0.0  # 张开
        self._held = False
        self._steps = 0
        self._phase, self._phase_step = PRIM_NULL, 0
        self._phase_start_dist = float(np.linalg.norm(self._hand - self._obj))
        self._episode_done = False
        return self.get_observation()

    def _sample_region(self, which: str) -> np.ndarray:
        regs = {
            0: {"object": (0.15, 0.30, 0.15, 0.30), "place": (0.70, 0.85, 0.70, 0.85)},
            1: {"object": (0.70, 0.85, 0.15, 0.30), "place": (0.15, 0.30, 0.70, 0.85)},
            2: {"object": (0.40, 0.60, 0.10, 0.20), "place": (0.40, 0.60, 0.80, 0.90)},
        }
        x0, x1, y0, y1 = regs[self.task_id][which]
        return self._rng.uniform((x0, y0), (x1, y1))

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32), -self.action_limit, self.action_limit)
        self._hand = np.clip(self._hand + action[:2], 0.0, 1.0)
        self._gripper = float(np.clip(self._gripper + action[2], 0.0, 1.0))
        self._steps += 1

        # 抓取/释放判定：夹爪越过阈值 + 位置条件
        hand_obj = float(np.linalg.norm(self._hand - self._obj))
        if not self._held and self._gripper > self.grasp_threshold and hand_obj < self.grasp_radius:
            self._held = True
        elif self._held and self._gripper < self.grasp_threshold:
            self._held = False  # 松爪，物体落在当前位置
        if self._held:
            self._obj = self._hand.copy()

        # 阶段内距离包络更新（当前阶段目标的最近距离，进度插值用）
        tgt = {PRIM_APPROACH: self._obj, PRIM_TRANSPORT: self._place, PRIM_RETREAT: HOME}.get(self._phase)
        if tgt is not None:
            d = float(np.linalg.norm(self._hand - tgt))
            self._phase_min_dist = min(self._phase_min_dist, d)

        self._advance_phase(hand_obj)
        truncated = self._steps >= self.max_episode_steps
        terminated = self._episode_done
        return StepResult(
            observation=self.get_observation(),
            reward=None,  # 延后打分：SARM2 奖励模型的消费场景
            terminated=terminated,
            truncated=truncated,
            info={
                "primitive": self._phase,
                "progress": self._progress(),
                "steps": self._steps,
            },
        )

    # ------------------------------------------------------------ 阶段机（真值标注）

    def _advance_phase(self, hand_obj: float) -> None:
        """脚本化阶段推进（真值原语标签来源；专家控制器与之同构）。"""
        p, k = self._phase, self._phase_step
        if p == PRIM_NULL:
            if k >= self.null_steps - 1:
                self._enter(PRIM_APPROACH)
        elif p == PRIM_APPROACH:
            if hand_obj < self.grasp_radius:
                self._enter(PRIM_GRASP)
        elif p == PRIM_GRASP:
            if self._held and k >= self.grasp_steps - 1:
                self._enter(PRIM_TRANSPORT)
        elif p == PRIM_TRANSPORT:
            # 抓着物体（obj 跟随 hand）到达放置区域
            if float(np.linalg.norm(self._obj - self._place)) < self.place_radius:
                self._enter(PRIM_RELEASE)
        elif p == PRIM_RELEASE:
            if not self._held and k >= self.release_steps - 1:
                self._enter(PRIM_RETREAT)
        elif p == PRIM_RETREAT:
            if float(np.linalg.norm(self._hand - HOME)) < 0.05:
                self._episode_done = True
        self._phase_step += 1

    def _enter(self, phase: int) -> None:
        self._phase = phase
        self._phase_step = 0
        # 记录阶段起点距离，供组内进度插值
        target = {
            PRIM_APPROACH: self._obj,
            PRIM_TRANSPORT: self._place,
            PRIM_RETREAT: HOME,
        }.get(phase)
        if target is not None:
            d = max(float(np.linalg.norm(self._hand - target)), 1e-6)
            self._phase_start_dist = d
            self._phase_min_dist = d

    def _progress(self) -> float:
        """[0,1] 任务进度：分段边界 + 组内几何插值（非线性于时间步）。

        移动阶段用「阶段内最近距离包络」插值，保证对示教噪声单调不减。
        """
        lo, hi = PROGRESS_BOUNDS[self._phase]
        if self._phase == PRIM_NULL:
            frac = min(self._phase_step / max(self.null_steps, 1), 1.0)
        elif self._phase in (PRIM_APPROACH, PRIM_TRANSPORT, PRIM_RETREAT):
            frac = np.clip(1.0 - self._phase_min_dist / self._phase_start_dist, 0.0, 1.0)
        else:  # grasp / release：按步数比例
            n = self.grasp_steps if self._phase == PRIM_GRASP else self.release_steps
            frac = min((self._phase_step + 1) / max(n, 1), 1.0)
        return float(lo + (hi - lo) * frac)

    # ------------------------------------------------------------ 观测/成功

    def get_observation(self) -> dict:
        state = np.array(
            [self._hand[0], self._hand[1], self._gripper, self._obj[0], self._obj[1], float(self._held)],
            dtype=np.float32,
        )
        return {
            "state": state,
            "instruction": TASK_VARIANTS[self.task_id],
            # 上下文字段（示教者可见，但不进紧凑 state 向量——模型须靠 task 嵌入推断放置区）
            "place": self._place.copy(),
            "steps": self._steps,
            "images": {},  # mock 无图像；真机任务在此填 3 相机帧（SARM2 输入）
        }

    def success(self) -> bool:
        return (
            not self._held
            and float(np.linalg.norm(self._obj - self._place)) < self.place_radius
        )

    def get_info(self) -> dict:
        return {
            **super().get_info(),
            "task_id": self.task_id,
            "action_dim": 3,
            "action_spec": "delta_xy_dgrip",
            "primitives": PRIMITIVES,
            "semantic_groups": SEMANTIC_GROUPS,
            "state_dim": 6,
        }


class MockPickPlaceExpert:
    """脚本化「专家」：自带与任务阶段机同构的状态机 + 比例控制器（带高斯噪声）。

    只依赖标准观测 dict（state/place/steps），满足 PolicyFn 签名；
    ``expert.phase`` 暴露当前所处阶段（原语标签来源）。
    阶段转移条件全部由观测量（距离/夹爪/held）推断，与环境侧阶段机一致。
    """

    def __init__(
        self,
        gain: float = 0.35,
        noise: float = 0.02,
        action_limit: float = 0.10,
        grasp_radius: float = 0.06,
        place_radius: float = 0.08,
        gripper_threshold: float = 0.5,
        null_steps: int = 3,
        grasp_steps: int = 3,
        release_steps: int = 3,
    ):
        self.gain = gain
        self.noise = noise
        self.action_limit = action_limit
        self.grasp_radius = grasp_radius
        self.place_radius = place_radius
        self.gripper_threshold = gripper_threshold
        self.null_steps, self.grasp_steps, self.release_steps = null_steps, grasp_steps, release_steps
        self._rng = np.random.default_rng()
        self.phase = PRIM_NULL
        self._phase_step = 0

    def _reset_state(self) -> None:
        self.phase, self._phase_step = PRIM_NULL, 0

    def _advance(self, obs: dict) -> None:
        """专家侧阶段机（观测量推断，语义同任务侧 _advance_phase）。"""
        state = obs["state"]
        hand, gripper, obj, held = state[:2], float(state[2]), state[3:5], float(state[5])
        place = obs["place"]
        p, k = self.phase, self._phase_step
        if p == PRIM_NULL and k >= self.null_steps - 1:
            self.phase, self._phase_step = PRIM_APPROACH, 0
        elif p == PRIM_APPROACH and float(np.linalg.norm(hand - obj)) < self.grasp_radius:
            self.phase, self._phase_step = PRIM_GRASP, 0
        elif (
            p == PRIM_GRASP
            and held > 0.5
            and k >= self.grasp_steps - 1
        ):
            self.phase, self._phase_step = PRIM_TRANSPORT, 0
        elif p == PRIM_TRANSPORT and float(np.linalg.norm(obj - place)) < self.place_radius:
            self.phase, self._phase_step = PRIM_RELEASE, 0
        elif (
            p == PRIM_RELEASE
            and held < 0.5
            and k >= self.release_steps - 1
        ):
            self.phase, self._phase_step = PRIM_RETREAT, 0
        else:
            self._phase_step += 1
            return
        self._phase_step = 0

    def act(self, observation: dict) -> np.ndarray:
        if observation.get("steps", 0) == 0:  # episode 边界自动复位
            self._reset_state()
        self._advance(observation)
        state = observation["state"]
        hand, obj = state[:2], state[3:5]
        place = observation["place"]
        p = self.phase
        if p == PRIM_GRASP:
            # 位置保持在物体上 + 闭爪（否则噪声把手推离抓取半径，held 迟迟不触发）
            cmd = np.array([*(self.gain * (obj - hand)), 0.5])
        elif p == PRIM_RELEASE:
            cmd = np.array([*(self.gain * (place - hand)), -0.5])  # 保持在放置点 + 松爪
        else:
            # 移动阶段：比例控制；transport 持续闭紧（夹爪钉在 1.0，避免停在释放阈值
            # 附近的临界开度被噪声扰动导致中途掉物）
            target = {
                PRIM_NULL: HOME,
                PRIM_APPROACH: obj,
                PRIM_TRANSPORT: place,
                PRIM_RETREAT: HOME,
            }[p]
            d_grip = 0.3 if p == PRIM_TRANSPORT else 0.0
            cmd = np.array([*(self.gain * (target - hand)), d_grip])
        if self.noise > 0:
            cmd = cmd + self._rng.normal(0, self.noise, size=3)
        return np.clip(cmd, -self.action_limit, self.action_limit).astype(np.float32)


def collect_demo(
    task: MockPickPlaceTask,
    expert: MockPickPlaceExpert,
    seed: int | None = None,
) -> dict:
    """跑一个完整示教 episode，返回 SARM2 训练所需的全部数组。

    返回::

        {
          "states":     (T+1, 6)  每步观测（含终态），
          "primitives": (T,)      每步原语标签（专家决策时所处阶段），
          "progress":   (T,)      每步任务进度真值，
          "task_id":    int,
          "success":    bool,
        }
    """
    obs = task.reset(seed=seed)
    states, prims, progress = [obs["state"]], [], []
    done = False
    while not done:
        action = expert.act(obs)  # act 内部先推进阶段机；phase 即本步动作的原语标签
        prims.append(expert.phase)
        res = task.step(action)
        states.append(res.observation["state"])
        progress.append(res.info["progress"])
        done = res.terminated or res.truncated
        obs = res.observation
    return {
        "states": np.stack(states).astype(np.float32),
        "primitives": np.asarray(prims, dtype=np.int64),
        "progress": np.asarray(progress, dtype=np.float32),
        "task_id": task.task_id,
        "success": task.success(),
    }
