"""Episode 数据格式与本地读写。

一个 episode = 一次完整任务尝试（示教或策略 rollout）。
存储：每 episode 一个 .npz + 一份 index.jsonl 索引（含元信息与分数），
打分/清洗只改 index 里的字段，不动物理数据——清洗可追溯、可撤销。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EpisodeMeta:
    """episode 元信息（index.jsonl 的一行）。"""

    episode_id: str
    task: str
    robot: str
    source: str = "expert"            # expert(示教) / policy(策略rollout) / teleop
    policy_version: str | None = None # 策略采集时的权重版本
    num_steps: int = 0
    success: bool | None = None       # 可能延后（人工/SARM2 打分）
    reward_sum: float | None = None
    score: float | None = None        # 数据质量分（data.scoring 填写）
    kept: bool = True                 # 清洗后是否保留（data.cleaning 填写）
    drop_reason: str | None = None
    instruction: str | None = None    # 语言指令（VLA 任务的 task prompt）
    timestamp: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)


@dataclass
class Episode:
    """内存中的 episode。数组按时间维堆叠。

    - states:         (T+1, state_dim)   每步观测状态（含 episode 末尾终态）
    - images:         {cam_name: (T+1, H, W, 3) uint8}，无图像任务为空 dict
    - actions:        (T, action_dim)    实际执行命令（安全链后），与 states[:-1] 对齐
    - policy_actions: (T, action_dim)    策略原始输出（安全链前）；safety/RTC/smoothing
                      会让两者不同，DAgger 等算法需要区分。缺省视为等于 actions
    - rewards:        (T,)，可能延后填充
    - timestamps:     (T+1,)，每拍观测的采集时间（unix 秒）
    """

    meta: EpisodeMeta
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray | None = None
    images: dict = field(default_factory=dict)
    policy_actions: np.ndarray | None = None
    timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.meta.num_steps = int(self.actions.shape[0])
        if self.rewards is not None and len(self.rewards) != len(self.actions):
            raise ValueError("rewards 与 actions 步数不一致")
        if self.policy_actions is not None and len(self.policy_actions) != len(self.actions):
            raise ValueError("policy_actions 与 actions 步数不一致")
        if self.timestamps is not None and len(self.timestamps) != len(self.states):
            raise ValueError("timestamps 与 states 步数不一致")


class EpisodeStore:
    """episode 的落盘/读取。目录结构::

        data_dir/
          index.jsonl
          episodes/ep_000001.npz
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.episode_dir = self.data_dir / "episodes"
        self.index_path = self.data_dir / "index.jsonl"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 写

    def save(self, episode: Episode) -> Path:
        path = self.episode_dir / f"{episode.meta.episode_id}.npz"
        arrays: dict = {
            "states": episode.states,
            "actions": episode.actions,
        }
        if episode.rewards is not None:
            arrays["rewards"] = episode.rewards
        if episode.policy_actions is not None:
            arrays["policy_actions"] = episode.policy_actions
        if episode.timestamps is not None:
            arrays["timestamps"] = episode.timestamps
        for cam, frames in episode.images.items():
            arrays[f"image/{cam}"] = frames
        np.savez_compressed(path, **arrays)
        self._append_index(episode.meta)
        return path

    def _append_index(self, meta: EpisodeMeta) -> None:
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ 读

    def load_index(self) -> list[EpisodeMeta]:
        if not self.index_path.exists():
            return []
        metas = []
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(EpisodeMeta(**json.loads(line)))
        return metas

    def load(self, episode_id: str) -> Episode:
        path = self.episode_dir / f"{episode_id}.npz"
        data = np.load(path)
        images = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("image/")}
        meta = next(m for m in self.load_index() if m.episode_id == episode_id)
        return Episode(
            meta=meta,
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"] if "rewards" in data.files else None,
            images=images,
            policy_actions=data["policy_actions"] if "policy_actions" in data.files else None,
            timestamps=data["timestamps"] if "timestamps" in data.files else None,
        )

    def rewrite_index(self, metas: list[EpisodeMeta]) -> None:
        """整体重写索引（打分/清洗后回写）。"""
        tmp = self.index_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
        tmp.replace(self.index_path)


def new_episode_id(existing: list[EpisodeMeta]) -> str:
    """生成下一个 episode_id（递增，带前缀）。"""
    n = len(existing)
    return f"ep_{n + 1:06d}"
