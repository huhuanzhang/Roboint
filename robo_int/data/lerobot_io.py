"""LeRobot 数据集格式转换（惰性依赖）。

基模生态（π0.5/GR00T 微调）都吃 LeRobot dataset，
本模块负责把内部 EpisodeStore 转成 LeRobot 数据集，打通「自采数据 -> 基模 SFT」。

兼容两代 lerobot API（按可用性自动选择）：
- 新版（0.5+）：``lerobot.datasets.lerobot_dataset.LeRobotDataset``
- 旧版 fork：  ``lerobot.common.datasets.lerobot_dataset.LeRobotDataset``
两者 create/add_frame/save_episode 语义一致；旧版的 feature shape 校验要求 tuple。

字段映射：states -> observation.state，images -> observation.images.<cam>，
actions（安全链后执行命令）-> action，meta.instruction -> task。
策略原始输出保留在源 EpisodeStore 的 policy_actions 里（LeRobot 端训练吃执行命令）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robo_int.data.episodes import Episode, EpisodeStore


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 新版

        return LeRobotDataset, "new"
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 旧版 fork

        return LeRobotDataset, "legacy"
    except Exception as exc:  # 新版包存在但依赖缺失等
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset, "legacy"
        except ImportError:
            raise ImportError(
                "转换 LeRobot 格式需要安装 lerobot：pip install 'lerobot'（见 pyproject [data] extra）"
            ) from exc


def _frame_features(episode: Episode, api: str) -> dict:
    """从首条 episode 推断 LeRobot feature 规格（3 相机 + state/action 14D 这类布局通用）。"""
    shape = tuple if api == "legacy" else list
    state_dim = int(episode.states.shape[-1])
    action_dim = int(episode.actions.shape[-1])
    features = {
        "observation.state": {"dtype": "float32", "shape": shape([state_dim]), "names": None},
        "action": {"dtype": "float32", "shape": shape([action_dim]), "names": None},
    }
    for cam, frames in episode.images.items():
        _, height, width, channels = frames.shape  # (T+1, H, W, 3) uint8
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": shape([channels, height, width]),
            "names": ("channels", "height", "width") if api == "legacy" else ["channels", "height", "width"],
        }
    return features


def episodes_to_lerobot(
    store_path: str | Path,
    out_path: str | Path,
    *,
    kept_only: bool = True,
    fps: int = 30,
    robot_type: str | None = None,
) -> dict:
    """把 EpisodeStore（可清洗后）转成 LeRobot 数据集。

    返回转换摘要（episode 数 / 帧数 / features / dry_run 标记）。
    时间戳说明：数据集内使用 fps 帧时钟（LeRobot 的 timestamp 同步检查假设均匀帧距），
    真实采集 wall-clock 时间保留在源 npz 的 timestamps 数组里。
    """
    LeRobotDataset, api = _import_lerobot_dataset()

    store = EpisodeStore(store_path)
    metas = store.load_index()
    if kept_only:
        metas = [m for m in metas if m.kept]
    if not metas:
        raise ValueError(f"{store.data_dir} 中没有可转换的 episode（kept_only={kept_only}）")

    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"LeRobot 输出目录已存在：{out_path}（lerobot create 要求全新目录，请删除或换一个路径）"
        )
    episodes = [store.load(m.episode_id) for m in metas]
    first = episodes[0]
    features = _frame_features(first, api)

    # 注意：root 即数据集目录本身，必须由 lerobot 创建（旧版 fork 禁止已存在）
    dataset = LeRobotDataset.create(
        repo_id=out_path.name,
        fps=fps,
        features=features,
        root=out_path,
        use_videos=False,  # 图像直存（无需视频编码依赖）
        robot_type=robot_type or first.meta.robot,
    )
    total_frames = 0
    for episode in episodes:
        task = episode.meta.instruction or "untitled"
        for t in range(episode.meta.num_steps):
            frame = {
                "observation.state": episode.states[t].astype(np.float32),
                "action": episode.actions[t].astype(np.float32),
                "task": task,
            }
            for cam in episode.images:
                frame[f"observation.images.{cam}"] = episode.images[cam][t]
            dataset.add_frame(frame)
            total_frames += 1
        dataset.save_episode()
    finalize = getattr(dataset, "finalize", None)  # 新版 API 需要显式收尾
    if callable(finalize):
        finalize()

    dry_run = [m.extra.get("dry_run") for m in metas if m.extra.get("dry_run") is not None]
    summary = {
        "episodes": len(episodes),
        "frames": total_frames,
        "features": sorted(features.keys()),
        "api": api,
        "dry_run": bool(dry_run and all(dry_run)),
        # dry-run episode 的动作未真正下发，只能作链路验证，不能进训练集
        "training_eligible": not (dry_run and all(dry_run)),
        "source": str(store.data_dir),
        "out": str(out_path),
    }
    (out_path.parent / f"{out_path.name}_notes.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
