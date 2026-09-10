"""LeRobot 数据集格式转换（惰性依赖）。

基模生态（π0.5/GR00T 微调）都吃 LeRobot dataset，
本模块负责把内部 EpisodeStore 转成 LeRobot 数据集，打通「自采数据 -> 基模 SFT」。
"""

from __future__ import annotations

from pathlib import Path


def episodes_to_lerobot(store_path: str | Path, out_path: str | Path) -> None:
    """把 EpisodeStore 转成 LeRobot 数据集。

    TODO(接入 pi05/gr00t 时实现)：
      - 依赖 lerobot 包（惰性 import，缺失时报错并给安装指引）
      - 字段映射：states -> observation.state, images -> observation.images.<cam>,
        actions -> action, instruction -> task
      - fps 对齐任务 control_hz
    """
    try:
        import lerobot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "转换 LeRobot 格式需要安装 lerobot：pip install 'lerobot'（见 pyproject [data] extra）"
        ) from e
    raise NotImplementedError("episodes_to_lerobot 将在接入 π0.5 SFT 时一并实现并测试")
