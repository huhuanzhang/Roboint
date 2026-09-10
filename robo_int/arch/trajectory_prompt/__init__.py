"""轨迹 Prompt 挂件（用户构思方向，参考线见 README：PromptDT 等）。"""

from robo_int.arch.base import BaseArchModule


class TrajectoryPromptModule(BaseArchModule):
    name = "trajectory_prompt"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("轨迹 prompt 尚未实现：见 arch/trajectory_prompt/README.md。")

    def attach(self, model) -> None:  # pragma: no cover
        raise NotImplementedError

    def detach(self) -> None:  # pragma: no cover
        raise NotImplementedError
