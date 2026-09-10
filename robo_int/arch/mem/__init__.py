"""MEM 多尺度具身记忆挂件（arXiv:2603.03596）。"""

from robo_int.arch.base import BaseArchModule


class MEMModule(BaseArchModule):
    name = "mem"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MEM 尚未实现：见 arch/mem/README.md 的复现要点。")

    def attach(self, model) -> None:  # pragma: no cover
        raise NotImplementedError

    def detach(self) -> None:  # pragma: no cover
        raise NotImplementedError
