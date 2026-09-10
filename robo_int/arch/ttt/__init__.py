"""TTT（Test-Time Training）层挂件。"""

from robo_int.arch.base import BaseArchModule


class TTTLayer(BaseArchModule):
    name = "ttt"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TTT 层尚未实现：见 arch/ttt/README.md。")

    def attach(self, model) -> None:  # pragma: no cover
        raise NotImplementedError

    def detach(self) -> None:  # pragma: no cover
        raise NotImplementedError
