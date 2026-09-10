"""In-Context（示教作为上下文）挂件（主参考 RICL, arXiv:2508.02062）。"""

from robo_int.arch.base import BaseArchModule


class InContextModule(BaseArchModule):
    name = "in_context"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("in-context 挂件尚未实现：见 arch/in_context/README.md。")

    def attach(self, model) -> None:  # pragma: no cover
        raise NotImplementedError

    def detach(self) -> None:  # pragma: no cover
        raise NotImplementedError
