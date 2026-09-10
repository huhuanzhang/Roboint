"""DSRL：扩散策略 latent-noise steering RL（用户构思：带 MEM）。"""

from robo_int.algos.base import BaseAlgo


class DSRL(BaseAlgo):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DSRL 尚未实现：见 algos/dsrl/README.md 的设计笔记与结合方案。"
        )

    def update(self, batch: dict) -> dict:  # pragma: no cover
        raise NotImplementedError
