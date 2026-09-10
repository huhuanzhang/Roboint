"""Fast-WAM（World Action Model）接入 stub。路线见 README.md。"""

from robo_int.models.base import BaseVLA


class FastWAM(BaseVLA):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Fast-WAM 尚未接入：见 models/fast_wam/README.md（大概率需按论文自复现）。"
        )

    def forward_actions(self, observation, instruction=None):  # pragma: no cover
        raise NotImplementedError

    def compute_loss(self, batch):  # pragma: no cover
        raise NotImplementedError
