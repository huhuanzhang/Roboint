"""GR00T N1.7 基模接入（NVIDIA Isaac-GR00T，惰性导入）。接入路线见 README.md。"""

from robo_int.models.base import BaseVLA


class Gr00tN17VLA(BaseVLA):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GR00T N1.7 尚未接入：见 models/gr00t_n17/README.md（先确认 early access 依赖链）。"
        )

    def forward_actions(self, observation, instruction=None):  # pragma: no cover
        raise NotImplementedError

    def compute_loss(self, batch):  # pragma: no cover
        raise NotImplementedError
