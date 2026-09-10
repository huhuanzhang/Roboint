"""π0.5 基模接入（openpi/LeRobot 生态，惰性导入）。接入路线见 README.md。"""

from robo_int.models.base import BaseVLA


class Pi05VLA(BaseVLA):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "π0.5 尚未接入：见 models/pi05/README.md 的接入步骤（先加载推理，后 SFT/LoRA）。"
        )

    def forward_actions(self, observation, instruction=None):  # pragma: no cover
        raise NotImplementedError

    def compute_loss(self, batch):  # pragma: no cover
        raise NotImplementedError
