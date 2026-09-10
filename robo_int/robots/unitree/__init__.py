"""Unitree 人形机器人客户端（unitree_sdk2，惰性导入）。路线见 README.md。"""

from robo_int.robots.base import BaseRobot


class UnitreeHumanoid(BaseRobot):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Unitree 人形尚未接入：见 robots/unitree/README.md（先 SDK 连通，再安全层，再 teleop）。"
        )

    def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def disconnect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_state(self):  # pragma: no cover
        raise NotImplementedError

    def send_action(self, action) -> None:  # pragma: no cover
        raise NotImplementedError
