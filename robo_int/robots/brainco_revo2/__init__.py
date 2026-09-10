"""强脑 Revo2 灵巧手客户端（SDK 待确认，惰性导入）。路线见 README.md。"""

from robo_int.robots.base import BaseRobot


class Revo2Hand(BaseRobot):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Revo2 尚未接入：见 robots/brainco_revo2/README.md（第一步确认 SDK 开放程度）。"
        )

    def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def disconnect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_state(self):  # pragma: no cover
        raise NotImplementedError

    def send_action(self, action) -> None:  # pragma: no cover
        raise NotImplementedError
