"""π0.5 推理接入：作为客户端复用 smolvla_piper_control_gui 的 policy server。

不重新实现 checkpoint 加载 / normalization / pre-postprocessor——那套逻辑
（processor 兼容别名、tokenizer 解析、normalizer 校验、RTC）已在
smolvla_piper_control_gui/smolvla_policy_server.py 中验证，本类只按其
msgpack WebSocket 协议发观测、收动作块（openpi-client）。

server 启动方式（与本类解耦，可由外部脚本拉起）：
    CHECKPOINT_DIR=<checkpoint> smolvla_piper_control_gui/start_smolvla_server.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from robo_int.models.base import BaseVLA

# 与 GUI client 一致的默认路径
DEFAULT_OPENPI_CLIENT_SRC = Path("/home/ps/Documents/kai0/packages/openpi-client/src")


class Pi05ServerPolicy(BaseVLA):
    """走现有 π0.5 policy server 的推理-only BaseVLA（compute_loss 不支持）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8011,
        prompt: str = "fold the clothes neatly",
        actions_per_inference: int = 8,
        state_dim: int = 14,
        action_dim: int = 14,
        openpi_client_src: str | Path = DEFAULT_OPENPI_CLIENT_SRC,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.prompt = prompt
        self.actions_per_inference = actions_per_inference
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = actions_per_inference
        self.openpi_client_src = Path(openpi_client_src)
        self._client = None
        self.server_metadata: dict | None = None

    # ------------------------------------------------------------------ 连接

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self.openpi_client_src.is_dir():
            raise FileNotFoundError(f"openpi-client 不存在：{self.openpi_client_src}")
        src = str(self.openpi_client_src.resolve())
        if src not in sys.path:
            sys.path.insert(0, src)
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
        self.server_metadata = self._client.get_server_metadata() or {}
        self._validate_server_metadata()

    def _validate_server_metadata(self) -> None:
        """连接时校验 server 契约（照抄 GUI client 的对端校验）。"""
        meta = self.server_metadata
        if meta.get("policy_io_mode", "dual-arm-14d") != "dual-arm-14d":
            raise RuntimeError(f"server policy_io_mode={meta.get('policy_io_mode')} != dual-arm-14d")
        if int(meta.get("state_dim", self.state_dim)) != self.state_dim:
            raise RuntimeError(f"server state_dim={meta.get('state_dim')} != {self.state_dim}")
        if int(meta.get("action_dim", self.action_dim)) != self.action_dim:
            raise RuntimeError(f"server action_dim={meta.get('action_dim')} != {self.action_dim}")
        server_api = int(meta.get("actions_per_inference", self.actions_per_inference))
        if server_api != self.actions_per_inference:
            raise RuntimeError(
                f"server actions_per_inference={server_api} != client {self.actions_per_inference}"
            )
        self.required_cameras = tuple(meta.get("required_policy_cameras", ()))
        if not self.required_cameras:
            raise RuntimeError("server metadata 未返回 required_policy_cameras")

    def connect(self) -> dict:
        """建立 WebSocket 连接并返回 server metadata（含 checkpoint/相机信息）。"""
        self._ensure_client()
        return dict(self.server_metadata)

    def close(self) -> None:
        if self._client is None:
            return
        ws = getattr(self._client, "_ws", None)
        close = getattr(ws, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001 — 关闭失败不应中断清理
                print(f"Warning: 关闭 policy websocket 失败: {exc}")
        self._client = None

    # ------------------------------------------------------------------ 推理

    def forward_actions(self, observation: dict, instruction: str | None = None):
        """observation 约定：{"state": (14,) float, "images": {name: (H,W,3) uint8}}。"""
        self._ensure_client()
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"期望 state ({self.state_dim},)，收到 {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("state 含 NaN/Inf")
        images = observation.get("images")
        if not isinstance(images, dict) or not images:
            raise ValueError("观测缺少 images")
        missing = [key for key in self.required_cameras if key not in images]
        if missing:
            raise ValueError(f"观测缺少 policy 相机: {missing}")

        request = {
            "active_state": state,
            "images": {key: np.asarray(images[key]) for key in self.required_cameras},
            "prompt": (instruction or self.prompt).strip(),
            "actions_per_inference": self.actions_per_inference,
        }
        result = self._client.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise RuntimeError(f"server 返回动作形状异常: {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("server 返回动作含 NaN/Inf")
        self.last_infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))
        # BaseVLA.predict_actions 按 (chunk_size, action_dim) reshape；不足时重复末步
        if actions.shape[0] < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        # msgpack 解出的是只读视图，拷贝后再暴露给 torch（与 server 端同理）
        return torch.from_numpy(np.array(actions[: self.chunk_size], copy=True))

    def compute_loss(self, batch: dict) -> dict:  # pragma: no cover — 推理-only
        raise NotImplementedError(
            "Pi05ServerPolicy 只做推理（policy server 托管 checkpoint）；"
            "训练请走 robo_int.models.pi05 的本地接入路线。"
        )
