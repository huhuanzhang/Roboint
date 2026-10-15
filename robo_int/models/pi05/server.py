#!/usr/bin/env python3
"""Serve a local LeRobot SmolVLA or PI0.5 checkpoint to the Piper client.

Robo_int fork（2026-09-10）：本文件复制自已验证的 reference 实现
``smolvla_piper_control_gui/smolvla_policy_server.py``
（复制时 md5=73dbb54d…），原文件保持不动作为 parity 基准。本 fork 的增量：
- VLM prefix hidden state 的被动捕获 hook（正常推理本就计算该张量，原实现丢弃）；
- 协议可选 ``return_features``：同一请求返回 actions + features（一次 π0.5 forward）；
- metadata 增加 feature_available / feature_name / feature_dim / feature_dtype。
不请求 features 时行为与 reference 完全一致（见 tests/test_pi05_server_parity.py）。

The wire protocol intentionally matches OpenPI's msgpack-numpy WebSocket
client, so the mature camera/CAN/safety code in
``openpi_piper_dual_arm_jax_rtc_client.py`` can be reused unchanged apart from
sending the already-connected front camera.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
from safetensors import safe_open

from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

try:  # 包导入（pytest / in-process parity）
    from robo_int.models.pi05.monitor import ModelInternalMonitor
except Exception:  # 脚本直跑（python robo_int/models/pi05/server.py）
    from monitor import ModelInternalMonitor

try:
    from robo_int.models.pi05.action_query_vision import ActionQueryVisionGate
except Exception:
    from action_query_vision import ActionQueryVisionGate


LOGGER = logging.getLogger("smolvla_policy_server")
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
SUPPORTED_IMAGE_KEYS = ("front_view", "head", "left_wrist", "right_wrist")
POLICY_IO_MODES = {
    "dual-arm-14d": {
        "state_dim": 14,
        "action_dim": 14,
        "controlled_arm": "both",
        "action_semantics": "absolute Piper14 in left-then-right arm order",
    },
    "right-arm-7d": {
        "state_dim": 7,
        "action_dim": 7,
        "controlled_arm": "right",
        "action_semantics": "absolute Piper7 for the right arm",
    },
}


def pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a LeRobot SmolVLA/PI0.5 checkpoint for Piper.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-inference", type=int, default=8)
    parser.add_argument(
        "--policy-io-mode",
        choices=tuple(POLICY_IO_MODES),
        default="dual-arm-14d",
        help="State/action layout saved in the checkpoint.",
    )
    parser.add_argument("--rtc-enabled", action="store_true")
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[item.value for item in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--monitor", action="store_true",
                        help="开启 ModelInternalMonitor（只读监测，默认关闭；开关均不改变推理行为）")
    parser.add_argument("--arch-action-query-vision", action="store_true",
                        help="B 类挂件：门控 Action Expert 对视觉 prefix KV 的 value 贡献"
                             "（gate=0 时 bit 级不变；B 类挂件在 server 侧切换，客户端不感知）")
    parser.add_argument("--aqv-gate", type=float, default=0.0,
                        help="AQV 推理期固定门控（0=零效应；可学习 per-layer gate 从 0 起训）")
    parser.add_argument("--arch-mem", action="store_true",
                        help="B 类挂件：MEM 多尺度具身记忆（短时历史+长时文本库 -> 记忆 token "
                             "注入 prefix；默认 inactive=零效应，--mem-active 开启注入）")
    parser.add_argument("--mem-active", action="store_true",
                        help="开启 MEM 注入（历史与文本库皆空时仍零注入，bit 级零效应）")
    parser.add_argument("--mem-history-len", type=int, default=4,
                        help="短时记忆窗口（步，来源=过去请求已缓存的 vlm_prefix_mean）")
    parser.add_argument("--mem-tokens", type=int, default=4,
                        help="注入 prefix 的记忆 token 数")
    parser.add_argument("--arch-ttt", action="store_true",
                        help="B 类挂件：TTT-Linear 快权重层（prefix token 流上的"
                             " mini-batch 在线自监督更新 + 残差注入；默认 inactive=零效应，"
                             "--ttt-active 开启）")
    parser.add_argument("--ttt-active", action="store_true",
                        help="开启 TTT 注入（gate=0 时仍零效应，bit 级）")
    parser.add_argument("--ttt-gate", type=float, default=0.0,
                        help="TTT 残差门控（0=零效应；RoboTTT 风格从 0 起训）")
    parser.add_argument("--ttt-latent", type=int, default=128,
                        help="快权重潜在维度 K（W ∈ R^{d_model×K}）")
    parser.add_argument("--ttt-lr", type=float, default=0.1,
                        help="快权重在线更新步长（内环学习率）")
    parser.add_argument("--ttt-chunk", type=int, default=64,
                        help="mini-batch TTT 的块大小（1=逐 token 在线）")
    parser.add_argument("--ttt-persist", action="store_true",
                        help="快权重跨请求保留（session 边界仍重置；默认每请求重开）")
    return parser.parse_args()


def validate_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    missing = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete LeRobot checkpoint; missing: {', '.join(missing)}")
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        payload = json.loads((checkpoint / processor_name).read_text(encoding="utf-8"))
        for step in payload.get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (checkpoint / state_file).is_file():
                raise FileNotFoundError(
                    f"Incomplete LeRobot checkpoint; {processor_name} references missing {state_file}"
                )


def load_policy_type(checkpoint: Path) -> str:
    policy_type = str(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get("type", ""))
    if policy_type not in {"smolvla", "pi05"}:
        raise ValueError(f"Unsupported LeRobot policy type: {policy_type!r}")
    return policy_type


def install_processor_compatibility_aliases() -> None:
    """Bridge processor registry names used by the H800 training branch."""
    try:
        ProcessorStepRegistry.get("relative_actions_processor")
    except KeyError:
        # This runtime branch registers the same class as
        # ``delta_actions_processor``.  Keep the checkpoint immutable and add
        # the historical serialized name as an in-memory alias.
        ProcessorStepRegistry._registry["relative_actions_processor"] = RelativeActionsProcessorStep


def resolve_tokenizer_name(checkpoint: Path) -> str:
    """Prefer a bundled tokenizer, otherwise reuse the checkpoint's model tokenizer."""
    local_tokenizer = checkpoint / "tokenizer"
    if (local_tokenizer / "tokenizer.json").is_file():
        return str(local_tokenizer)

    payload = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    for step in payload.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        tokenizer_name = step.get("config", {}).get("tokenizer_name")
        if isinstance(tokenizer_name, str) and tokenizer_name and tokenizer_name != "tokenizer":
            return tokenizer_name
    raise FileNotFoundError(
        f"Checkpoint {checkpoint} has no bundled tokenizer and no reusable tokenizer_name"
    )


def load_checkpoint_image_keys(checkpoint: Path) -> tuple[str, ...]:
    """Read camera names from rename_map, or normalizer features as a fallback."""
    payload = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    rename_map = None
    for step in payload.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            rename_map = step.get("config", {}).get("rename_map")
            break

    camera_pairs = []
    prefix = "observation.images."
    if isinstance(rename_map, dict):
        for source, target in rename_map.items():
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if not source.startswith(prefix) or not target.startswith(prefix):
                continue
            source_key = source.removeprefix(prefix)
            if source_key not in SUPPORTED_IMAGE_KEYS:
                raise ValueError(f"Checkpoint requires unsupported camera source: {source_key}")
            camera_pairs.append((target, source_key))

    if camera_pairs:
        return tuple(source for _, source in sorted(camera_pairs))

    # Some checkpoints have an empty rename_map but record visual inputs in
    # normalizer features. Use those names without editing the artifact.
    normalizer_features = None
    for step in payload.get("steps", []):
        if step.get("registry_name") == "normalizer_processor":
            normalizer_features = step.get("config", {}).get("features")
            break
    if isinstance(normalizer_features, dict):
        for feature_name in normalizer_features:
            if not isinstance(feature_name, str) or not feature_name.startswith(prefix):
                continue
            source_key = feature_name.removeprefix(prefix)
            if source_key not in SUPPORTED_IMAGE_KEYS:
                raise ValueError(f"Checkpoint requires unsupported camera source: {source_key}")
            camera_pairs.append((feature_name, source_key))

    if camera_pairs:
        return tuple(source for _, source in camera_pairs)
    raise ValueError("Checkpoint preprocessor has no observation rename map or visual features")

def load_compatible_config(checkpoint: Path, device: str) -> PreTrainedConfig:
    """Decode checkpoints produced by the training fork without editing them.

    The artifact contains ``pretrained_revision=null``. It is provenance-only
    and absent from upstream LeRobot 0.5.2's dataclass, so it is ignored here.
    All architecture-affecting fields remain strictly decoded.
    """

    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    raw.pop("pretrained_revision", None)
    config = draccus.decode(PreTrainedConfig, raw)
    config.device = device
    if hasattr(config, "compile_model"):
        config.compile_model = False
    return config


def load_normalizer_feature_dim(checkpoint: Path, feature: str) -> int:
    """Return the dataset feature width saved in the normalizer state.

    Some older training forks saved stale feature shapes in config.json while
    keeping the correct statistics tensors. The tensor is authoritative because
    it is what preprocessing actually uses during inference.
    """

    payload = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    normalizer = next(
        (step for step in payload.get("steps", []) if step.get("registry_name") == "normalizer_processor"),
        None,
    )
    if normalizer is None or not normalizer.get("state_file"):
        raise ValueError("Checkpoint preprocessor has no normalizer state file")
    state_path = checkpoint / normalizer["state_file"]
    tensor_name = f"{feature}.mean"
    with safe_open(state_path, framework="pt", device="cpu") as tensors:
        if tensor_name not in tensors.keys():
            raise ValueError(f"Checkpoint normalizer has no {tensor_name} tensor")
        shape = tuple(tensors.get_tensor(tensor_name).shape)
    if len(shape) != 1:
        raise ValueError(f"Checkpoint normalizer {tensor_name} must be rank 1, got {shape}")
    return int(shape[0])


def validate_normalization_files(checkpoint: Path) -> dict[str, Any]:
    """Validate and describe the exact normalization artifacts used at runtime."""
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    postprocessor_path = checkpoint / "policy_postprocessor.json"
    preprocessor_payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    postprocessor_payload = json.loads(postprocessor_path.read_text(encoding="utf-8"))

    normalizer = next(
        (step for step in preprocessor_payload.get("steps", []) if step.get("registry_name") == "normalizer_processor"),
        None,
    )
    unnormalizer = next(
        (step for step in postprocessor_payload.get("steps", []) if step.get("registry_name") == "unnormalizer_processor"),
        None,
    )
    if normalizer is None or unnormalizer is None:
        raise ValueError("Checkpoint must define both normalizer and unnormalizer processors")

    pre_state_name = normalizer.get("state_file")
    post_state_name = unnormalizer.get("state_file")
    if not pre_state_name or not post_state_name:
        raise ValueError("Normalization processors must reference their safetensors state files")
    pre_state_path = checkpoint / pre_state_name
    post_state_path = checkpoint / post_state_name

    norm_map = normalizer.get("config", {}).get("norm_map", {})
    post_norm_map = unnormalizer.get("config", {}).get("norm_map", {})
    state_method = norm_map.get("STATE")
    action_method = norm_map.get("ACTION")
    if state_method not in {"MEAN_STD", "QUANTILES"}:
        raise ValueError(f"Unsupported state normalization method: {state_method}")
    if action_method not in {"MEAN_STD", "QUANTILES"}:
        raise ValueError(f"Unsupported action normalization method: {action_method}")
    if post_norm_map.get("ACTION") != action_method:
        raise ValueError(
            f"Action normalizer/unnormalizer methods differ: {action_method} != {post_norm_map.get('ACTION')}"
        )

    state_suffixes = ("mean", "std") if state_method == "MEAN_STD" else ("q01", "q99")
    action_suffixes = ("mean", "std") if action_method == "MEAN_STD" else ("q01", "q99")
    required_pre = tuple(f"observation.state.{name}" for name in state_suffixes) + tuple(
        f"action.{name}" for name in action_suffixes
    )
    required_post = tuple(f"action.{name}" for name in action_suffixes)
    with safe_open(pre_state_path, framework="pt", device="cpu") as tensors:
        missing = [name for name in required_pre if name not in tensors.keys()]
        if missing:
            raise ValueError(f"Normalizer state is missing tensors: {missing}")
        state_tensors = {
            name: tensors.get_tensor(f"observation.state.{name}") for name in state_suffixes
        }
        action_tensors = {name: tensors.get_tensor(f"action.{name}") for name in action_suffixes}
        sample_count = float(tensors.get_tensor("observation.state.count").reshape(-1)[0])
    with safe_open(post_state_path, framework="pt", device="cpu") as tensors:
        missing = [name for name in required_post if name not in tensors.keys()]
        if missing:
            raise ValueError(f"Unnormalizer state is missing tensors: {missing}")
        post_action_tensors = {name: tensors.get_tensor(f"action.{name}") for name in action_suffixes}

    for name, tensor in {
        **{f"observation.state.{key}": value for key, value in state_tensors.items()},
        **{f"action.{key}": value for key, value in action_tensors.items()},
    }.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Normalization tensor {name} contains NaN or Inf")
    if state_method == "MEAN_STD" and torch.any(state_tensors["std"] < 0):
        raise ValueError("State normalization standard deviations must not be negative")
    if action_method == "MEAN_STD" and torch.any(action_tensors["std"] < 0):
        raise ValueError("Action normalization standard deviations must not be negative")
    if state_method == "QUANTILES" and torch.any(state_tensors["q99"] < state_tensors["q01"]):
        raise ValueError("State q99 must not be less than q01")
    if action_method == "QUANTILES" and torch.any(action_tensors["q99"] < action_tensors["q01"]):
        raise ValueError("Action q99 must not be less than q01")
    if any(not torch.equal(action_tensors[name], post_action_tensors[name]) for name in action_suffixes):
        raise ValueError("Action normalizer and unnormalizer statistics do not match")

    return {
        "enabled": True,
        "state_method": state_method,
        "action_method": action_method,
        "preprocessor_state_file": pre_state_name,
        "postprocessor_state_file": post_state_name,
        "state_dim": int(next(iter(state_tensors.values())).numel()),
        "action_dim": int(next(iter(action_tensors.values())).numel()),
        "sample_count": int(sample_count),
        "round_trip_stats_match": True,
    }


def image_to_policy_tensor(image: Any, size: int | None) -> torch.Tensor:
    # msgpack-numpy decodes a read-only view. Own the memory before exposing it
    # to torch, which otherwise warns that writes would be undefined behavior.
    array = np.array(image, copy=True)
    if array.ndim != 3:
        raise ValueError(f"Camera image must be rank 3, got {array.shape}")
    if array.shape[-1] in (3, 4):
        array = array[..., :3]
    elif array.shape[0] in (3, 4):
        array = np.moveaxis(array[:3], 0, -1)
    else:
        raise ValueError(f"Camera image must have 3 RGB channels, got {array.shape}")
    if size is not None and array.shape[:2] != (size, size):
        array = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)
    array = np.ascontiguousarray(array)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(torch.float32)
    if array.dtype == np.uint8 or float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp_(0.0, 1.0)


@dataclass
class RTCRequest:
    prefix: torch.Tensor | None = None
    prefix_start_index: int = 0
    prefix_steps: int = 0
    inference_delay: int = 0
    execution_horizon: int = 0


class RTCSession:
    """Server-side mirror of the client's currently executing normalized chunk."""

    def __init__(self, config: RTCConfig):
        self.queue = ActionQueue(config)
        self.chunk_sequence = 0

    def prepare_request(self, context: dict[str, Any] | None, chunk_size: int) -> RTCRequest:
        if not context:
            return RTCRequest()
        if self.queue.original_queue is None:
            raise ValueError("RTC context was received before this connection produced its first chunk")

        prefix_start = int(context.get("previous_action_start_index", 0))
        if not 0 <= prefix_start < chunk_size:
            raise ValueError(f"RTC previous_action_start_index must be within [0, {chunk_size}), got {prefix_start}")
        current_index = self.queue.get_action_index()
        if prefix_start < current_index:
            raise ValueError(
                f"RTC cursor moved backwards: server={current_index}, client={prefix_start}"
            )
        while self.queue.get_action_index() < prefix_start:
            if self.queue.get() is None:
                raise ValueError(f"RTC cursor {prefix_start} exceeds the server-side action queue")

        prefix = self.queue.get_left_over()
        if prefix is None or len(prefix) == 0:
            raise ValueError("RTC prefix is empty after synchronizing the client cursor")
        delay = max(0, int(context.get("delay_steps", 0)))
        delay = min(delay, len(prefix), chunk_size - 1)
        requested_horizon = max(1, int(context.get("execution_horizon", self.queue.cfg.execution_horizon)))
        horizon = min(requested_horizon, self.queue.cfg.execution_horizon, len(prefix), chunk_size)
        return RTCRequest(
            prefix=prefix.unsqueeze(0),
            prefix_start_index=prefix_start,
            prefix_steps=len(prefix),
            inference_delay=delay,
            execution_horizon=horizon,
        )

    def store_chunk(self, normalized_actions: torch.Tensor) -> None:
        chunk = normalized_actions[0].detach()
        # The client owns execution and reports its absolute chunk cursor on the
        # next request. Keep the complete normalized chunk here, so that cursor
        # indexes stay identical on both sides of the WebSocket.
        self.queue.merge(chunk, chunk, real_delay=0)
        self.chunk_sequence += 1


class SmolVLAPiperPolicy:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        actions_per_inference: int,
        *,
        policy_io_mode: str = "dual-arm-14d",
        rtc_enabled: bool = False,
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
        rtc_prefix_attention_schedule: str = RTCAttentionSchedule.EXP.value,
        monitor_enabled: bool = False,
        arch_action_query_vision: bool = False,
        aqv_gate: float = 0.0,
        arch_mem: bool = False,
        mem_active: bool = False,
        mem_history_len: int = 4,
        mem_tokens: int = 4,
        arch_ttt: bool = False,
        ttt_active: bool = False,
        ttt_gate: float = 0.0,
        ttt_latent: int = 128,
        ttt_lr: float = 0.1,
        ttt_chunk: int = 64,
        ttt_persist: bool = False,
    ):
        validate_checkpoint(checkpoint)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if actions_per_inference < 1:
            raise ValueError("actions_per_inference must be positive")

        self.checkpoint = checkpoint
        self.policy_type = load_policy_type(checkpoint)
        self.rtc_supported = self.policy_type in {"smolvla", "pi05"}
        if rtc_enabled and not self.rtc_supported:
            raise ValueError(f"RTC is not enabled for policy type {self.policy_type}")
        self.device = device
        self.actions_per_inference = actions_per_inference
        self.policy_io_mode = policy_io_mode
        io_spec = POLICY_IO_MODES[policy_io_mode]
        self.state_dim = int(io_spec["state_dim"])
        self.action_dim = int(io_spec["action_dim"])
        self.controlled_arm = str(io_spec["controlled_arm"])
        self.action_semantics = str(io_spec["action_semantics"])
        self.expected_image_keys = load_checkpoint_image_keys(checkpoint)
        self.config = load_compatible_config(checkpoint, device)
        checkpoint_action_dim = int(self.config.output_features["action"].shape[0])
        normalized_state_dim = load_normalizer_feature_dim(checkpoint, "observation.state")
        normalized_action_dim = load_normalizer_feature_dim(checkpoint, "action")
        if (
            normalized_state_dim != self.state_dim
            or normalized_action_dim != self.action_dim
            or checkpoint_action_dim != self.action_dim
        ):
            raise ValueError(
                f"policy_io_mode={self.policy_io_mode} requires {self.state_dim}D state / "
                f"{self.action_dim}D action, but checkpoint contains "
                f"{normalized_state_dim}D normalized state / {normalized_action_dim}D normalized action "
                f"and {checkpoint_action_dim}D configured action"
            )
        self.config.rtc_config = RTCConfig(
            enabled=rtc_enabled,
            execution_horizon=rtc_execution_horizon,
            max_guidance_weight=rtc_max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule(rtc_prefix_attention_schedule),
            debug=False,
        )

        self.normalization_metadata = validate_normalization_files(checkpoint)

        install_processor_compatibility_aliases()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {
                    "tokenizer_name": resolve_tokenizer_name(checkpoint)
                },
            },
        )
        policy_class = SmolVLAPolicy if self.policy_type == "smolvla" else PI05Policy
        self.policy = policy_class.from_pretrained(
            checkpoint,
            config=self.config,
            strict=True,
        )
        self.policy.eval()

        if self.actions_per_inference > self.config.chunk_size:
            raise ValueError(
                f"actions_per_inference={self.actions_per_inference} exceeds chunk_size={self.config.chunk_size}"
            )

        # ------------------------------------------------------------------
        # Robo_int fork：VLM prefix 特征捕获（被动包裹，不改前向计算）
        # sample_actions 的 prefix pass 计算 paligemma language_model 的
        # last_hidden_state 后丢弃（只留 KV cache）——这里把这份「视觉+语言
        # 上下文表征」抄下来：固定长度、单次正常推理已产生。
        # 注意不能用 register_forward_hook：molmoact2 直接调
        # language_model.forward(...)，直接 .forward() 绕过 nn.Module.__call__
        # 的 hook 机制，故以同签名包裹 forward、原样返回 output。
        # feature = 对 prefix 全部 token 的 hidden state 取均值 (hidden_size,)。
        self.feature_name: str | None = None
        self.feature_dim: int = 0
        self._last_prefix_hidden = None
        self._restore_vlm_forward = None
        # ------------------------------------------------------------------
        # ModelInternalMonitor：B 类 model-internal 模块的统一只读监测。
        # 默认关闭；开启也不改前向/动作（数值 parity 由 parity 测试守护）。
        # 关闭时除 feature capture 包裹（特征通道必需）外零额外接线。
        self.monitor = ModelInternalMonitor(enabled=monitor_enabled)
        # aqv/mem 须在 _install_feature_capture 的 warmup infer() 之前初始化
        self.aqv: ActionQueryVisionGate | None = None
        self.mem = None  # MEMModule；infer()/new_session() 会访问
        self.ttt = None  # TTTLinearLayer；infer()/new_session() 会访问
        self.monitor.register_hook(
            "vlm_prefix_capture",
            hook_point="paligemma_with_expert.paligemma.model.language_model.forward(wrap)",
            source_module="PI05Model.paligemma_with_expert.paligemma.model.language_model",
            source_tensor="last_hidden_state",
            enabled=True,
        )
        self.monitor.register_tensor("vlm_prefix_mean")
        if self.policy_type == "pi05":
            self._install_feature_capture()
            self.monitor.register_parameters(
                "base_vlm", self.policy.model.paligemma_with_expert.paligemma.model.language_model
            )
            self.monitor.register_parameters(
                "action_expert", self.policy.model.paligemma_with_expert.gemma_expert
            )
            if self.monitor.enabled:
                # 参数 snapshot 基准（以后验证 LoRA/TTT 是否改动 VLM/Expert）
                self.monitor.snapshot_parameters("base_vlm")
                self.monitor.snapshot_parameters("action_expert")
                self._install_expert_timing_wrap()

        # ------------------------------------------------------------------
        # B 类挂件：Action-query Vision Attention Gate（gate=0 零效应）
        if arch_action_query_vision and self.policy_type == "pi05":
            self._install_action_query_vision(aqv_gate)

        # ------------------------------------------------------------------
        # B 类挂件：MEM 多尺度具身记忆（inactive 时零效应）
        if arch_mem and self.policy_type == "pi05":
            self._install_memory(mem_active, mem_history_len, mem_tokens)

        # ------------------------------------------------------------------
        # B 类挂件：TTT-Linear 快权重层（gate=0 / inactive 零效应）
        if arch_ttt and self.policy_type == "pi05":
            self._install_ttt(ttt_active, ttt_gate, ttt_latent, ttt_lr,
                              ttt_chunk, ttt_persist)

    def _install_action_query_vision(self, fixed_gate: float) -> None:
        """门控 Action Expert 对视觉 prefix KV 的 value 贡献。

        视觉 token 数 = 实测 prefix_len − checkpoint tokenizer max_length（不猜）；
        若不能被相机数整除则只做整体视觉门控，per-camera 统计如实标 unmapped。
        """
        pwe = self.policy.model.paligemma_with_expert
        num_layers = int(pwe.paligemma.config.text_config.num_hidden_layers)
        prefix_len = int(getattr(self, "_prefix_len", 0))
        lang_len = 200
        try:
            payload = json.loads((self.checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
            for step in payload.get("steps", []):
                if step.get("registry_name") == "tokenizer_processor":
                    lang_len = int(step.get("config", {}).get("max_length", lang_len))
                    break
        except (OSError, ValueError):
            pass
        camera_names = list(self.policy.config.image_features)
        vision_tokens = prefix_len - lang_len
        if vision_tokens <= 0:
            raise ValueError(f"视觉 token 数异常：prefix_len={prefix_len} lang_len={lang_len}")
        if vision_tokens % max(len(camera_names), 1) != 0:
            print(f"Warning: camera token mapping unavailable（prefix_len={prefix_len} "
                  f"lang={lang_len} cams={len(camera_names)}），AQV 仅整体视觉门控", flush=True)
            slice_cameras: list[str] = []
        else:
            slice_cameras = camera_names
        self.aqv = ActionQueryVisionGate(
            pwe, num_layers=num_layers, vision_tokens=vision_tokens,
            fixed_gate=fixed_gate, camera_names=slice_cameras,
        )
        self.aqv.attach()
        self.monitor.register_hook(
            "aqv_gate",
            hook_point="paligemma_with_expert.forward(prefix-only)->past_key_values",
            source_module="PaliGemmaWithExpert KV cache",
            source_tensor="past_key_values[layer][1][..., :vision_tokens, :]",
            enabled=True,
        )
        self.monitor.register_tensor("aqv_gate_values")
        for cam in slice_cameras:
            self.monitor.register_tensor(f"aqv_v_mass_{cam}")
        self.monitor.register_tensor("aqv_v_mass_language")
        print(
            f"ActionQueryVisionGate active: layers={num_layers} vision_tokens={vision_tokens} "
            f"fixed_gate={fixed_gate} cameras={slice_cameras or 'unmapped'}",
            flush=True,
        )

    def _install_memory(self, active: bool, history_len: int, num_tokens: int) -> None:
        """MEM：短时历史 + 长时文本库 -> 记忆 token 注入 embed_prefix 输出末尾。

        注入点=pi05.model.embed_prefix（sample_actions 与训练 forward 都经该属性
        查找）；pad=1/att=0 追加，prefix pass 与 Action Expert denoise 步经既有
        mask 机制自然 attend 记忆 token，不改 denoise/mask 代码。
        active=False 或无可记忆内容时原对象原样返回（bit 级零效应）。
        """
        try:
            from robo_int.arch.mem import MEMModule
        except Exception as exc:
            raise ImportError("MEM 需要 robo_int 包可导入（robo_int.arch.mem）") from exc
        d_model = int(self.feature_dim) or 2048
        self.mem = MEMModule(d_model=d_model, history_len=history_len,
                             num_tokens=num_tokens, bank_capacity=16)
        self.mem.encode_text = self._encode_memory_text
        self.mem.to(self.device)
        self.mem.attach(self.policy.model)
        self.mem.active = bool(active)
        self.monitor.register_hook(
            "mem_inject",
            hook_point="pi05.model.embed_prefix(wrap)->prefix 追加记忆 token",
            source_module="robo_int.arch.mem.MEMModule",
            source_tensor="compute_tokens()->(1,num_tokens,d_model)",
            enabled=True,
        )
        self.monitor.register_tensor("mem_tokens")
        self.monitor.register_parameters("mem_module", self.mem)
        if self.monitor.enabled:
            self.monitor.snapshot_parameters("mem_module")
        print(
            f"MEM installed: d_model={d_model} history_len={history_len} "
            f"num_tokens={num_tokens} active={self.mem.active} "
            f"feature_source=vlm_prefix_mean(no robot state)",
            flush=True,
        )

    def _install_ttt(self, active: bool, gate: float, latent_dim: int, lr: float,
                     chunk_size: int, persist: bool) -> None:
        """TTT-Linear：prefix token 流上的 mini-batch 快权重序列变换（残差注入）。

        注入点=pi05.model.embed_prefix（与 MEM 同面，可叠加）；gate=0 或
        inactive 时 embs 原对象原样返回（bit 级零效应）。fast weights 默认每
        请求从 fast_init 重开（per-sequence 语义）；--ttt-persist 跨请求保留
        （RoboTTT 上下文扩展风味），new_session() 边界仍重置。
        """
        try:
            from robo_int.arch.ttt import TTTLinearLayer
        except Exception as exc:
            raise ImportError("TTT 需要 robo_int 包可导入（robo_int.arch.ttt）") from exc
        d_model = int(self.feature_dim) or 2048
        self.ttt = TTTLinearLayer(d_model=d_model, latent_dim=latent_dim, lr=lr,
                                  gate=gate, chunk_size=chunk_size, persist=persist)
        self.ttt.to(self.device)
        self.ttt.attach(self.policy.model)
        self.ttt.active = bool(active)
        self.monitor.register_hook(
            "ttt_apply",
            hook_point="pi05.model.embed_prefix(wrap)->prefix token 残差变换",
            source_module="robo_int.arch.ttt.TTTLinearLayer",
            source_tensor="forward_tokens(embs)->(B,L,d_model)",
            enabled=True,
        )
        self.monitor.register_tensor("ttt_out")
        self.monitor.register_parameters("ttt_module", self.ttt)
        if self.monitor.enabled:
            self.monitor.snapshot_parameters("ttt_module")
        print(
            f"TTT installed: d_model={d_model} latent={latent_dim} lr={lr} "
            f"chunk={chunk_size} gate={gate} active={self.ttt.active} "
            f"persist={persist}",
            flush=True,
        )

    def _encode_memory_text(self, text: str) -> "np.ndarray":
        """文本 -> checkpoint tokenizer + 冻结 embed 层 masked-mean 嵌入 (2048,)。

        只做 embedding 查表（无 transformer forward，不计入 forward 计数），
        与推理 prompt 走同一 preprocessor/tokenizer。
        """
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        obs = self._build_policy_observation({
            "active_state": np.zeros((self.state_dim,), dtype=np.float32),
            "images": {k: np.zeros((480, 640, 3), dtype=np.uint8)
                       for k in self.expected_image_keys},
            "prompt": text,
            "actions_per_inference": 1,
        })
        policy_obs = self.preprocessor(obs)
        tokens = policy_obs[f"{OBS_LANGUAGE_TOKENS}"]
        mask = policy_obs[f"{OBS_LANGUAGE_ATTENTION_MASK}"].bool()
        embed = self.policy.model.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings()
        with torch.no_grad():
            emb = embed(tokens.to(embed.weight.device)).float()
            m = mask.to(emb.device).unsqueeze(-1).float()
            pooled = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled[0].cpu().numpy()

    def _install_expert_timing_wrap(self) -> None:
        """action expert 前向计时包裹（仅 monitor enabled 时安装；也是直接
        .forward() 调用，register_forward_hook 不适用，同 VLM 包裹的理由）。"""
        expert = self.policy.model.paligemma_with_expert.gemma_expert.model
        original_forward = expert.forward

        def timed_forward(*args, **kwargs):
            import time as _time

            t0 = _time.perf_counter()
            output = original_forward(*args, **kwargs)
            self.monitor.add_stage_time("action_expert", _time.perf_counter() - t0)
            return output

        expert.forward = timed_forward

    def _install_feature_capture(self) -> None:
        vlm = self.policy.model.paligemma_with_expert.paligemma.model.language_model
        original_forward = vlm.forward
        holder = self

        def wrapped_forward(*args, **kwargs):
            t0 = time.perf_counter()
            output = original_forward(*args, **kwargs)
            holder.monitor.add_stage_time("vlm", time.perf_counter() - t0)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, tuple):
                hidden = output[0]
            if hidden is not None:
                holder._last_prefix_hidden = hidden.detach()
                holder.monitor.record_hook("vlm_prefix_capture", hidden)
            return output

        vlm.forward = wrapped_forward
        self._restore_vlm_forward = original_forward
        self.feature_name = "vlm_prefix_mean"
        # 启动期一次 dummy 前向确定 feature_dim（顺带预热；hidden 大小不在序列化 config 里）
        state = np.zeros((self.state_dim,), dtype=np.float32)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = {
            "active_state": state,
            "images": {key: image for key in self.expected_image_keys},
            "prompt": "warmup",
            "actions_per_inference": 1,
        }
        self.infer(observation, self.new_session())
        captured = self._last_prefix_hidden
        self._last_prefix_hidden = None  # warmup 的捕获不外泄
        if captured is not None:
            self.feature_dim = int(captured.shape[-1])
            self._prefix_len = int(captured.shape[1])
            print(
                f"Feature capture active: name={self.feature_name} "
                f"dim={self.feature_dim} prefix_len={captured.shape[1]}",
                flush=True,
            )
        else:
            print("Warning: feature capture 未在 warmup 中捕获到 prefix hidden", flush=True)

    def _extract_feature(self) -> np.ndarray | None:
        """最近一次 forward 捕获的 prefix hidden -> 均值池化 (feature_dim,)。"""
        hidden = self._last_prefix_hidden
        if hidden is None:
            return None
        return hidden.to(torch.float32).mean(dim=1)[0].cpu().numpy()

    @property
    def rtc_enabled(self) -> bool:
        return bool(self.config.rtc_config and self.config.rtc_config.enabled)

    def new_session(self) -> RTCSession:
        # 连接=episode 边界：清短时记忆；长时文本库保留（跨 episode 语义）
        if self.mem is not None:
            self.mem.reset_episode()
        if self.ttt is not None:
            self.ttt.reset_episode()  # episode 边界：清持久快权重
        return RTCSession(self.config.rtc_config)

    def _build_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(observation.get("active_state"), dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected active_state shape ({self.state_dim},), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("active_state contains NaN or Inf")

        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("Observation is missing the images dictionary")
        missing = [key for key in self.expected_image_keys if key not in images]
        if missing:
            raise ValueError(f"Observation is missing policy cameras: {missing}")

        prompt = observation.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Observation prompt must be a non-empty string")

        policy_observation = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": prompt.strip(),
        }
        for key in self.expected_image_keys:
            # Preserve the native 640x480 camera geometry used by training.
            # SmolVLA and PI0.5 both apply their own aspect-ratio-preserving
            # resize-with-padding inside the policy.
            policy_observation[f"observation.images.{key}"] = image_to_policy_tensor(
                images[key], None
            )
        return policy_observation

    # RTCProcessor temporarily enables autograd to compute its flow-guidance
    # correction. torch.inference_mode() cannot be overridden inside that
    # block; no_grad() can, while still avoiding parameter-gradient storage.
    @torch.no_grad()
    def infer(
        self,
        observation: dict[str, Any],
        session: RTCSession | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self.monitor.begin_inference(request_id=observation.get("request_id"))
        want_features = bool(observation.get("return_features"))
        want_monitor = bool(observation.get("return_monitor_report"))
        if want_features:
            self._last_prefix_hidden = None  # 保证捕获的是本次 forward
        preprocess_t0 = time.perf_counter()
        policy_observation = self.preprocessor(self._build_policy_observation(observation))
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0
        rtc_request = RTCRequest()
        predict_kwargs: dict[str, Any] = {}
        if self.rtc_enabled:
            if session is None:
                raise ValueError("RTC inference requires a per-connection RTCSession")
            rtc_request = session.prepare_request(observation.get("rtc_context"), self.config.chunk_size)
            predict_kwargs = {
                "prev_chunk_left_over": rtc_request.prefix,
                "inference_delay": rtc_request.inference_delay,
                "execution_horizon": rtc_request.execution_horizon or self.config.rtc_config.execution_horizon,
            }
        predict_t0 = time.perf_counter()
        normalized_actions = self.policy.predict_action_chunk(
            policy_observation,
            noise=noise,
            **predict_kwargs,
        )
        predict_ms = (time.perf_counter() - predict_t0) * 1000.0
        # flow 采样编排 = predict 墙钟 − 内部已计时的 vlm/expert 前向
        flow_sampling_ms = predict_ms - self.monitor._stage_buffers["vlm"] * 1000.0 \
            - self.monitor._stage_buffers["action_expert"] * 1000.0
        if session is not None:
            session.store_chunk(normalized_actions)
        post_t0 = time.perf_counter()
        actions = self.postprocessor(normalized_actions)
        postprocess_ms = (time.perf_counter() - post_t0) * 1000.0
        if isinstance(actions, dict):
            actions = actions["action"]
        actions = np.asarray(actions.detach().cpu(), dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != self.action_dim:
            raise RuntimeError(f"Unexpected {self.policy_type} action chunk shape: {actions.shape}")
        full_actions = actions[0]
        if not np.isfinite(full_actions).all():
            raise RuntimeError(f"{self.policy_type} produced NaN or Inf actions")

        requested = int(observation.get("actions_per_inference", self.actions_per_inference))
        requested = min(max(1, requested), self.actions_per_inference, full_actions.shape[0])
        # RTC actions before inference_delay are intentionally preserved as the
        # continuity prefix. Return that prefix *plus* the requested number of
        # fresh executable actions so the client does not lose horizon after
        # skipping stale timesteps.
        returned_steps = requested
        if self.rtc_enabled:
            returned_steps = min(full_actions.shape[0], rtc_request.inference_delay + requested)
        infer_ms = (time.perf_counter() - started) * 1000.0
        # ---- MEM 功能状态：本步特征推入短时历史；客户端可带 mem_texts 写长时库 ----
        # （feature=过去请求已缓存的 vlm_prefix_mean，MEM 不为当前观测多跑 forward）
        if self.mem is not None:
            feature = self._extract_feature()
            if feature is not None:
                self.mem.push_history(torch.from_numpy(feature).to(self.device))
                mem_texts = observation.get("mem_texts") or []
                if isinstance(mem_texts, (list, tuple)):
                    for text in mem_texts:
                        if isinstance(text, str) and text.strip():
                            self.mem.write_memory(text, feature)

        # ---- ModelInternalMonitor 收尾：计数/耗时/显存/张量统计/动作影响 ----
        # 统计类计算只在 enabled 时做（关闭零开销）；无 B 类 modifier 时
        # base==modified（同一份动作），有 modifier 后在此之前调 record_impact 即可。
        if self.monitor.enabled:
            feature = self._extract_feature() if self._last_prefix_hidden is not None else None
            if feature is not None:
                self.monitor.record_tensor("vlm_prefix_mean", torch.from_numpy(feature))
            if self.aqv is not None:
                gates = self.aqv.effective_gates()
                self.monitor.record_hook("aqv_gate", gates)
                self.monitor.record_tensor("aqv_gate_values", gates)
                for cam, mass in (self.aqv.last_stats.get("camera_v_mass") or {}).items():
                    key = f"aqv_v_mass_{cam}"
                    if key in self.monitor._tensors:
                        self.monitor.record_tensor(key, torch.tensor([mass]))
            if self.mem is not None:
                mem_stats = self.mem.last_stats
                self.monitor.record_hook(
                    "mem_inject",
                    torch.tensor([1.0 if mem_stats.get("injected") else 0.0]),
                )
                if mem_stats.get("injected"):
                    self.monitor.record_tensor("mem_tokens", torch.tensor([
                        mem_stats.get("token_l2_norm", 0.0),
                        mem_stats.get("history_len", 0),
                        mem_stats.get("bank_size", 0),
                        mem_stats.get("num_tokens", 0),
                    ], dtype=torch.float32))
            if self.ttt is not None:
                ttt_stats = self.ttt.last_stats
                self.monitor.record_hook(
                    "ttt_apply",
                    torch.tensor([1.0 if ttt_stats.get("applied") else 0.0]),
                )
                if ttt_stats.get("applied"):
                    self.monitor.record_tensor("ttt_out", torch.tensor([
                        ttt_stats.get("out_delta_l2", 0.0),
                        ttt_stats.get("w_delta_norm", 0.0),
                        ttt_stats.get("w_end_norm", 0.0),
                        ttt_stats.get("recon_per_token", 0.0),
                    ], dtype=torch.float32))
            modifiers = bool(
                (self.aqv is not None and bool((self.aqv.effective_gates() != 0).any()))
                or (self.mem is not None and self.mem.active
                    and bool(self.mem.last_stats.get("injected")))
                or (self.ttt is not None and self.ttt.active
                    and bool(self.ttt.last_stats.get("applied")))
            )
            self.monitor.record_impact(full_actions, full_actions, modifiers_active=modifiers)
        self.monitor.end_inference(
            preprocess_ms=preprocess_ms, total_ms=infer_ms,
            flow_sampling_ms=flow_sampling_ms, postprocess_ms=postprocess_ms,
            device=self.device,
        )
        result = {
            "actions": full_actions[:returned_steps],
            "full_actions": full_actions,
            "request_id": observation.get("request_id"),
            "policy_timing": {"infer_ms": infer_ms},
            "rtc": {
                "rtc_enabled": self.rtc_enabled,
                "rtc_applied": rtc_request.prefix is not None,
                "rtc_skip_steps": rtc_request.inference_delay,
                "rtc_delay_steps": rtc_request.inference_delay,
                "rtc_execution_horizon": rtc_request.execution_horizon,
                "rtc_prior_horizon": rtc_request.prefix_steps,
                "rtc_prefix_start_index": rtc_request.prefix_start_index,
                "rtc_chunk_sequence": session.chunk_sequence if session is not None else 0,
                "rtc_returned_steps": returned_steps,
                "rtc_executable_steps": returned_steps - rtc_request.inference_delay,
            },
        }
        if want_features:
            features = self._extract_feature()
            if features is None:
                raise RuntimeError("return_features 请求但本次 forward 未捕获到 VLM prefix hidden")
            result["features"] = features
        if want_monitor:
            result["monitor_report"] = self.monitor.report()
        return result

    def validate_inference(self) -> None:
        state = np.zeros((self.state_dim,), dtype=np.float32)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        observation = {
            "active_state": state,
            "images": {key: image for key in self.expected_image_keys},
            "prompt": "find the green block and place it into the basket.",
            "actions_per_inference": self.actions_per_inference,
            "request_id": "offline-validation",
        }
        session = self.new_session()
        result = self.infer(observation, session)
        actions = np.asarray(result["actions"])
        full_actions = np.asarray(result["full_actions"])
        if actions.shape != (self.actions_per_inference, self.action_dim):
            raise RuntimeError(f"Validation action shape mismatch: {actions.shape}")
        if full_actions.shape != (self.config.chunk_size, self.action_dim):
            raise RuntimeError(f"Validation full chunk shape mismatch: {full_actions.shape}")
        print(
            f"{self.policy_type} offline validation passed: "
            f"actions={actions.shape} full_actions={full_actions.shape} "
            f"min={actions.min():.6f} max={actions.max():.6f} "
            f"infer_ms={result['policy_timing']['infer_ms']:.1f}",
            flush=True,
        )
        if not self.rtc_enabled:
            return

        prefix_start = 5
        inference_delay = 3
        switch_previous_index = prefix_start + inference_delay - 1
        torch.manual_seed(20260803)
        noise = self.policy.model.sample_noise(
            (1, self.config.chunk_size, self.config.max_action_dim), self.device
        )
        policy_observation = self.preprocessor(self._build_policy_observation(observation))
        self.config.rtc_config.enabled = False
        try:
            plain_normalized = self.policy.predict_action_chunk(policy_observation, noise=noise.clone())
        finally:
            self.config.rtc_config.enabled = True
        plain_actions = np.asarray(self.postprocessor(plain_normalized).detach().cpu())[0]

        rtc_observation = dict(observation)
        rtc_observation["request_id"] = "offline-validation-rtc"
        rtc_observation["rtc_context"] = {
            "previous_action_start_index": prefix_start,
            "delay_steps": inference_delay,
            "execution_horizon": self.config.rtc_config.execution_horizon,
        }
        rtc_result = self.infer(rtc_observation, session, noise=noise.clone())
        rtc_actions = np.asarray(rtc_result["full_actions"])
        previous_actions = np.asarray(result["full_actions"])
        plain_delta = plain_actions[0] - previous_actions[switch_previous_index]
        rtc_delta = rtc_actions[inference_delay] - previous_actions[switch_previous_index]
        metrics = {
            "previous_last_timestep": switch_previous_index,
            "plain_first_timestep": 0,
            "rtc_first_timestep": inference_delay,
            "prefix_start_timestep": prefix_start,
            "prefix_steps": rtc_result["rtc"]["rtc_prior_horizon"],
            "plain_boundary_l2": float(np.linalg.norm(plain_delta)),
            "rtc_boundary_l2": float(np.linalg.norm(rtc_delta)),
            "plain_boundary_max_abs": float(np.max(np.abs(plain_delta))),
            "rtc_boundary_max_abs": float(np.max(np.abs(rtc_delta))),
        }
        if not rtc_result["rtc"]["rtc_applied"] or rtc_result["rtc"]["rtc_skip_steps"] != inference_delay:
            raise RuntimeError(f"RTC offline validation did not apply the expected prefix: {rtc_result['rtc']}")
        print(
            f"{self.policy_type} RTC offline validation passed: "
            + json.dumps(metrics, sort_keys=True),
            flush=True,
        )


class MsgpackPolicyServer:
    def __init__(self, policy: SmolVLAPiperPolicy, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = {
            "policy_type": policy.policy_type,
            "robot_layout": "dual-arm",
            "policy_io_mode": policy.policy_io_mode,
            "state_dim": policy.state_dim,
            "action_dim": policy.action_dim,
            "controlled_arm": policy.controlled_arm,
            "action_semantics": policy.action_semantics,
            "actions_per_inference": policy.actions_per_inference,
            "full_action_horizon": policy.config.chunk_size,
            "rtc_supported": policy.rtc_supported,
            "rtc_enabled": policy.rtc_enabled,
            "rtc_mode": f"lerobot_{policy.policy_type}_flow_matching" if policy.rtc_enabled else "off",
            "rtc_execution_horizon": policy.config.rtc_config.execution_horizon,
            "rtc_max_guidance_weight": policy.config.rtc_config.max_guidance_weight,
            "rtc_prefix_attention_schedule": policy.config.rtc_config.prefix_attention_schedule.value,
            "required_policy_cameras": list(policy.expected_image_keys),
            "checkpoint": str(policy.checkpoint),
            "normalization": policy.normalization_metadata,
            # Robo_int fork：可选特征通道（不请求不影响旧 client）
            "feature_available": policy.feature_name is not None,
            "feature_name": policy.feature_name,
            "feature_dim": policy.feature_dim,
            "feature_dtype": "float32" if policy.feature_name else None,
            "monitor_enabled": policy.monitor.enabled,
            "arch_plugins": {
                "action_query_vision": {
                    "enabled": policy.aqv is not None,
                    "fixed_gate": policy.aqv.fixed_gate if policy.aqv is not None else None,
                    "num_layers": policy.aqv.num_layers if policy.aqv is not None else None,
                    "vision_tokens": policy.aqv.vision_tokens if policy.aqv is not None else None,
                    "camera_mapping": policy.aqv.camera_names if policy.aqv is not None else None,
                },
                "memory": {
                    "enabled": policy.mem is not None,
                    "active": policy.mem.active if policy.mem is not None else None,
                    "history_len": policy.mem.history_len if policy.mem is not None else None,
                    "num_tokens": policy.mem.num_tokens if policy.mem is not None else None,
                    "bank_size": len(policy.mem.bank) if policy.mem is not None else None,
                    "feature_source": "vlm_prefix_mean(no robot state)",
                },
            },
        }

    async def handler(self, websocket: websocket_server.ServerConnection) -> None:
        LOGGER.info("Connection opened from %s", websocket.remote_address)
        packer = Packer()
        session = self.policy.new_session()
        await websocket.send(packer.pack(self.metadata))
        previous_total = None
        while True:
            try:
                started = time.perf_counter()
                observation = unpackb(await websocket.recv())
                result = self.policy.infer(observation, session)
                rtc_info = result["rtc"]
                LOGGER.info(
                    "request_id=%s rtc_enabled=%s rtc_applied=%s sequence=%d "
                    "prefix_start=%d prefix_steps=%d delay_steps=%d execution_horizon=%d",
                    result.get("request_id"),
                    rtc_info["rtc_enabled"],
                    rtc_info["rtc_applied"],
                    rtc_info["rtc_chunk_sequence"],
                    rtc_info["rtc_prefix_start_index"],
                    rtc_info["rtc_prior_horizon"],
                    rtc_info["rtc_delay_steps"],
                    rtc_info["rtc_execution_horizon"],
                )
                result["server_timing"] = {
                    "infer_ms": result["policy_timing"]["infer_ms"],
                }
                if previous_total is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(result))
                previous_total = time.perf_counter() - started
            except websockets.ConnectionClosed:
                LOGGER.info("Connection closed from %s", websocket.remote_address)
                return
            except Exception:
                LOGGER.exception("Inference request failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason=f"{self.policy.policy_type} inference error; traceback sent in previous frame",
                )
                return

    @staticmethod
    def health_check(
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> websocket_server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def run(self) -> None:
        async with websocket_server.serve(
            self.handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self.health_check,
        ) as server:
            LOGGER.info("Serving %s on %s:%d", self.policy.policy_type, self.host, self.port)
            await server.serve_forever()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        f"Loading LeRobot checkpoint={args.checkpoint} device={args.device} "
        f"actions_per_inference={args.actions_per_inference}",
        flush=True,
    )
    policy = SmolVLAPiperPolicy(
        checkpoint=args.checkpoint.resolve(),
        device=args.device,
        actions_per_inference=args.actions_per_inference,
        policy_io_mode=args.policy_io_mode,
        rtc_enabled=args.rtc_enabled,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        monitor_enabled=args.monitor,
        arch_action_query_vision=args.arch_action_query_vision,
        aqv_gate=args.aqv_gate,
        arch_mem=args.arch_mem,
        mem_active=args.mem_active,
        mem_history_len=args.mem_history_len,
        mem_tokens=args.mem_tokens,
        arch_ttt=args.arch_ttt,
        ttt_active=args.ttt_active,
        ttt_gate=args.ttt_gate,
        ttt_latent=args.ttt_latent,
        ttt_lr=args.ttt_lr,
        ttt_chunk=args.ttt_chunk,
        ttt_persist=args.ttt_persist,
    )
    print(
        "Normalization active: " + json.dumps(policy.normalization_metadata, sort_keys=True),
        flush=True,
    )
    if args.validate_only:
        policy.validate_inference()
        return
    asyncio.run(MsgpackPolicyServer(policy, args.host, args.port).run())


if __name__ == "__main__":
    main()
