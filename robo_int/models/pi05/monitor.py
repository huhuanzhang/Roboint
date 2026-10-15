"""ModelInternalMonitor：B 类 model-internal 模块的统一只读监测。

服务对象：MEM / action-query attention / LoRA / TTT 以后都复用这一套监测，
不各写各的日志。五类能力：

1. hook health —— 每个 hook 记录调用计数/时间戳/来源，能发现
   ``enabled=true 但 hook_call_count=0`` 的静默失效
   （本仓库曾因 molmoact2 直接调 .forward() 绕过 register_forward_hook 踩坑）；
2. tensor health —— 注册的内部 tensor 只存统计量（min/max/mean/std/l2/nan/inf），
   不落完整 tensor，防日志爆炸；
3. policy impact —— base/modified action 的差异指标；无 modifier 时 base==modified；
4. gradient/parameter —— trainable/frozen 计数、梯度统计（推理态 inactive 不造梯度）、
   轻量参数 checksum snapshot（验证 VLM/Expert/LoRA/TTT 是否被改动）；
5. performance —— 分段耗时 / GPU 显存 / forward 与 backward 计数，
   专查「plugin 偷偷造成额外 VLA forward」。

只读契约：monitor 不改前向、不改动作；enable 与否都不影响模型数值行为
（parity 由 tests/test_pi05_server_parity.py 守护）。本文件零重依赖
（torch/numpy/time），可独立单测。
"""

from __future__ import annotations

import time

import numpy as np


def _tensor_stats(tensor) -> dict:
    """单 tensor 统计量（float32 化后计算，含 nan/inf 计数）。"""
    import torch

    t = tensor.detach().to(torch.float32).reshape(-1)
    finite = t[torch.isfinite(t)]
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nan_count": int(torch.isnan(t).sum()),
        "inf_count": int(torch.isinf(t).sum()),
    }
    if finite.numel() == 0:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "l2_norm": None})
    else:
        stats.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()) if finite.numel() > 1 else 0.0,
                "l2_norm": float(finite.norm()),
            }
        )
    return stats


def _param_checksum(param) -> float:
    """轻量参数校验和：分块 float64 绝对值累加 + 元素数（不 hash 整个模型，
    也不整参 float64 拷贝——那会让单个大参数瞬间多占一倍以上显存）。"""
    import torch

    with torch.no_grad():
        p = param.detach().reshape(-1)
        total = torch.zeros((), dtype=torch.float64, device=p.device)
        chunk = 1 << 22  # 4M 元素/块，峰值额外显存 ~32MB
        for i in range(0, p.numel(), chunk):
            total += p[i : i + chunk].to(torch.float64).abs().sum()
        return float(total) + float(p.numel())


def _action_impact(base, modified) -> dict:
    """base/modified 动作差异指标（numpy 数组，shape 任意但需一致）。"""
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    modified = np.asarray(modified, dtype=np.float64).reshape(-1)
    if base.shape != modified.shape:
        raise ValueError(f"base/modified 形状不一致: {base.shape} vs {modified.shape}")
    diff = modified - base
    norm_base = float(np.linalg.norm(base))
    norm_mod = float(np.linalg.norm(modified))
    denom = norm_base * norm_mod
    cosine = float(np.dot(base, modified) / denom) if denom > 0 else None
    return {
        "mean_abs_delta": float(np.abs(diff).mean()),
        "max_abs_delta": float(np.abs(diff).max()),
        "l2_delta": float(np.linalg.norm(diff)),
        "cosine_similarity": cosine,
    }


class ModelInternalMonitor:
    """统一监测器（默认 disabled；enabled 也不改变模型行为）。"""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._hooks: dict[str, dict] = {}
        self._tensors: dict[str, dict] = {}
        self._perf = {
            "preprocess_ms": 0.0, "vlm_ms": 0.0, "action_expert_ms": 0.0,
            "flow_sampling_ms": 0.0, "postprocess_ms": 0.0,
            "arch_plugin_ms": 0.0, "other_ms": 0.0, "total_inference_ms": 0.0,
        }
        self._stage_buffers = {"vlm": 0.0, "action_expert": 0.0}
        self._counts = {"forward_count": 0, "backward_count": 0, "extra_forward_count": 0}
        self._gpu = {"gpu_memory_allocated_mb": None, "gpu_memory_peak_mb": None}
        self._impact = {"modifiers_active": False, "last": None}
        self._param_modules: dict[str, object] = {}
        self._snapshots: dict[str, dict] = {}
        self._optimizer_state: dict = {"params": [], "copies": [], "updated_count": 0, "update_norm": None}
        self._extra_reasons: list[str] = []
        self._current_request_id = None

    # ---------------------------------------------------------------- hook health

    def register_hook(self, name: str, *, hook_point: str, source_module: str,
                      source_tensor: str, enabled: bool = True) -> None:
        self._hooks[name] = {
            "enabled": bool(enabled), "hook_active": False,
            "hook_count_total": 0, "hook_count_this_request": 0,
            "last_hook_request_id": None,
            "last_hook_timestamp": None, "hook_point": hook_point,
            "source_module": source_module, "source_tensor": source_tensor,
            "shape": None, "dtype": None, "device": None,
        }

    def record_hook(self, name: str, tensor) -> None:
        entry = self._hooks.get(name)
        if entry is None:
            raise KeyError(f"未注册的 hook: {name}")
        entry["hook_count_total"] += 1
        entry["hook_count_this_request"] += 1
        entry["last_hook_request_id"] = self._current_request_id
        entry["hook_active"] = True
        entry["last_hook_timestamp"] = time.time()
        entry["shape"] = tuple(tensor.shape)
        entry["dtype"] = str(tensor.dtype)
        entry["device"] = str(tensor.device)

    # ---------------------------------------------------------------- tensor health

    def register_tensor(self, name: str) -> None:
        self._tensors[name] = {"n_samples": 0, "last_timestamp": None, **{
            k: None for k in ("min", "max", "mean", "std", "l2_norm", "nan_count", "inf_count",
                              "shape", "dtype", "device")}}

    def record_tensor(self, name: str, tensor) -> None:
        if name not in self._tensors:
            raise KeyError(f"未注册的 tensor: {name}")
        stats = _tensor_stats(tensor)
        self._tensors[name].update(stats)
        self._tensors[name]["n_samples"] += 1
        self._tensors[name]["last_timestamp"] = time.time()

    # ---------------------------------------------------------------- policy impact

    def record_impact(self, base_action, modified_action, *, modifiers_active: bool | None = None) -> dict:
        metrics = _action_impact(base_action, modified_action)
        if modifiers_active is not None:
            self._impact["modifiers_active"] = bool(modifiers_active)
        self._impact["last"] = metrics
        return metrics

    # ---------------------------------------------------------------- gradient / parameter

    def register_parameters(self, name: str, module) -> None:
        """登记模块供梯度统计与参数 snapshot（以后 LoRA/TTT 验证改动用）。"""
        self._param_modules[name] = module

    def gradient_report(self) -> dict:
        """参数所有权与梯度统计。

        五个所有权字段各答一问：
        - total_parameter_count            注册模块共有多少参数
        - requires_grad_parameter_count    声明可训练的有多少（配置声明）
        - optimizer_owned_parameter_count  真正被 optimizer 持有的有多少（实际归属）
        - gradient_present_parameter_count backward 后带梯度的有多少（实际流动）
        - updated_parameter_count          上次 step 实际数值变化的有多少（实际生效）
        健康判定看后三者，不看配置。
        """
        import torch

        report: dict = {
            "total_parameter_count": 0,
            "requires_grad_parameter_count": 0,
            "optimizer_owned_parameter_count": 0,
            "gradient_present_parameter_count": 0,
            "updated_parameter_count": self._optimizer_state["updated_count"],
            "nan_gradient_count": 0, "inf_gradient_count": 0,
        }
        opt_ids = {id(p) for p in self._optimizer_state["params"]}
        grad_sq, grad_count = 0.0, 0
        for module in self._param_modules.values():
            for p in module.parameters():
                report["total_parameter_count"] += p.numel()
                if p.requires_grad:
                    report["requires_grad_parameter_count"] += p.numel()
                if id(p) in opt_ids:
                    report["optimizer_owned_parameter_count"] += p.numel()
                if p.grad is not None:
                    grad_count += 1
                    report["gradient_present_parameter_count"] += 1
                    g = p.grad.detach().to(torch.float32)
                    grad_sq += float((g * g).sum())
                    report["nan_gradient_count"] += int(torch.isnan(g).sum())
                    report["inf_gradient_count"] += int(torch.isinf(g).sum())
        if grad_count == 0:
            # 推理模式没有 backward——如实标注，不造梯度
            report["gradient_monitor"] = "inactive"
            report["gradient_norm"] = None
        else:
            report["gradient_monitor"] = "active"
            report["gradient_norm"] = float(grad_sq**0.5)
        report["parameter_update_norm"] = self._optimizer_state["update_norm"]
        return report

    def register_optimizer(self, optimizer) -> None:
        """登记 optimizer（LoRA/TTT 等小参数集）：供所有权统计与 step 差分。

        注意：会 clone 一份参数副本做 update 差分，只适合 adapter 规模参数集，
        不要注册持有整模型参数的 optimizer。
        """
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self._optimizer_state = {
            "params": params,
            "copies": [p.detach().cpu().clone() for p in params],
            "updated_count": 0,
            "update_norm": None,
        }

    def after_optimizer_step(self) -> dict:
        """optimizer.step() 后调用：差分统计 updated 数与 update 范数。"""
        import torch

        params = self._optimizer_state["params"]
        copies = self._optimizer_state["copies"]
        updated, sq = 0, 0.0
        for p, old in zip(params, copies):
            delta = p.detach().cpu().float() - old
            sq += float((delta * delta).sum())
            if float(delta.abs().sum()) != 0.0:
                updated += 1
            old.copy_(p.detach().cpu())
        self._optimizer_state["updated_count"] = updated
        self._optimizer_state["update_norm"] = float(sq**0.5) if params else None
        return {"updated_parameter_count": updated, "parameter_update_norm": self._optimizer_state["update_norm"]}

    def snapshot_parameters(self, name: str, keys: list[str] | None = None) -> dict:
        """对已登记模块做轻量 checksum snapshot {param_name: checksum}。"""
        import torch

        module = self._param_modules[name]
        snap = {}
        with torch.no_grad():
            for param_name, p in module.named_parameters():
                if keys is not None and not any(k in param_name for k in keys):
                    continue
                snap[param_name] = _param_checksum(p)
        self._snapshots[name] = snap
        return snap

    def diff_snapshot(self, name: str, keys: list[str] | None = None) -> dict:
        """当前参数 vs 最近 snapshot：changed 列表（无 snapshot 返回 None）。"""
        import torch

        stored = self._snapshots.get(name)
        if stored is None:
            return {"snapshot": None, "changed": None}
        changed = []
        with torch.no_grad():
            for param_name, p in self._param_modules[name].named_parameters():
                if param_name not in stored:
                    continue
                if _param_checksum(p) != stored[param_name]:
                    changed.append(param_name)
        return {"snapshot": "stored", "num_params": len(stored), "changed": changed}

    # ---------------------------------------------------------------- performance

    def begin_inference(self, request_id=None) -> None:
        """一次推理请求开始：重置分段计时与 request-scoped hook 计数。"""
        self._stage_buffers = {"vlm": 0.0, "action_expert": 0.0}
        self._current_request_id = request_id
        for entry in self._hooks.values():
            entry["hook_count_this_request"] = 0

    def add_stage_time(self, stage: str, seconds: float) -> None:
        """forward 包裹层内累计耗时（廉价；report 仅在 enabled 时展示）。"""
        if stage in self._stage_buffers:
            self._stage_buffers[stage] += seconds

    def end_inference(self, *, preprocess_ms: float, total_ms: float,
                      flow_sampling_ms: float = 0.0, postprocess_ms: float = 0.0,
                      arch_plugin_ms: float = 0.0, device: str = "cpu") -> None:
        vlm_ms = self._stage_buffers["vlm"] * 1000.0
        expert_ms = self._stage_buffers["action_expert"] * 1000.0
        self._perf.update({
            "preprocess_ms": preprocess_ms,
            "vlm_ms": vlm_ms,
            "action_expert_ms": expert_ms,
            "flow_sampling_ms": flow_sampling_ms,
            "postprocess_ms": postprocess_ms,
            "arch_plugin_ms": arch_plugin_ms,
            "total_inference_ms": total_ms,
            # 残差 = 总耗时 − 已分类分段（含计时器误差，可能为极小负值，如实记录）
            "other_ms": total_ms - (preprocess_ms + vlm_ms + expert_ms
                                    + flow_sampling_ms + postprocess_ms + arch_plugin_ms),
        })
        self._counts["forward_count"] += 1
        if "cuda" in str(device):
            import torch

            self._gpu["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            self._gpu["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    def record_extra_forward(self, reason: str = "") -> None:
        """同一请求本可复用却再次 forward 时调用（plugin 偷跑检测点）。"""
        self._counts["extra_forward_count"] += 1
        if reason:
            self._extra_reasons.append(reason)

    def record_backward(self) -> None:
        self._counts["backward_count"] += 1

    def reset(self) -> None:
        """清零计数/统计（保留注册表与 snapshot）。"""
        self._counts = {"forward_count": 0, "backward_count": 0, "extra_forward_count": 0}
        for entry in self._hooks.values():
            entry.update({"hook_active": False, "hook_count_total": 0,
                          "hook_count_this_request": 0,
                          "last_hook_request_id": None, "last_hook_timestamp": None})
        for name in self._tensors:
            self.register_tensor(name)

    # ---------------------------------------------------------------- report

    def report(self) -> dict:
        hooks = {}
        for name, entry in self._hooks.items():
            # 健康判定必须是 request-scoped：warmup 的累计计数不能证明
            # 当前请求的 hook 仍在工作（warmup 成功 + 后续静默失效必须判 FAIL）
            if not entry["enabled"]:
                status = "disabled"
            elif entry["hook_count_this_request"] > 0:
                status = "ACTIVE"
            else:
                status = "HOOK INACTIVE / FAIL"
            hooks[name] = {**entry, "status": status}
        return {
            "enabled": self.enabled,
            "request_id": self._current_request_id,
            "hooks": hooks,
            "tensors": dict(self._tensors),
            "policy_impact": {
                "modifiers_active": self._impact["modifiers_active"],
                **(self._impact["last"] or {}),
            },
            "parameters": {
                "stats": self.gradient_report(),
                "snapshots": {name: self.diff_snapshot(name) for name in self._param_modules
                              if name in self._snapshots},
            },
            "performance": {**self._perf, **self._counts, **self._gpu},
        }
