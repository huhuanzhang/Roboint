"""ModelInternalMonitor 离线单测（零重依赖，无 GPU/真机/server）。

重点一：FakeHook 场景必须在 report 里显式 "HOOK INACTIVE / FAIL"。
重点二：request-scoped 健康判定——warmup 成功（total>0）但当前请求未触发
（this_request=0）同样判 FAIL。
"""

import numpy as np
import pytest
import torch

from robo_int.models.pi05.monitor import ModelInternalMonitor


def _monitor(enabled=True):
    m = ModelInternalMonitor(enabled=enabled)
    m.register_hook(
        "vlm_prefix_capture",
        hook_point="paligemma.model.language_model.forward(wrap)",
        source_module="PI05Model...language_model", source_tensor="last_hidden_state",
    )
    m.register_tensor("vlm_prefix_mean")
    return m


def _fake_infer(m, request_id, fire_hook=True):
    """模拟一次请求的 monitor 生命周期。"""
    m.begin_inference(request_id=request_id)
    if fire_hook:
        m.record_hook("vlm_prefix_capture", torch.randn(1, 968, 2048))
    m.end_inference(preprocess_ms=1.0, total_ms=10.0, flow_sampling_ms=5.0,
                    postprocess_ms=1.0, device="cpu")


def test_hook_health_counts_and_metadata():
    m = _monitor()
    _fake_infer(m, "req-1")
    _fake_infer(m, "req-2")
    entry = m.report()["hooks"]["vlm_prefix_capture"]
    assert entry["hook_count_total"] == 2
    assert entry["hook_count_this_request"] == 1  # request-scoped
    assert entry["last_hook_request_id"] == "req-2"
    assert entry["status"] == "ACTIVE"
    assert entry["shape"] == (1, 968, 2048)


def test_inactive_hook_detected():
    m = _monitor()
    report = m.report()["hooks"]["vlm_prefix_capture"]  # 从未触发
    assert report["enabled"] is True
    assert report["hook_count_total"] == 0
    assert report["status"] == "HOOK INACTIVE / FAIL"


def test_warmup_success_then_failure_detected():
    """warmup（req-0）触发过，但当前请求未触发 -> 必须判 FAIL，不是 ACTIVE。"""
    m = _monitor()
    _fake_infer(m, "warmup", fire_hook=True)
    assert m.report()["hooks"]["vlm_prefix_capture"]["status"] == "ACTIVE"
    _fake_infer(m, "req-1", fire_hook=False)  # hook 静默失效
    entry = m.report()["hooks"]["vlm_prefix_capture"]
    assert entry["hook_count_total"] == 1  # 累计看不出来
    assert entry["hook_count_this_request"] == 0  # request-scoped 抓得住
    assert entry["status"] == "HOOK INACTIVE / FAIL"


def test_disabled_hook_reports_disabled():
    m = _monitor()
    m.register_hook("future_mem", hook_point="x", source_module="y",
                    source_tensor="z", enabled=False)
    assert m.report()["hooks"]["future_mem"]["status"] == "disabled"


def test_tensor_stats_finite_and_nans():
    m = _monitor()
    m.record_tensor("vlm_prefix_mean", torch.linspace(-1.0, 1.0, 2048))
    stats = m.report()["tensors"]["vlm_prefix_mean"]
    assert stats["n_samples"] == 1
    assert stats["min"] == -1.0 and stats["max"] == 1.0
    assert stats["nan_count"] == 0 and stats["inf_count"] == 0
    assert np.isfinite(stats["mean"]) and np.isfinite(stats["std"]) and stats["l2_norm"] > 0
    m.record_tensor("vlm_prefix_mean", torch.tensor([1.0, float("nan"), float("inf")]))
    stats2 = m.report()["tensors"]["vlm_prefix_mean"]
    assert stats2["nan_count"] == 1 and stats2["inf_count"] == 1


def test_policy_impact_metrics():
    m = _monitor()
    base = np.ones((8, 14), dtype=np.float32)
    m.record_impact(base, base, modifiers_active=False)
    impact = m.report()["policy_impact"]
    assert impact["modifiers_active"] is False
    assert impact["mean_abs_delta"] == 0.0 and impact["l2_delta"] == 0.0
    assert impact["cosine_similarity"] == pytest.approx(1.0)
    modified = base + 0.5
    metrics = m.record_impact(base, modified, modifiers_active=True)
    assert metrics["mean_abs_delta"] == 0.5
    assert metrics["max_abs_delta"] == 0.5
    assert abs(metrics["l2_delta"] - 0.5 * np.sqrt(8 * 14)) < 1e-9
    assert m.report()["policy_impact"]["modifiers_active"] is True


def test_parameter_ownership_and_gradient_stats():
    m = _monitor()
    base = torch.nn.Linear(4, 2)          # 假装 base 模型（冻结）
    adapter = torch.nn.Linear(4, 2)       # 假装 LoRA（可训练）
    for p in base.parameters():
        p.requires_grad_(False)
    m.register_parameters("base", base)
    m.register_parameters("adapter", adapter)
    opt = torch.optim.SGD(adapter.parameters(), lr=0.1)
    m.register_optimizer(opt)

    stats = m.report()["parameters"]["stats"]
    assert stats["gradient_monitor"] == "inactive"  # 无 backward
    assert stats["total_parameter_count"] == 2 * (4 * 2 + 2)
    assert stats["requires_grad_parameter_count"] == 4 * 2 + 2  # 只有 adapter
    assert stats["optimizer_owned_parameter_count"] == 4 * 2 + 2  # optimizer 只持 adapter

    # backward + step 后：梯度流动 / 更新生效
    loss = adapter(torch.ones(2, 4)).sum()
    loss.backward()
    opt.step()
    after = m.after_optimizer_step()
    stats = m.report()["parameters"]["stats"]
    assert stats["gradient_monitor"] == "active"
    assert stats["gradient_present_parameter_count"] == 2
    assert stats["nan_gradient_count"] == 0 and stats["inf_gradient_count"] == 0
    assert after["updated_parameter_count"] == 2
    assert after["parameter_update_norm"] > 0
    assert stats["updated_parameter_count"] == 2
    assert stats["parameter_update_norm"] > 0


def test_parameter_snapshot_change_detection():
    m = _monitor()
    linear = torch.nn.Linear(4, 2)
    m.register_parameters("module_a", linear)
    m.snapshot_parameters("module_a")
    assert m.report()["parameters"]["snapshots"]["module_a"]["changed"] == []
    with torch.no_grad():
        linear.weight[0, 0] += 1.0
    changed = m.diff_snapshot("module_a")["changed"]
    assert len(changed) == 1 and "weight" in changed[0]


def test_performance_counters_segments_and_residual():
    m = _monitor()
    m.begin_inference(request_id="r")
    m.add_stage_time("vlm", 0.005)
    m.add_stage_time("action_expert", 0.002)
    m.end_inference(preprocess_ms=1.0, total_ms=20.0, flow_sampling_ms=10.0,
                    postprocess_ms=2.0, arch_plugin_ms=0.5, device="cpu")
    perf = m.report()["performance"]
    assert perf["forward_count"] == 1 and perf["extra_forward_count"] == 0
    assert perf["vlm_ms"] == 5.0 and perf["action_expert_ms"] == 2.0
    assert perf["flow_sampling_ms"] == 10.0 and perf["postprocess_ms"] == 2.0
    # 残差 = total − 全部已分类分段
    assert perf["other_ms"] == pytest.approx(20.0 - (1.0 + 5.0 + 2.0 + 10.0 + 2.0 + 0.5))
    m.record_extra_forward("plugin_sneaky_second_pass")
    assert m.report()["performance"]["extra_forward_count"] == 1
    m.reset()
    perf = m.report()["performance"]
    assert perf["forward_count"] == 0 and perf["extra_forward_count"] == 0
    assert "vlm_prefix_capture" in m.report()["hooks"]


def test_monitor_disabled_by_default():
    assert ModelInternalMonitor().enabled is False
    m = _monitor(enabled=False)
    assert m.report()["enabled"] is False
