#!/usr/bin/env python3
"""LoRA B 类功能测试（非主入口；需 molmoact2 解释器 + GPU + 150000 checkpoint）。

链路：native π0.5 -> 真实 PEFT LoRA attach（π0.5 官方默认 target）-> forward ->
BC loss（仓库 compute_loss 约定的动作块 MSE）-> backward -> LoRA optimizer step ->
base 冻结验证（monitor checksum 实测）-> adapter 独立保存/重载 -> parity -> 性能对比。

运行：
    .venv-smolvla/bin/python scripts/pi05_lora_functional_test.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = Path(
    os.environ.get("ROBO_INT_PI05_CHECKPOINT", str(ROOT / "checkpoints" / "pi05"))
).expanduser()
ADAPTER_DIR = ROOT / "outputs" / "lora_functional_test_adapter"

CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return bool(ok)


def load_server_module():
    # server.py 的双路径 monitor import 依赖 sys.path 上有仓库根（包路径）
    # 或 models/pi05 目录（裸模块路径）——两者都给
    for path in (str(ROOT), str(ROOT / "robo_int" / "models" / "pi05")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("fork_server", ROOT / "robo_int" / "models" / "pi05" / "server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fork_server"] = module
    spec.loader.exec_module(module)
    return module


def build_observation(server, seed=7):
    rng = np.random.default_rng(seed)
    d = np.load(ROOT / "data" / "piper_dryrun" / "episodes" / "ep_000001.npz")
    cameras = ["head", "left_wrist", "right_wrist"]
    images = {k: d[f"image/{k}"][0] for k in cameras}
    state = d["states"][0].astype(np.float32)
    return state, images


def main() -> int:
    import re

    from peft import LoraConfig, PeftModel, get_peft_model

    server = load_server_module()
    print("加载 π0.5（150000，monitor on）...", flush=True)
    policy = server.SmolVLAPiperPolicy(
        CHECKPOINT, device="cuda", actions_per_inference=8,
        rtc_enabled=False, monitor_enabled=True,
    )
    monitor = policy.monitor
    pi05 = policy.policy

    # ---------------------------------------------------------------- targets
    # π0.5 官方默认 PEFT target（modeling_pi05._get_default_peft_targets）：
    # gemma_expert 的 self_attn q/v 投影 + action expert 的 in/out 投影
    default_targets = pi05._get_default_peft_targets()
    target_regex = default_targets["target_modules"]
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_regex,
                             lora_dropout=0.0, bias="none",
                             modules_to_save=default_targets["modules_to_save"])

    base_param_count = sum(p.numel() for p in pi05.model.parameters())
    pattern = re.compile(target_regex)
    matched = [name for name, m in pi05.model.named_modules()
               if isinstance(m, torch.nn.Linear) and pattern.search(name)]

    # ---------------------------------------------------------------- 固定 obs/noise + base 基线
    state, images = build_observation(server)
    observation = {"active_state": state, "images": images,
                   "prompt": "fold the clothes neatly", "actions_per_inference": 8}
    policy_obs = policy.preprocessor(policy._build_policy_observation(observation))
    torch.manual_seed(20260911)
    noise_main = pi05.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cuda")
    noise_target = pi05.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cuda")

    base_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )

    # base 性能基准必须在 attach 前测（LoRA 注入是原地的，摘除需重建实例）
    def bench(n=5):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))

    base_ms = bench()

    # ================================================================ Test A: attach
    print("\n== Test A：LoRA attach / detach ==", flush=True)
    check("A.target_modules（π0.5 官方默认）", len(matched) > 0,
          f"{len(matched)} 个 Linear（gemma_expert self_attn q/v + action in/out 投影）")
    check("A.parameter count before", True, f"base={base_param_count/1e9:.3f}B")

    peft_model = get_peft_model(pi05.model, lora_config)
    peft_model.eval()
    lora_param_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    check("A.native π0.5 可 attach LoRA", True, f"LoRA 参数 {lora_param_count:,}（{100*lora_param_count/base_param_count:.4f}% of base）")

    # monitor 实测参数所有权（不是看配置）。
    # 用独立 ownership monitor 只注册 peft_model——policy.monitor 里已注册的
    # base_vlm/action_expert 模块与 peft 树重叠（expert 的 LoRA 子参数会被
    # 两个注册模块重复计入），所有权语义要求无重叠注册。
    from robo_int.models.pi05.monitor import ModelInternalMonitor

    own_monitor = ModelInternalMonitor(enabled=True)
    own_monitor.register_parameters("pi05_with_lora", peft_model)
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-4)
    own_monitor.register_optimizer(optimizer)
    stats = own_monitor.report()["parameters"]["stats"]
    check("A.base VLM + Expert params frozen",
          stats["requires_grad_parameter_count"] == lora_param_count,
          f"requires_grad={stats['requires_grad_parameter_count']:,} == LoRA={lora_param_count:,}"
          f"（total={stats['total_parameter_count']:,}，其余全部冻结）")
    check("A.只有 LoRA requires_grad=true",
          stats["requires_grad_parameter_count"] == stats["optimizer_owned_parameter_count"] == lora_param_count)
    check("A.optimizer 只持有 LoRA 参数", stats["optimizer_owned_parameter_count"] == lora_param_count)

    # base 冻结的 checksum 证据（训练前 snapshot，训练后 diff）。
    # attach 后 expert 模块的参数树已含注入的 lora_A/lora_B（它们训练时本来就该变），
    # snapshot 用 keys 过滤只盯 base_layer（真实 base 权重）
    monitor.snapshot_parameters("base_vlm")
    monitor.snapshot_parameters("action_expert", keys=["base_layer"])

    # ================================================================ Test B: disabled parity
    print("\n== Test B：LoRA 零初始化 parity（硬门槛）==", flush=True)
    # get_peft_model 原地替换目标 Linear —— pi05.model 的预测路径自动经过 LoRA
    lora_zero_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )
    diff_b = float(np.abs(lora_zero_actions - base_actions).max())
    check("B.LoRA attach 后（零权重）max_abs_action_diff ≈ 0", diff_b < 1e-4, f"max_abs_diff={diff_b:.3e}")

    # ================================================================ Test C: 最小训练
    print("\n== Test C：1 batch × 3 steps（π0.5 原生 flow-matching loss：PI05Policy.forward）==", flush=True)
    # predict_action_chunk / sample_actions 均为 @torch.no_grad 推理路径——训练必须走
    # forward()（真实训练损失）。目标动作块 = base policy 另一 noise 下的预测
    # （模仿 base 自采样行为的标准 BC 目标形式），action 维度按 prepare_action 约定。
    with torch.no_grad():
        action_dim = pi05.config.output_features["action"].shape[0]
        target_full = pi05.predict_action_chunk(policy_obs, noise=noise_target.clone())[:, :, :action_dim]
    train_batch = {**policy_obs, "action": target_full}
    losses = []
    for step in range(3):
        loss, info = pi05.forward(train_batch)
        optimizer.zero_grad()
        loss.backward()
        lora_grads = [p.grad for p in optimizer.param_groups[0]["params"]]
        grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in lora_grads if g is not None)))
        optimizer.step()
        own_monitor.after_optimizer_step()
        losses.append(float(loss.detach()))
    c_stats = own_monitor.report()["parameters"]["stats"]
    check("C.loss finite", all(np.isfinite(l) for l in losses) and losses[-1] > 0,
          f"losses={['%.4f' % l for l in losses]}")
    check("C.backward + LoRA gradient finite",
          np.isfinite(grad_norm) and grad_norm > 0, f"LoRA grad_norm={grad_norm:.4f}")
    check("C.optimizer step", c_stats["updated_parameter_count"] > 0 and c_stats["parameter_update_norm"] > 0,
          f"updated={c_stats['updated_parameter_count']} update_norm={c_stats['parameter_update_norm']:.6f}")
    check("C.nan/inf gradients = 0",
          c_stats["nan_gradient_count"] == 0 and c_stats["inf_gradient_count"] == 0)
    snap = monitor.report()["parameters"]["snapshots"]
    check("C.base VLM changed=false", snap["base_vlm"]["changed"] == [])
    check("C.action Expert changed=false", snap["action_expert"]["changed"] == [])

    # 训练后动作确有变化（LoRA 生效的直接证据）
    lora_trained_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )
    diff_trained = float(np.abs(lora_trained_actions - base_actions).max())
    check("C.LoRA 生效（训练后动作偏离 base）", diff_trained > 0, f"max_abs_diff={diff_trained:.4e}")

    # ================================================================ Test D: save / reload
    print("\n== Test D：adapter 独立保存 / 重载 ==", flush=True)
    base_ckpt_files_before = sorted(p.name for p in CHECKPOINT.iterdir())
    shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
    peft_model.save_pretrained(str(ADAPTER_DIR))
    base_ckpt_files_after = sorted(p.name for p in CHECKPOINT.iterdir())
    check("D.save adapter（不覆盖 150000）",
          (ADAPTER_DIR / "adapter_model.safetensors").is_file()
          and base_ckpt_files_before == base_ckpt_files_after,
          f"saved to {ADAPTER_DIR}；150000 目录文件数不变（{len(base_ckpt_files_after)}）")

    print("加载第二个 π0.5 实例（base 150000 + attach/load LoRA）...", flush=True)
    policy2 = server.SmolVLAPiperPolicy(
        CHECKPOINT, device="cuda", actions_per_inference=8, rtc_enabled=False, monitor_enabled=False,
    )
    policy2.policy.model = PeftModel.from_pretrained(policy2.policy.model, str(ADAPTER_DIR))
    policy2.policy.model.eval()
    reloaded_actions = np.asarray(
        policy2.infer(dict(observation), policy2.new_session(), noise=noise_main.clone())["full_actions"]
    )
    diff_d = float(np.abs(reloaded_actions - lora_trained_actions).max())
    check("D.reload 后同确定性输入输出一致", diff_d < 1e-5, f"max_abs_diff={diff_d:.3e}")

    # ================================================================ Test E: performance
    print("\n== Test E：性能对比（base vs LoRA enabled，base 基准在 attach 前已测）==", flush=True)
    lora_ms = bench()
    perf = monitor.report()["performance"]
    check("E.extra_forward_count = 0", perf["extra_forward_count"] == 0)
    check("E.性能记录", True,
          f"base={base_ms:.1f}ms lora={lora_ms:.1f}ms（+{100*(lora_ms-base_ms)/base_ms:.1f}%），"
          f"pre={perf['preprocess_ms']:.1f} vlm={perf['vlm_ms']:.1f} expert={perf['action_expert_ms']:.1f} "
          f"flow={perf['flow_sampling_ms']:.1f} post={perf['postprocess_ms']:.1f} other={perf['other_ms']:.1f}，"
          f"gpu_peak={perf['gpu_memory_peak_mb']}MB")

    # ================================================================ 总结
    print("\n===== LoRA Functional Test =====", flush=True)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"===== {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} =====", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
