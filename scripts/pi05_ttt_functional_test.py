#!/usr/bin/env python3
"""TTT-Linear 快权重层功能测试（B 类；venv-smolvla + GPU）。

链路（全部真实）：
  真实 Piper 观测（ep_000001）-> π0.5 embed_prefix -> prefix token 流 (1,968,2048)
     ↓ TTTLinearLayer（mini-batch 快权重在线更新 + 残差注入，fp32 内核）
  变换后的 prefix -> prefix pass + Action Expert 全部 denoise 步

硬门槛：TTT inactive / gate=0 -> embs 原对象原样返回 -> base 动作 bit 级一致。
运行：.venv-smolvla/bin/python scripts/pi05_ttt_functional_test.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = Path(
    os.environ.get("ROBO_INT_PI05_CHECKPOINT", str(ROOT / "checkpoints" / "pi05"))
).expanduser()
EP = ROOT / "data" / "piper_dryrun" / "episodes" / "ep_000001.npz"

CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return bool(ok)


def load_server_module():
    for path in (str(ROOT), str(ROOT / "robo_int" / "models" / "pi05")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "fork_server", ROOT / "robo_int" / "models" / "pi05" / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fork_server"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    data = np.load(EP)
    states, executed = data["states"], data["actions"]

    def obs_at(step: int) -> dict:
        return {
            "active_state": states[step].astype(np.float32),
            "images": {k: data[f"image/{k}"][step] for k in ("head", "left_wrist", "right_wrist")},
            "prompt": "fold the clothes neatly",
            "actions_per_inference": 8,
        }

    server = load_server_module()
    print("加载 π0.5（150000，monitor on，先无 TTT = 真 base）...", flush=True)
    policy = server.SmolVLAPiperPolicy(
        CHECKPOINT, device="cuda", actions_per_inference=8,
        rtc_enabled=False, monitor_enabled=True,
    )
    print(f"prefix_len={policy._prefix_len} feature_dim={policy.feature_dim}", flush=True)
    torch.manual_seed(20260916)
    noise = policy.policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cuda")

    cur = obs_at(8)
    sd_keys_before = set(policy.policy.model.state_dict().keys())

    # ================================================== 真 base（无 TTT 挂载）
    base_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )

    # ================================================== 安装 TTT（inactive）
    policy._install_ttt(active=False, gate=0.0, latent_dim=128, lr=0.1,
                        chunk_size=64, persist=False)
    disabled_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )
    diff_disabled = float(np.abs(disabled_actions - base_actions).max())
    check("TTT inactive max_abs_action_diff = 0（bit 级，挂件已 attach）",
          diff_disabled == 0.0, f"max_abs_diff={diff_disabled:.3e}")
    check("π0.5 state_dict 键集不受挂载污染",
          set(policy.policy.model.state_dict().keys()) == sd_keys_before,
          f"keys={len(sd_keys_before)}")

    # ================================================== active + gate=0
    policy.ttt.active = True
    policy.ttt.gate = 0.0
    zero_gate_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )
    diff_zero = float(np.abs(zero_gate_actions - base_actions).max())
    check("TTT active + gate=0 max_abs_action_diff = 0（bit 级）",
          diff_zero == 0.0, f"max_abs_diff={diff_zero:.3e}")

    # ================================================== 真实激活（gate=0.05）
    policy.ttt.gate = 0.05
    session = policy.new_session()
    ttt_actions = np.asarray(
        policy.infer(dict(cur), session, noise=noise.clone())["full_actions"]
    )
    stats = policy.ttt.last_stats
    report = policy.monitor.report()
    hook = report["hooks"]["ttt_apply"]
    diff_a = float(np.abs(ttt_actions - base_actions).max())
    check("TTT enabled 动作偏离 base（快权重真实影响策略）", diff_a > 0,
          f"max_abs_diff={diff_a:.4e}")
    check("快权重在线更新真实发生（w_end != w_start，chunk 更新）",
          stats.get("applied") and stats.get("w_delta_norm", 0.0) > 0
          and stats.get("w_end_norm", 0.0) > stats.get("w_start_norm", 1e18),
          f"seq_len={stats.get('seq_len')} chunks={stats.get('chunks')} "
          f"w_start={stats.get('w_start_norm'):.3e} w_end={stats.get('w_end_norm'):.3e} "
          f"w_delta={stats.get('w_delta_norm'):.3e}")
    check("重建自监督目标 finite（‖W k_t − v_t‖²）",
          bool(np.isfinite(stats.get("recon_loss", float("nan")))),
          f"recon_loss={stats.get('recon_loss'):.4e} per_token={stats.get('recon_per_token'):.4e}")
    check("request-scoped hook ACTIVE", hook["hook_count_this_request"] > 0,
          f"this_request={hook['hook_count_this_request']} total={hook['hook_count_total']} status={hook['status']}")
    check("输出 token 有限（无 NaN/Inf）",
          stats.get("out_nan_count") == 0 and stats.get("out_inf_count") == 0,
          f"out_delta_l2={stats.get('out_delta_l2'):.4e}")
    ttt_tensor = report["tensors"].get("ttt_out", {})
    ttt_mean = ttt_tensor.get("mean")
    check("monitor ttt_out 统计 finite", ttt_mean is not None and bool(np.isfinite(ttt_mean)),
          f"mean={ttt_mean} stats_keys={sorted(k for k in ttt_tensor if ttt_tensor[k] is not None)}")

    # ================================================== per-request 语义（非 persist）
    again_actions = np.asarray(
        policy.infer(dict(cur), session, noise=noise.clone())["full_actions"]
    )
    check("非 persist：每请求快权重从 W0 重开（同 obs 同 noise -> bit 级一致）",
          float(np.abs(again_actions - ttt_actions).max()) == 0.0,
          f"max_abs_diff={float(np.abs(again_actions - ttt_actions).max()):.3e}")

    # ================================================== persist 模式（RoboTTT 上下文风味）
    policy.ttt.persist = True
    policy.ttt.reset_episode()
    p1 = np.asarray(policy.infer(dict(cur), session, noise=noise.clone())["full_actions"])
    st1 = dict(policy.ttt.last_stats)
    p2 = np.asarray(policy.infer(dict(cur), session, noise=noise.clone())["full_actions"])
    st2 = dict(policy.ttt.last_stats)
    check("persist：跨请求快权重复用（同输入输出变化）",
          float(np.abs(p2 - p1).max()) > 0 and st2.get("reused_state") is True
          and not st1.get("reused_state", False),
          f"max_abs_diff={float(np.abs(p2 - p1).max()):.4e} "
          f"w_end_1={st1.get('w_end_norm'):.3e} w_end_2={st2.get('w_end_norm'):.3e}")
    policy.ttt.reset_episode()  # new_session 同义路径
    p3 = np.asarray(policy.infer(dict(cur), session, noise=noise.clone())["full_actions"])
    check("persist：episode 边界重置回 per-sequence 起点",
          float(np.abs(p3 - p1).max()) == 0.0,
          f"max_abs_diff={float(np.abs(p3 - p1).max()):.3e}")
    policy.ttt.persist = False
    policy.ttt.reset_episode()

    # ================================================== Policy impact
    base64, ttt64 = base_actions.astype(np.float64), ttt_actions.astype(np.float64)
    d = ttt64 - base64
    cos = float(np.dot(base64.ravel(), ttt64.ravel())
                / (np.linalg.norm(base64) * np.linalg.norm(ttt64)))
    check("impact 指标", diff_a > 0,
          f"mean_abs={np.abs(d).mean():.4e} max_abs={np.abs(d).max():.4e} "
          f"l2={np.linalg.norm(d):.4e} cos={cos:.6f}")

    # ================================================== forward 数 + 性能
    perf = report["performance"]
    check("extra_forward_count = 0（TTT 复用同一次 prefix pass，无额外 VLA forward）",
          perf["extra_forward_count"] == 0,
          f"forward_count={perf['forward_count']} backward={perf['backward_count']}")
    policy.ttt.active = False
    def bench(n=3):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            policy.infer(dict(cur), session, noise=noise.clone())
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(ts))
    base_ms = bench()
    policy.ttt.active = True
    ttt_ms = bench()
    check("性能记录", True, f"base={base_ms:.1f}ms ttt={ttt_ms:.1f}ms（+{100*(ttt_ms-base_ms)/base_ms:.1f}%）"
          f" gpu_peak={perf['gpu_memory_peak_mb']}MB")

    # ================================================== 最小 backward（原生 flow-matching loss）
    print("\n== 最小 backward（π0.5 原生 loss，1 batch / 2 steps）==", flush=True)
    model = policy.policy.model
    for p in model.parameters():
        p.requires_grad_(False)
    policy_obs = policy.preprocessor(policy._build_policy_observation(dict(cur)))
    batch = dict(policy_obs)
    real_window = executed[8:]
    pad_len = policy.config.chunk_size - len(real_window)
    window = np.concatenate([real_window, np.repeat(real_window[-1:], pad_len, axis=0)], axis=0)
    batch["action"] = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1,50,14)
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    torch.manual_seed(20260917)
    loss, loss_dict = policy.policy.forward(batch)
    check("native loss finite", bool(torch.isfinite(loss)), f"loss={float(loss):.6f}")
    loss.backward()
    grads = [p.grad for p in policy.ttt.parameters() if p.grad is not None]
    grad_norm = float(sum(g.norm().item() for g in grads))
    check("梯度流向 TTT 参数（经残差注入的 prefix token）", len(grads) > 0 and grad_norm > 0
          and all(torch.isfinite(g).all() for g in grads),
          f"n_grad_params={len(grads)}/{len(list(policy.ttt.parameters()))} grad_norm={grad_norm:.3e}")
    optimizer = torch.optim.AdamW(policy.ttt.parameters(), lr=1e-4)
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    ttt_param_ids = {id(p) for p in policy.ttt.parameters()}
    check("optimizer 只持有 TTT 参数", opt_param_ids == ttt_param_ids)
    before = [p.detach().clone() for p in policy.ttt.parameters()]
    for _ in range(2):
        optimizer.zero_grad()
        torch.manual_seed(20260918)
        l2, _ = policy.policy.forward(batch)
        l2.backward()
        optimizer.step()
    changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, policy.ttt.parameters()))
    check("TTT 参数 changed=true（2 steps 后）", changed)
    snap = policy.monitor.report()["parameters"]["snapshots"]
    check("base VLM changed=false", snap["base_vlm"]["changed"] == [])
    check("base Action Expert changed=false", snap["action_expert"]["changed"] == [])
    ttt_changed = policy.monitor.diff_snapshot("ttt_module")
    check("monitor 参数所有权：仅 ttt_module changed",
          len(ttt_changed.get("changed", [])) > 0,
          f"ttt changed {len(ttt_changed.get('changed', []))} params")

    # ================================================== save / load
    print("\n== Save / Load（不触碰 π0.5 150000）==", flush=True)
    policy.ttt.persist = True
    policy.ttt.reset_episode()
    policy.infer(dict(cur), session)  # 产生持久快权重
    path = "/tmp/ttt_b4_state.pt"
    cfg = policy.ttt.save(path)
    from robo_int.arch.ttt import TTTLinearLayer
    fresh = TTTLinearLayer(**{**cfg, "persist": True})
    fresh.load(path)
    fresh.to("cuda")
    fresh.active = True
    x_probe = torch.randn(1, policy._prefix_len, policy.feature_dim, device="cuda")
    with torch.no_grad():
        out_ref, _ = policy.ttt.forward_tokens(x_probe.clone())
        out_new, _ = fresh.forward_tokens(x_probe.clone())
    check("save->new instance->load 同输入同输出（参数+快权重，bit 级）",
          float((out_ref - out_new).abs().max()) == 0.0,
          f"max_diff={float((out_ref - out_new).abs().max()):.3e} config={cfg}")

    # ================================================== 总结
    print("\n===== TTT Functional Test =====", flush=True)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"===== {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} =====", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
