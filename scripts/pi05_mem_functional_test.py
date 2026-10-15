#!/usr/bin/env python3
"""MEM 多尺度具身记忆功能测试（B 类；venv-smolvla + GPU）。

链路（全部真实）：
  episode 内真实历史（过去请求已缓存的 vlm_prefix_mean，2048D，不含 robot state）
  + 真实文本事件库（由真实 state 测量值生成的文本，checkpoint 冻结 embed 编码）
     ↓ MEMModule（temporal cross-attention + 文本 cross-attention）
  记忆 token (1,4,2048)
     ↓ 注入 embed_prefix 输出末尾（pad=1/att=0）
  prefix pass + Action Expert 全部 denoise 步自然 attend

硬门槛：MEM disabled / 无可记忆内容 -> 原对象原样返回 -> base 动作 bit 级一致。
运行：.venv-smolvla/bin/python scripts/pi05_mem_functional_test.py
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

    def gripper_text(step: int) -> str:
        # 真实测量值文本（不发明语义阈值）：14D = 双臂各 6 关节 + 夹爪
        l, r = float(states[step][6]), float(states[step][13])
        return f"step {step}: left gripper at {l:.3f}, right gripper at {r:.3f}"

    server = load_server_module()
    print("加载 π0.5（150000，monitor on，先无 MEM = 真 base）...", flush=True)
    policy = server.SmolVLAPiperPolicy(
        CHECKPOINT, device="cuda", actions_per_inference=8,
        rtc_enabled=False, monitor_enabled=True,
    )
    print(f"prefix_len={policy._prefix_len} feature_dim={policy.feature_dim}", flush=True)
    torch.manual_seed(20260913)
    noise = policy.policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cuda")

    cur = obs_at(8)

    # ================================================== 真 base（无 MEM 挂载）
    base_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )

    # ================================================== 安装 MEM（inactive）
    policy._install_memory(active=False, history_len=4, num_tokens=4)
    disabled_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )
    diff_disabled = float(np.abs(disabled_actions - base_actions).max())
    check("MEM disabled max_abs_action_diff ≈ 0（bit 级，挂件已 attach）",
          diff_disabled == 0.0, f"max_abs_diff={diff_disabled:.3e}")

    # ================================================== active 但无可记忆内容
    policy.mem.active = True
    policy.mem.reset_episode()
    empty_actions = np.asarray(
        policy.infer(dict(cur), policy.new_session(), noise=noise.clone())["full_actions"]
    )
    diff_empty = float(np.abs(empty_actions - base_actions).max())
    check("MEM active + 空记忆 max_abs_diff ≈ 0（bit 级，无假记忆）",
          diff_empty == 0.0, f"max_abs_diff={diff_empty:.3e}")

    # ================================================== 真实历史：回放 ep 步 4..7
    # （这些就是“过去的请求”：真实 Piper dry-run 观测经真实 π0.5 forward，
    #   server 端把每步 vlm_prefix_mean 缓存进短时环形历史；MEM 不为当前观测
    #   额外跑 forward）
    session = policy.new_session()  # 连接=episode 边界，同 session 保历史
    for t in (4, 5, 6, 7):
        o = obs_at(t)
        o["mem_texts"] = [gripper_text(t)]  # 真实测量文本 -> 长时库
        policy.infer(o, session)
    print(f"history_len={policy.mem.history_len_current} bank_size={len(policy.mem.bank)}", flush=True)
    check("真实 history 输入（4 步真实 Piper 观测，来源 ep_000001 步 4..7）",
          policy.mem.history_len_current == 4,
          f"history_len={policy.mem.history_len_current} source=vlm_prefix_mean(2048D,no robot state)")
    check("真实文本库写入（真实 state 测量文本，冻结 embed 编码）",
          len(policy.mem.bank) == 4, f"bank_size={len(policy.mem.bank)}")

    # ================================================== 当前步：记忆注入生效
    mem_actions_a = np.asarray(
        policy.infer(dict(cur), session, noise=noise.clone())["full_actions"]
    )
    report = policy.monitor.report()
    hook = report["hooks"]["mem_inject"]
    stats = policy.mem.last_stats
    diff_a = float(np.abs(mem_actions_a - base_actions).max())
    check("MEM enabled 动作偏离 base（记忆真实影响策略）", diff_a > 0, f"max_abs_diff={diff_a:.4e}")
    check("request-scoped hook ACTIVE", hook["hook_count_this_request"] > 0,
          f"this_request={hook['hook_count_this_request']} total={hook['hook_count_total']} status={hook['status']}")
    check("memory module executed + 记忆 token 非零有限",
          stats.get("injected") and stats.get("token_l2_norm", 0) > 0
          and stats.get("token_nan_count") == 0 and stats.get("token_inf_count") == 0,
          f"norm={stats.get('token_l2_norm'):.3f} shape=(1,{stats.get('num_tokens')},{policy.feature_dim}) "
          f"used_short={stats.get('used_short')} used_long={stats.get('used_long')}")
    mem_tensor = report["tensors"].get("mem_tokens", {})
    mem_mean = mem_tensor.get("mean")
    check("monitor mem_tokens 统计 finite", mem_mean is not None and bool(np.isfinite(mem_mean)),
          f"mean={mem_mean} stats_keys={sorted(k for k in mem_tensor if mem_tensor[k] is not None)}")

    # ================================================== 同 obs 不同真实历史 -> 记忆变化
    tokens_hist_a = policy.mem.compute_tokens().detach().clone()
    hist_a = list(policy.mem._history)
    session_b = policy.new_session()  # episode 边界：清短时，保长时库
    for t in (12, 13, 14, 15):
        policy.infer(obs_at(t), session_b)
    tokens_hist_b = policy.mem.compute_tokens().detach()
    token_diff = float((tokens_hist_a - tokens_hist_b).abs().max())
    mem_actions_b = np.asarray(
        policy.infer(dict(cur), session_b, noise=noise.clone())["full_actions"]
    )
    hist_action_diff = float(np.abs(mem_actions_b - mem_actions_a).max())
    check("same current obs + 不同真实历史 -> memory token 变化", token_diff > 0,
          f"token_max_diff={token_diff:.4e}")
    check("same current obs + 不同真实历史 -> action 变化", hist_action_diff > 0,
          f"max_abs_diff={hist_action_diff:.4e}")

    # ================================================== Policy impact
    base64, mem64 = base_actions.astype(np.float64), mem_actions_a.astype(np.float64)
    d = mem64 - base64
    cos = float(np.dot(base64.ravel(), mem64.ravel())
                / (np.linalg.norm(base64) * np.linalg.norm(mem64)))
    check("impact 指标", diff_a > 0,
          f"mean_abs={np.abs(d).mean():.4e} max_abs={np.abs(d).max():.4e} "
          f"l2={np.linalg.norm(d):.4e} cos={cos:.6f}")

    # ================================================== forward 数 + 性能
    perf = report["performance"]
    check("extra_forward_count = 0（MEM 用缓存表征，无额外 VLA forward）",
          perf["extra_forward_count"] == 0,
          f"forward_count={perf['forward_count']} backward={perf['backward_count']}")
    policy.mem.active = False
    policy.mem.reset_episode()
    def bench(n=3):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            policy.infer(dict(cur), session_b, noise=noise.clone())
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(ts))
    base_ms = bench()
    # 恢复历史 A 再测（active 状态下注入路径）
    policy.mem._history.extend(hist_a)
    policy.mem.active = True
    mem_ms = bench()
    check("性能记录", True, f"base={base_ms:.1f}ms mem={mem_ms:.1f}ms（+{100*(mem_ms-base_ms)/base_ms:.1f}%）"
          f" gpu_peak={perf['gpu_memory_peak_mb']}MB")

    # ================================================== 最小 backward（原生 flow-matching loss）
    print("\n== 最小 backward（π0.5 原生 loss，1 batch / 2 steps）==", flush=True)
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    model = policy.policy.model
    for p in model.parameters():
        p.requires_grad_(False)
    policy_obs = policy.preprocessor(policy._build_policy_observation(dict(cur)))
    batch = dict(policy_obs)
    # 真实执行动作窗口（ep_000001 t=8 起的剩余 8 步）按 LeRobot episode 末尾
    # 补齐语义 pad 到 chunk_size=50（重复末动作），pad_vector 内部补 14->32 维
    real_window = executed[8:]
    pad_len = policy.config.chunk_size - len(real_window)
    window = np.concatenate([real_window, np.repeat(real_window[-1:], pad_len, axis=0)], axis=0)
    batch["action"] = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1,50,14)
    # LeRobot 训练由 accelerate/数据加载保证 batch 在 GPU；此处手动对齐到模型设备
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    torch.manual_seed(20260914)
    loss, loss_dict = policy.policy.forward(batch)
    check("native loss finite", bool(torch.isfinite(loss)), f"loss={float(loss):.6f}")
    loss.backward()
    grads = [p.grad for p in policy.mem.parameters() if p.grad is not None]
    grad_norm = float(sum(g.norm().item() for g in grads))
    check("梯度流向 MEM 参数（经注入的记忆 token）", len(grads) > 0 and grad_norm > 0
          and all(torch.isfinite(g).all() for g in grads),
          f"n_grad_params={len(grads)}/{len(list(policy.mem.parameters()))} grad_norm={grad_norm:.3e}")
    optimizer = torch.optim.AdamW(policy.mem.parameters(), lr=1e-4)
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    mem_param_ids = {id(p) for p in policy.mem.parameters()}
    check("optimizer 只持有 MEM 参数", opt_param_ids == mem_param_ids)
    before = [p.detach().clone() for p in policy.mem.parameters()]
    for _ in range(2):
        optimizer.zero_grad()
        torch.manual_seed(20260915)
        l2, _ = policy.policy.forward(batch)
        l2.backward()
        optimizer.step()
    changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, policy.mem.parameters()))
    check("MEM 参数 changed=true（2 steps 后）", changed)
    snap = policy.monitor.report()["parameters"]["snapshots"]
    check("base VLM changed=false", snap["base_vlm"]["changed"] == [])
    check("base Action Expert changed=false", snap["action_expert"]["changed"] == [])
    mem_changed = policy.monitor.diff_snapshot("mem_module")
    check("monitor 参数所有权：仅 mem_module changed",
          len(mem_changed.get("changed", [])) > 0,
          f"mem changed {len(mem_changed.get('changed', []))} params")

    # ================================================== save / load
    print("\n== Save / Load（不触碰 π0.5 150000）==", flush=True)
    # bench 阶段的 policy.infer 还会向 history 推当前观测；比较前先对齐为
    # 同一输入（hist_a），否则两边短时记忆不同，bit 级比较无意义
    policy.mem._history.clear()
    for f in hist_a:
        policy.mem.push_history(f)
    path = "/tmp/mem_b3_state.pt"
    policy.mem.save(path)
    from robo_int.arch.mem import MEMModule
    fresh = MEMModule(d_model=policy.feature_dim, history_len=4, num_tokens=4, bank_capacity=16)
    fresh.load(path)
    fresh.encode_text = policy._encode_memory_text
    fresh.to("cuda")
    # 对齐 policy.mem 当前真实短时历史：perf bench 的 infer 会像真实服务一样 push，
    # deque(maxlen) 已漂移，不能用旧 hist_a 快照冒充"同输入"
    for f in policy.mem._history:
        fresh.push_history(f)
    t_ref = policy.mem.compute_tokens().detach()
    t_new = fresh.compute_tokens().detach()
    check("save->new instance->load 同输入同输出（bit 级）",
          float((t_ref - t_new).abs().max()) == 0.0,
          f"max_diff={float((t_ref - t_new).abs().max()):.3e}")
    q = list(policy.mem._history)[-1].cpu().numpy()
    check("文本库 reload 后检索一致",
          fresh.bank.retrieve(q, k=1)[0][0] == policy.mem.bank.retrieve(q, k=1)[0][0],
          f"top1={fresh.bank.retrieve(q, k=1)[0][0][:40]!r}")

    # ================================================== 总结
    print("\n===== MEM Functional Test =====", flush=True)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"===== {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} =====", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
