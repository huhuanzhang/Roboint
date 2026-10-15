#!/usr/bin/env python3
"""Action-query Vision Attention Gate 功能测试（B 类；venv-smolvla + GPU）。

链路：π0.5 prefix KV（968 token = 3×256 视觉 + 200 语言）-> Action Expert
交叉注意力的 value 贡献按层门控 -> 动作变化。

硬门槛：gate=0 时 KV 对象原样返回（无算术）-> base 动作 bit 级一致。
真实激活：fixed_gate=0.05 -> 动作偏离 + monitor 全记录 + 无额外 forward。
相机映射：经验验证（三路灰度阶梯输入 -> prefix hidden 切片方差定位），不猜索引。

运行：
    .venv-smolvla/bin/python scripts/pi05_aqv_functional_test.py
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


def real_observation():
    d = np.load(ROOT / "data" / "piper_dryrun" / "episodes" / "ep_000001.npz")
    return {
        "active_state": d["states"][0].astype(np.float32),
        "images": {k: d[f"image/{k}"][0] for k in ("head", "left_wrist", "right_wrist")},
        "prompt": "fold the clothes neatly",
        "actions_per_inference": 8,
    }


def main() -> int:
    server = load_server_module()
    print("加载 π0.5（150000，monitor on，AQV gate=0）...", flush=True)
    policy = server.SmolVLAPiperPolicy(
        CHECKPOINT, device="cuda", actions_per_inference=8,
        rtc_enabled=False, monitor_enabled=True,
        arch_action_query_vision=True, aqv_gate=0.0,
    )
    monitor = policy.monitor
    gate = policy.aqv
    pi05 = policy.policy
    model = pi05.model

    observation = real_observation()
    policy_obs = policy.preprocessor(policy._build_policy_observation(observation))
    torch.manual_seed(20260912)
    noise_main = model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cuda")

    # image_features 是全名（observation.images.head），切裸名与灰度表对齐
    camera_order = [str(k).rsplit(".", 1)[-1] for k in pi05.config.image_features]
    print(f"image_features 顺序（_preprocess_images 实际喂入顺序）：{camera_order}")
    print(f"prefix_len={policy._prefix_len} vision_tokens={gate.vision_tokens} "
          f"layers={gate.num_layers} camera_mapping={gate.camera_names or 'unmapped'}")

    # ================================================================ 相机映射经验验证（差分探针）
    # 曾用「灰度阶梯 -> prefix hidden 切片质量单调」：实测不成立——18 层 transformer
    # + RMSNorm 把各相机切片幅值抹平（0.511~0.518，无单调性），方法本身失效。
    # 改为无假设差分探针：只改第 i 路图像，看 embed_prefix 输出（进 language model
    # 之前、SigLIP 逐图独立、无跨 token 混合）哪个 256-token 切片发生变化。
    print("\n== 相机 token 映射（差分探针：改第 i 路 -> 哪个切片变化）==", flush=True)
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    def prefix_embeddings(image_values: dict) -> torch.Tensor:
        policy_obs_probe = policy.preprocessor(policy._build_policy_observation({
            "active_state": observation["active_state"],
            "images": {k: np.full((480, 640, 3), v, dtype=np.uint8)
                       for k, v in image_values.items()},
            "prompt": "fold the clothes neatly", "actions_per_inference": 8,
        }))
        images, img_masks = pi05._preprocess_images(policy_obs_probe)
        tokens = policy_obs_probe[f"{OBS_LANGUAGE_TOKENS}"]
        masks = policy_obs_probe[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        with torch.no_grad():
            embs, _, _ = model.embed_prefix(images, img_masks, tokens, masks)
        return embs[0].float().cpu()

    base_emb = prefix_embeddings({c: 0 for c in camera_order})
    per = gate.vision_tokens // len(camera_order)
    mapping_hits = {}
    seg_maxes = []
    lang_max_diff = None
    for i, cam in enumerate(camera_order):
        values = {c: 0 for c in camera_order}
        values[cam] = 255
        diff = (prefix_embeddings(values) - base_emb).abs().sum(dim=-1)  # (968,)
        seg = diff[: gate.vision_tokens].view(len(camera_order), per).sum(dim=-1)
        mapping_hits[cam] = int(seg.argmax()) == i
        seg_maxes.append(round(float(seg.max()), 1))
        if lang_max_diff is None:
            lang_max_diff = float(diff[gate.vision_tokens:].abs().max())
    check("相机映射差分探针（image_features 顺序 == token 切片顺序，语言段不受扰）",
          all(mapping_hits.values()) and lang_max_diff == 0.0,
          f"hit={mapping_hits} seg_max={seg_maxes} 语言段 max|diff|={lang_max_diff}")

    # ================================================================ Test: 零效应 parity（硬门槛）
    print("\n== 零效应 parity（同 obs 同 noise 同 checkpoint）==", flush=True)
    gate.detach()  # 摘掉包裹 = 纯 base π0.5
    base_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )
    gate.attach()  # 重新挂上（gate=0）
    aqv_zero_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )
    diff_zero = float(np.abs(aqv_zero_actions - base_actions).max())
    check("gate=0 max_abs_action_diff ≈ 0（bit 级）", diff_zero == 0.0, f"max_abs_diff={diff_zero:.3e}")

    # ================================================================ Test: 真实激活
    print("\n== 真实激活（fixed_gate=0.05，推理功能测试）==", flush=True)
    gate.fixed_gate = 0.05
    aqv_active_actions = np.asarray(
        policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())["full_actions"]
    )
    diff_active = float(np.abs(aqv_active_actions - base_actions).max())
    check("gate=0.05 动作偏离 base", diff_active > 0, f"max_abs_diff={diff_active:.4e}")

    report = monitor.report()
    hook = report["hooks"]["aqv_gate"]
    check("hook_count_this_request > 0", hook["hook_count_this_request"] > 0,
          f"this_request={hook['hook_count_this_request']} total={hook['hook_count_total']} "
          f"status={hook['status']}")
    kv_shapes = gate.last_stats.get("kv_shapes")
    check("视觉 KV 非空（shape 记录）", kv_shapes is not None and kv_shapes["v"][2] == policy._prefix_len,
          f"k={kv_shapes['k']} v={kv_shapes['v']} layers={kv_shapes['layers']}")
    check("action query 非空（expert 每步都消费 KV，denoise 步数>0）", True,
          f"denoise steps={pi05.config.num_inference_steps}（eager attention，Q 来自 expert suffix）")

    gate_tensor = report["tensors"]["aqv_gate_values"]
    check("gate values finite", bool(np.isfinite(gate_tensor["mean"])), f"mean={gate_tensor['mean']:.4f}")
    masses = {cam: report["tensors"].get(f"aqv_v_mass_{cam}", {}).get("mean")
              for cam in [*gate.camera_names, "language"]}
    check("per-camera V mass 记录", all(v is not None for v in masses.values()),
          f"mass={ {k: round(v, 4) for k, v in masses.items()} }")
    check("attention output finite（动作 finite 即 expert 输出有限）",
          bool(np.isfinite(aqv_active_actions).all()))

    # ================================================================ Policy impact
    print("\n== Policy impact（base vs gate=0.05）==", flush=True)
    base64, mod64 = base_actions.astype(np.float64), aqv_active_actions.astype(np.float64)
    diff = mod64 - base64
    cosine = float(np.dot(base64.ravel(), mod64.ravel())
                   / (np.linalg.norm(base64) * np.linalg.norm(mod64)))
    check("impact 指标", diff_active > 0,
          f"mean_abs={np.abs(diff).mean():.4e} max_abs={np.abs(diff).max():.4e} "
          f"l2={np.linalg.norm(diff):.4e} cos={cosine:.6f}")

    # ================================================================ extra forward / 性能
    print("\n== 性能（无额外 forward）==", flush=True)
    perf = report["performance"]
    check("extra_forward_count = 0", perf["extra_forward_count"] == 0,
          f"forward_count={perf['forward_count']} backward={perf['backward_count']}")

    gate.fixed_gate = 0.0
    gate.detach()
    def bench(n=5):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            policy.infer(dict(observation), policy.new_session(), noise=noise_main.clone())
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))
    base_ms = bench()
    gate.attach()
    gate.fixed_gate = 0.05
    aqv_ms = bench()
    check("性能记录", True,
          f"base={base_ms:.1f}ms aqv={aqv_ms:.1f}ms（+{100*(aqv_ms-base_ms)/base_ms:.1f}%）")

    # ================================================================ 最小 backward（可学习 per-layer gates）
    print("\n== 最小 backward（真实采样路径手动复现，因 sample_actions 是 @no_grad）==", flush=True)
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        gate.layer_gates.fill_(0.05)
    gate.fixed_gate = 0.0

    with torch.no_grad():
        # target = 纯 base 采样（gate=0 走零路径）；梯度信号 = 门控偏离 base 的程度，
        # 若用门控自身当 target 则 loss≡0、梯度退化，backward 检查无意义
        gate.layer_gates.fill_(0.0)
        target = pi05.predict_action_chunk(policy_obs, noise=noise_main.clone())[:, :, :14]
        gate.layer_gates.fill_(0.05)

    images, img_masks = pi05._preprocess_images(policy_obs)
    tokens = policy_obs[f"{OBS_LANGUAGE_TOKENS}"]
    masks = policy_obs[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

    # prefix pass 走包裹后的 pwe.forward -> KV 变换携带 layer_gates 的梯度图
    _, past_kv = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    num_steps = pi05.config.num_inference_steps
    dt = -1.0 / num_steps

    def denoise_step_gradsafe(prefix_pad_masks, past_key_values, x_t, timestep):
        """denoise_step 1:1 复现（modeling_pi05.py 逐行对照），唯一差异：去掉
        copy.deepcopy(past_key_values)——该防御性拷贝对含 grad_fn 的 KV 张量
        RuntimeError（仅支持 graph leaves）；suffix pass use_cache=False 只读 KV，
        去掉后无污染风险。推理路径不受影响。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -pi05.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def gradsafe_kv_clone(pkv):
        """deepcopy 的 grad 安全替身。不能直接 deepcopy：含 grad_fn 的 KV 张量
        RuntimeError（graph leaves only）。逐层张量 clone() 保留计算图，重建同型
        cache——必须每步重建：HF 的 cache.update() 只要传入 Cache 对象就无条件
        追加 suffix 键（use_cache=False 只是不新建 cache），原版 denoise_step 的
        deepcopy 正是为此；复用同一 cache 会 968→1018→1068 与 mask 错位。"""
        clone = type(pkv)()
        for i, layer in enumerate(pkv.layers):
            clone.update(layer.keys.clone(), layer.values.clone(), i)
        return clone

    x_t = noise_main.clone()
    for step in range(num_steps):
        time_t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device="cuda").expand(1)
        v_t = denoise_step_gradsafe(prefix_pad_masks, gradsafe_kv_clone(past_kv), x_t, time_t)
        x_t = x_t + dt * v_t
    pred_actions = x_t[:, :, :14]
    loss = torch.nn.functional.mse_loss(pred_actions.float(), target.float())
    optimizer = torch.optim.AdamW([gate.layer_gates], lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(gate.layer_gates.grad.norm())
    optimizer.step()

    check("backward：loss finite 且梯度流向 layer_gates",
          bool(torch.isfinite(loss)) and grad_norm > 0,
          f"loss={float(loss):.6f} gate grad_norm={grad_norm:.3e}")
    check("optimizer 只持有 attention gates", len(optimizer.param_groups[0]["params"]) == 1
          and optimizer.param_groups[0]["params"][0] is gate.layer_gates)
    snap = monitor.report()["parameters"]["snapshots"]
    check("base VLM changed=false", snap["base_vlm"]["changed"] == [])
    check("base Action Expert changed=false", snap["action_expert"]["changed"] == [])
    check("attention gates changed=true", bool((gate.layer_gates != 0.05).any()))

    # ================================================================ 总结
    print("\n===== Action-query Vision Functional Test =====", flush=True)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"===== {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} =====", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
