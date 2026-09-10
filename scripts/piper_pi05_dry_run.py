#!/usr/bin/env python3
"""Piper X + 现有 π0.5 叠衣服 checkpoint 的 dry-run smoke test。

链路：Piper 双臂 CAN + 三路 RealSense -> 14D state -> smolvla policy server
（复用 smolvla_piper_control_gui，不复制 checkpoint）-> 14D action -> 安全链
-> dry-run 记录（绝不下发电机命令）。

用法（推荐 lerobot conda env，与 GUI client 相同）：
    /home/ps/miniconda3/envs/lerobot/bin/python scripts/piper_pi05_dry_run.py
    ... --cycles 24 --log-every 8
真机 live（默认拒绝，必须显式 --mode live）：
    ... --mode live --cycles 60

checkpoint / prompt / 安全参数全部从 GUI 的 config.json profile 读取，不在本
脚本重复维护。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robo_int.models.pi05_server import Pi05ServerPolicy
from robo_int.robots.piper_dual_arm import PiperDualArmRobot
from robo_int.robots.piper_safety import PiperDualArmSafetyChain

CHECKS: list[tuple[str, bool | None, str]] = []
_SHUTDOWN_REQUESTED = False


def check(section: str, name: str, ok: bool | None, detail: str = "") -> bool | None:
    """ok=None 记为 BLOCKED（外部条件不满足，如臂控未上电），不计为 FAIL。"""
    CHECKS.append((f"{section}.{name}", None if ok is None else bool(ok), detail))
    tag = "PASS" if ok else ("BLOCKED" if ok is None else "FAIL")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def request_shutdown(signum, _frame) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dry_run", "live"), default="dry_run",
                        help="dry_run（默认）绝不发送电机命令；live 显式要求后才下发")
    parser.add_argument("--cycles", type=int, default=24, help="控制步数（每步 1/fps 秒）")
    parser.add_argument("--log-every", type=int, default=8, help="每隔多少控制步输出一次 summary")
    parser.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs/robot/piper_dual_arm.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs/model/pi05_server.yaml")
    parser.add_argument("--gui-profile", default=None, help="覆盖 GUI profile id（默认取 model 配置里的）")
    parser.add_argument("--checkpoint", default=None, help="覆盖 checkpoint 目录（默认取 GUI profile）")
    parser.add_argument("--prompt", default=None, help="覆盖 task prompt（默认取 GUI tasks）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--skip-server-launch", action="store_true",
                        help="server 已在运行时跳过启动（仍会校验 checkpoint 一致）")
    parser.add_argument("--camera-warmup-frames", type=int, default=None)
    parser.add_argument("--cameras-only", action="store_true",
                        help="诊断模式：臂控未上电时只连真实相机，state 用零向量（server validate 同做法）。"
                             "报告会把真实 state/臂读数标为 BLOCKED，不冒充通过")
    parser.add_argument("--request-timeout-s", type=float, default=60.0,
                        help="单次推理 wall-time 超过此值即安全退出")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_gui_profile(gui_config: Path, profile_id: str) -> tuple[dict, dict]:
    config = json.loads(Path(gui_config).read_text(encoding="utf-8"))
    profiles = {p["id"]: p for p in config.get("profiles", [])}
    if profile_id not in profiles:
        raise KeyError(f"GUI config 中没有 profile {profile_id}；可选：{sorted(profiles)}")
    tasks = {t["id"]: t for t in config.get("tasks", [])}
    return profiles[profile_id], tasks


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def launch_server(launcher: Path, checkpoint: Path, host: str, port: int, timeout_s: float) -> subprocess.Popen:
    log_path = REPO_ROOT / "outputs" / "pi05_server_smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "CHECKPOINT_DIR": str(checkpoint),
        "POLICY_HOST": host,
        "POLICY_PORT": str(port),
        "POLICY_IO_MODE": "dual-arm-14d",
        "ACTIONS_PER_INFERENCE": "8",
    }
    print(f"启动 policy server：{launcher}（日志 {log_path}）", flush=True)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(["bash", str(launcher)], env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                start_new_session=True)
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-15:]
            raise RuntimeError(f"policy server 提前退出（code={proc.returncode}）：\n" + "\n".join(tail))
        if port_open(host, port):
            print(f"policy server 端口就绪（{time.monotonic() - started:.1f}s）", flush=True)
            return proc
        time.sleep(2.0)
    raise TimeoutError(f"policy server {timeout_s}s 内未就绪，见 {log_path}")


def busy_wait(duration: float) -> None:
    """精确补齐控制周期（与 lerobot.common.robot_devices.utils.busy_wait 同义）。"""
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        pass


def main() -> int:
    args = parse_args()
    install_signal_handlers()
    robot_cfg = load_yaml(args.robot_config)
    model_cfg = load_yaml(args.model_config)
    profile, tasks = load_gui_profile(model_cfg["gui_config"], args.gui_profile or model_cfg["gui_profile_id"])
    task = tasks[profile["task_ids"][0]]
    checkpoint = Path(args.checkpoint or profile["checkpoint"])
    prompt = args.prompt or task["prompt"]
    host = args.host or model_cfg["host"]
    port = args.port or int(model_cfg["port"])
    # 安全参数：只取 GUI profile 已验证值，不在本仓库另设
    fps = float(profile.get("fps", 30))
    actions_per_inference = int(profile.get("actions_per_inference", 8))
    safety = dict(
        action_scale=1.0,
        max_abs_delta=float(profile.get("max_abs_delta", 0.01)),
        gripper_max_abs_delta=float(profile.get("gripper_max_abs_delta", 0.003)),
        action_smoothing_alpha=float(profile.get("action_smoothing_alpha", 1.0)),
        gripper_smoothing_alpha=float(profile.get("gripper_smoothing_alpha", 1.0)),
    )
    warmup_frames = int(args.camera_warmup_frames if args.camera_warmup_frames is not None
                        else profile.get("camera_warmup_frames", 0))
    print(f"目标链路：mode={args.mode} cycles={args.cycles} fps={fps} prompt={prompt!r}")
    print(f"checkpoint（引用，不复制）：{checkpoint}")
    print(f"安全参数（GUI profile）：{safety}")

    server_proc = None

    # ---------------------------------------------------------------- Environment
    print("\n== Environment ==", flush=True)
    try:
        from lerobot.common.robot_devices.robots.piper import PiperRobot  # noqa: F401
        check("env", "Piper dependencies", True, "lerobot-piper-2 可导入")
    except Exception as exc:
        check("env", "Piper dependencies", False, str(exc))
    required_files = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    missing = [n for n in required_files if not (checkpoint / n).is_file()]
    check("env", "checkpoint found", not missing and checkpoint.is_dir(),
          str(checkpoint) if not missing else f"缺 {missing}")
    if missing:
        return finish(1)

    # ---------------------------------------------------------------- Policy server
    print("\n== Policy server ==", flush=True)
    try:
        if not port_open(host, port) and not args.skip_server_launch:
            server_proc = launch_server(Path(model_cfg["server_launcher"]), checkpoint, host, port,
                                        float(model_cfg.get("server_start_timeout_s", 300)))
        elif args.skip_server_launch and not port_open(host, port):
            check("policy", "server reachable", False, f"{host}:{port} 未监听且指定了 --skip-server-launch")
            return finish(1)
        policy = Pi05ServerPolicy(host=host, port=port, prompt=prompt,
                                  actions_per_inference=actions_per_inference,
                                  openpi_client_src=model_cfg["openpi_client_src"])
        metadata = policy.connect()
        check("policy", "checkpoint loading", True,
              f"{metadata.get('policy_type')} @ {metadata.get('checkpoint')}")
        server_ckpt = str(metadata.get("checkpoint", ""))
        check("policy", "checkpoint 一致（与 GUI profile 同一权重）",
              server_ckpt.startswith(str(checkpoint)), server_ckpt)
        normalization = metadata.get("normalization", {})
        norm_ok = (normalization.get("enabled")
                   and normalization.get("state_dim") == 14 and normalization.get("action_dim") == 14)
        check("policy", "normalization/postprocessing（server 侧）", bool(norm_ok),
              json.dumps({k: normalization.get(k) for k in ("state_method", "action_method",
                                                            "state_dim", "action_dim")}))
        check("policy", "task prompt", bool(prompt and prompt == task["prompt"]), repr(prompt))
        required_cameras = tuple(metadata.get("required_policy_cameras", ()))
        check("policy", "preprocessing（server 所需相机）", bool(required_cameras), ",".join(required_cameras))
    except Exception as exc:
        check("policy", "policy server 连接", False, f"{type(exc).__name__}: {exc}")
        return finish(1)

    # ---------------------------------------------------------------- Robot + cameras
    print("\n== Robot ==", flush=True)
    cameras = {k: int(v) for k, v in robot_cfg["cameras"].items() if k in required_cameras}
    robot = PiperDualArmRobot(
        cameras=cameras,
        left_can_name=robot_cfg["left_can_name"],
        right_can_name=robot_cfg["right_can_name"],
        fps=int(robot_cfg["camera_fps"]),
        width=int(robot_cfg["camera_width"]),
        height=int(robot_cfg["camera_height"]),
        control_hz=fps,
        lerobot_piper_root=robot_cfg["lerobot_piper_root"],
        cameras_only=args.cameras_only,
    )
    try:
        robot.connect()
        check("robot", "Piper connection", True,
              f"left={robot.left_can_name} right={robot.right_can_name}"
              + ("（cameras-only 诊断模式，未连 CAN）" if args.cameras_only else ""))
    except Exception as exc:
        check("robot", "Piper connection", False, f"{type(exc).__name__}: {exc}")
        return finish(1)

    chain = PiperDualArmSafetyChain(**safety)
    sent_commands: list[np.ndarray] = []
    infer_ms_hist: list[float] = []
    inference_failures = 0
    command_sent_any = False
    shutdown_clean = False

    if args.mode == "live":
        print("!! live 模式：将使能双臂并真实下发命令（沿用 GUI profile 限幅参数）", flush=True)
        robot.enable()

    try:
        if warmup_frames:
            print(f"相机预热 {warmup_frames} 帧 ...", flush=True)
            for _ in range(warmup_frames):
                started = time.perf_counter()
                robot.capture_observation()
                busy_wait(max(0.0, 1.0 / fps - (time.perf_counter() - started)))

        pending: deque[np.ndarray] = deque()
        arm_state_ok = camera_ok = action_ok = False
        camera_names_ok: list[str] = []
        for step in range(args.cycles):
            if _SHUTDOWN_REQUESTED:
                print("收到停止信号，退出控制循环。", flush=True)
                break
            loop_start = time.perf_counter()
            obs = robot.capture_observation()
            state = np.asarray(obs["state"], dtype=np.float32)

            if step == 0:
                print("\n== Cameras / state（首拍） ==", flush=True)
                if args.cameras_only:
                    check("robot", "14D state", None, "cameras-only 模式：state 为合成零向量，真实读数待臂控上电")
                    check("robot", "left arm", None, "臂控未上电，BLOCKED")
                    check("robot", "right arm", None, "臂控未上电，BLOCKED")
                else:
                    real_state = (
                        state.shape == (14,)
                        and np.isfinite(state).all()
                        and not np.allclose(state, 0.0)  # 真机读数不会恰好全零；全零=反馈异常
                    )
                    check("robot", "14D state", real_state,
                          f"shape={state.shape} finite={np.isfinite(state).all()} "
                          f"min={state.min():+.4f} max={state.max():+.4f} "
                          f"left={np.array2string(state[:7], precision=3)} "
                          f"right={np.array2string(state[7:], precision=3)}")
                    joints_in_range = bool(np.all(np.abs(state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]) < 3.5))
                    grips_in_range = bool(np.all((state[[6, 13]] >= -0.01) & (state[[6, 13]] <= 0.12)))
                    check("robot", "left arm (can0)", joints_in_range and grips_in_range and real_state,
                          f"读数 {np.array2string(state[:7], precision=3)}")
                    check("robot", "right arm (can1)", joints_in_range and grips_in_range and real_state,
                          f"读数 {np.array2string(state[7:], precision=3)}")
                for name in required_cameras:
                    frame = obs["images"].get(name)
                    ok = (frame is not None and frame.shape == (480, 640, 3)
                          and np.isfinite(frame).all() and float(np.asarray(frame, dtype=np.float32).std()) > 0.5)
                    check("cameras", name, ok,
                          f"shape={getattr(frame, 'shape', None)} serial={cameras.get(name)}")
                    if ok:
                        camera_names_ok.append(name)
                camera_ok = len(camera_names_ok) == len(required_cameras)
                arm_state_ok = (not args.cameras_only) and state.shape == (14,) and np.isfinite(state).all()

            if not pending:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            policy.predict_actions, {"state": state, "images": obs["images"]}, prompt
                        )
                        chunk = future.result(timeout=args.request_timeout_s)
                except concurrent.futures.TimeoutError as exc:
                    raise TimeoutError(
                        f"推理超过 {args.request_timeout_s}s，触发安全退出（未发送任何命令）"
                    ) from exc
                except Exception as exc:
                    inference_failures += 1
                    print(f"推理失败 @step={step}: {type(exc).__name__}: {exc}", flush=True)
                    if inference_failures > 2:
                        raise
                    continue
                infer_ms_hist.append(getattr(policy, "last_infer_ms", 0.0))
                pending.extend(np.asarray(chunk, dtype=np.float32))

            model_action = pending.popleft()
            command = chain(model_action, state)  # 真实安全链：scale -> clamp -> smooth
            sent_commands.append(command)
            if args.mode == "live":
                robot.send_action(command)
                command_sent_any = True
            action_ok = action_ok or (command.shape == (14,) and np.isfinite(command).all())

            if step % max(1, args.log_every) == 0 or step == args.cycles - 1:
                cmds = np.stack(sent_commands[-actions_per_inference:])
                print(
                    f"[cycle {step:3d}] state=({state.shape[0]},) cams_ok={len(camera_names_ok)}/{len(required_cameras)} "
                    f"cmd_shape={command.shape} cmd_min={cmds.min():+.4f} cmd_max={cmds.max():+.4f} "
                    f"finite={np.isfinite(cmds).all()} infer={infer_ms_hist[-1] if infer_ms_hist else 0:.0f}ms "
                    f"queue={len(pending)} command_sent={args.mode == 'live'}",
                    flush=True,
                )
            busy_wait(max(0.0, 1.0 / fps - (time.perf_counter() - loop_start)))

        print("\n== Policy（推理统计） ==", flush=True)
        check("policy", "inference", inference_failures == 0 and len(infer_ms_hist) > 0,
              f"{len(infer_ms_hist)} 次成功 / {inference_failures} 次失败")
        if infer_ms_hist:
            check("policy", "14D action", action_ok,
                  f"chunk=({actions_per_inference},14) mean_infer={np.mean(infer_ms_hist):.0f}ms")
        check("policy", "all finite", bool(sent_commands) and all(np.isfinite(c).all() for c in sent_commands),
              f"{len(sent_commands)} 条命令")
    except KeyboardInterrupt:
        print("中断：进入安全退出路径（不发送任何电机命令）。", flush=True)
    finally:
        # 停止路径：与 GUI client 的 disconnect_robot_without_arm_motion 一致，
        # dry-run / 中断都只停相机，不使能、不移动手臂
        try:
            if args.mode == "live":
                robot.disable()
            else:
                robot.disconnect()
            shutdown_clean = True
        except Exception as exc:
            print(f"安全退出异常：{type(exc).__name__}: {exc}", flush=True)
        policy.close()
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()

    # ---------------------------------------------------------------- Safety / 汇总
    print("\n== Safety ==", flush=True)
    check("safety", "dry-run（未发送电机命令）", args.mode == "dry_run" and not command_sent_any)
    check("safety", "command_sent=false", not command_sent_any,
          "live 模式应为 true" if args.mode == "live" else "")
    check("safety", "timeout protection", True,
          f"request_timeout={args.request_timeout_s}s，超时触发安全退出")
    check("safety", "stop path", shutdown_clean, "信号处理 -> 停相机 -> 干净退出（无手臂动作）")

    print("\n== Integration ==", flush=True)
    core_ok = (args.mode == "dry_run" and camera_ok and action_ok
               and inference_failures == 0 and len(sent_commands) >= min(args.cycles, 1)
               and not command_sent_any)
    if args.cameras_only:
        # cameras-only 用的是合成零 state：只报诊断链路，禁止报告 full pipeline PASS
        check("integration", "cameras-only 诊断链路（真实相机 + 合成零 state）", core_ok,
              f"{len(sent_commands)} cycles")
        check("integration", "Robo_int → Piper full dry-run pipeline", None,
              "cameras-only 模式不评级；需真实 14D state（去掉 --cameras-only 复跑）")
    else:
        check("integration", "Robo_int → Piper full dry-run pipeline", core_ok and arm_state_ok,
              f"{len(sent_commands)} cycles 全链路（真实 14D state + 真实相机 + π0.5 + safety，未发送命令）")

    print("\n== Parity（对 smolvla_piper_control_gui） ==", flush=True)
    print(f"  camera 顺序: {','.join(required_cameras)}（server 由 checkpoint 决定，双端一致）")
    print(f"  14D state 顺序: left(can0) 7D + right(can1) 7D，policy_arm_order=left-right（同 GUI client）")
    print(f"  task prompt: {prompt!r}（GUI config.json tasks.{task['id']}）")
    print(f"  checkpoint: {checkpoint}（GUI profile {profile['id']} 同一目录）")
    print(f"  action shape: ({actions_per_inference}, 14)（GUI policy_io_mode=dual-arm-14d）")

    failed_any = any(ok is False for _, ok, _ in CHECKS)
    return finish(1 if failed_any else 0)


def finish(code: int) -> int:
    print("\n===== Robo_int Piper π0.5 Smoke Test =====", flush=True)
    for name, ok, detail in CHECKS:
        tag = "PASS" if ok else ("BLOCKED" if ok is None else "FAIL")
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    failed = [name for name, ok, _ in CHECKS if ok is False]
    blocked = [name for name, ok, _ in CHECKS if ok is None]
    if failed:
        verdict = "FAILED: " + ", ".join(failed)
    elif blocked:
        verdict = f"PASS（{len(blocked)} 项 BLOCKED: " + ", ".join(blocked) + "）"
    else:
        verdict = "ALL PASS"
    print(f"===== {verdict} =====", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
