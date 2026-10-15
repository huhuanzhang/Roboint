> 以下为 2026 年 9 月的原测试记录，依赖当时的硬件、外部 checkpoint 和参考服务；不代表任意新环境已复测。

# 真机验证记录（2026-09-10，Piper X + π0.5）

**验证环境**：双臂 Piper（左 can0 / 右 can1，14D = 每臂 6 关节 rad + 夹爪 m，左臂在前）；
RealSense head / left_wrist / right_wrist（640×480@30）；π0.5 叠衣服 checkpoint
`pi05_piper_fold_clothes_full_finetune/checkpoints/150000`（引用不复制，经
`smolvla_piper_control_gui` 的 policy server @127.0.0.1:8011）；安全参数
（max_abs_delta=0.01 rad/步、gripper=0.003 m/步、smoothing=1.0）取自该 GUI 的已验证 profile。

**验证性质**：功能链路 dry-run（真机读取、真机推理、真数据落盘），**不含**算法效果
（成功率未测）、live 运动（全程 `command_sent=false`）与训练。

| 阶段 | 范围 | 结果 |
|---|---|---|
| F0 硬件 smoke test | CAN 双臂 200Hz 反馈 + 真实 14D state + 三路相机 → π0.5 → (8,14) action → 安全链 → dry-run 记录 | ALL PASS（`scripts/piper_pi05_dry_run.py`，硬件级独立于 generic 管线） |
| F1 generic pipeline | `eval_on_robot.py → factory → BaseRobot/BaseVLA/BaseTask → rollout`，`robot=piper_dual_arm model=pi05_server task=fold_clothes` | PASS（24 步 dry-run；动作块经 `execute_chunk` 消费，步数预算正确） |
| F2 data 管线 | rollout → Episode（含 policy_actions/timestamps/instruction/dry_run 标记）→ score_and_clean → LeRobot 转换 | PASS（人工标 success 后 composite=0.72 kept；npz md5 清洗前后不变；LeRobot 数据集回读 schema 正确） |
| F3 arch 挂件 | A 类（external/runtime）`VLAWrapper` 挂载机制，`arch=none/identity` Hydra 切换 | PASS（IdentityArch 数值 bit 级一致；真机 arch=identity 16 步 dry-run；robot/task/rollout 零改动） |
| F4 RLT | Episode→transition→SAC update→checkpoint→generic 在线入口（`train_rl.py algo=rlt`） | **infra PASS**：真实 14D Piper transition 过最小 update，冻结边界/存取/入口全过（80 tests）；**真实 π0.5 表征 BLOCKED BY SERVER BOUNDARY**（`extract_features` 未实现，特征用 MockVLA stand-in 验证链路） |
| F5 native π0.5 server | fork `smolvla_policy_server.py` → `robo_int/models/pi05/server.py`（原文件不动），加 VLM prefix 特征通道，真实 feature→RLT | **PASS**：数值 parity **bit 级一致**（同 obs+同 noise，max_abs_diff=0.0）；feature=`vlm_prefix_mean`（PaliGemma prefix hidden 均值池化，2048 维，prefix_len=968，单次 forward 已产生）；一次请求同返 actions+features；真实 Piper observation→π0.5 feature→RLTokenHead→actor/critic→3 次 update 全过（86 tests） |
| F6 SARM2 reward | 统一 `BaseRewardProvider` 契约 + `Sarm2RewardProvider`；真实 episode + 真实 π0.5 feature 过奖励管线 | **管线 PASS**：[vlm_prefix_mean;state]=2062D 输入，value/stage/reward 形状与有限性全过，两阶段训练路径功能可用，reward 可灌入 RLT replay 且来源可追溯（relabel 副本含 provenance，`not_for_training=true`）；**reward 本身不可用**——SARM2 未训练（无示语料），value≈常数是诚实预期，不算可用 RL reward（96 tests） |
| F7 内部监测 | `ModelInternalMonitor`（fork server 内，`--monitor` 开关，默认关）：hook 健康/张量统计/动作影响/梯度与参数 checksum/分段耗时与 forward 计数 | **PASS**：开关均 bit 级 parity（0.0 diff）；vlm hook ACTIVE (1,968,2048)；stats 全 finite；forward/extra/backward 计数正确（extra=0）；GPU 显存可读；FakeHook 静默失效单测显式 `HOOK INACTIVE / FAIL`。修复 checksum 整参 float64 拷贝 bug（峰值显存 17.8→9.5GB）。B 类模块（MEM/attention/LoRA/TTT）以后统一复用此监测（105 tests） |
| F8 monitor 加固 + LoRA B 类 | request-scoped hook 健康（warmup 后失效可检出）/ timing 残差（flow/post/other）/ 参数所有权五计数；首个真实 B 类：π0.5 PEFT LoRA（官方默认 target） | **ALL PASS**：36 个 target Linear（expert self_attn q/v + action in/out），LoRA 626,688 参数（0.015%）；零权重 parity **0.0**；π0.5 原生 flow-matching loss 3 步训练（loss/backward/step/冻结全证）；adapter 独立存取 reload parity 0.0；+9.3% 推理耗时、extra_forward=0（`scripts/pi05_lora_functional_test.py`，106 tests） |
| B2 Action-query→Vision attention | model-internal 挂件：按层门控 Action Expert 对视觉 prefix KV 的 value 贡献（`robo_int/models/pi05/action_query_vision.py`；server CLI `--arch-action-query-vision --aqv-gate`） | **ALL PASS**：审计确认 expert 本就经 prefix KV attend 全部 968 token（基座机制，不重复实现）；相机映射差分探针（改第 i 路→仅第 i 个 256-token 切片变化，语言段 max|diff|=0.0）；gate=0 bit 级 parity=0.0；gate=0.05 真实激活 max_abs=8.1e-3（cos=0.999963）；monitor hook ACTIVE + per-camera V mass（head .845/left .768/right .812/lang 1.148）；extra_forward=0、+1.1% 耗时；采样路径手动复现 backward：梯度流向 18 维 layer_gates，base VLM/Expert 参数零改动；Piper dry-run exit=0 steps=8 command_sent=0（`scripts/pi05_aqv_functional_test.py`：7 项离线单测 + 17 项 GPU 检查；回归 104 passed + 10 环境门 skip）；2026-09-12 真机 live smoke 24 cycles 通过（用户亲自运行，命令真实下发、结束回安全姿态并去使能） |
| B3 多尺度具身记忆 MEM | 审计确认原为纯 stub（README 有设计、无实现，如实记录）；本阶段补最小真实实现 `robo_int/arch/mem/memory.py`（短时=缓存 vlm_prefix_mean 历史的时间交叉注意力，长时=文本库写/检索/FIFO 遗忘，冻结 tokenizer 查表编码文本；注入点=实例属性遮蔽 `embed_prefix`，推理与原生训练 forward 同路径）；server CLI `--arch-mem --mem-active --mem-history-len --mem-tokens` | **ALL PASS**：双零效应 parity（disabled 与 active+空记忆均 **bit 级 0.0**）；真实历史 ep_000001 步 4..7（vlm_prefix_mean 2048D，不含 robot state，未硬拼 proprio）+ 真实夹爪测量文本写库；同 obs 不同历史→token diff 1.5e-2/action diff 2.3e-2；impact mean 3.7e-2 max 2.4e-1 cos 0.9555；extra_forward=0、+1.5% 耗时；π0.5 原生 flow-matching loss backward（loss=1.685 finite，梯度到 11/11 MEM 参数 grad_norm 3.9e2，AdamW 2 步后 MEM changed、base VLM/Expert 零改动、monitor 所有权证明）；save→新实例→load **bit 级 0.0**（134MB=仅 MEM 参数）+ 文本库检索一致；Piper dry-run exit=0 steps=12 command_sent=false（双臂上电真实读数+真相机 3 路），monitor probe：请求级时序=先推理后 push 当前 obs，REQ0 空记忆不注入、REQ1-3 注入且 history 1→2→3、hook total 1→4、extra_forward 全 0。修复 attach 子模块泄漏 bug（`object.__setattr__` 防 π0.5 全量进 state_dict/parameters）（`scripts/pi05_mem_functional_test.py`：9 项离线单测 + 22 项 GPU 检查；回归 113 passed + 2 skip + 8 环境门 error——外来 8012 server 元数据缺字段所致，非代码回归） |
| B4 TTT-Linear 快权重层 | 审计确认原为纯 stub（README 有设计、无实现，如实记录）；本阶段补最小真实实现 `robo_int/arch/ttt/ttt_layer.py`（arXiv:2407.04620 线性 TTT 形式：块内 mini-batch 在线自监督重建 ‖Wk−v‖²，W←W−lr·(1/C)Σ(Wk−v)kᵀ，k/q 逐 token 单位归一（数值稳定，尺度由可学习投影吸收），块内查询共享块起点 W；输出 z=Wq 经 wo/LayerNorm 后按 gate 残差注入 prefix token 流；persist 模式跨请求保快权重、session 边界 reset；fp32 内核）；注入点与 MEM 同面（embed_prefix 实例遮蔽，可与 AQV/MEM 叠加）；server CLI `--arch-ttt --ttt-active --ttt-gate --ttt-latent --ttt-lr --ttt-chunk --ttt-persist` | **软件腿 ALL PASS；硬件 dry-run 腿 BLOCKED（臂控未上电：CAN 静默 status_hz=0.0，非代码问题）**：双零效应 parity（inactive 与 active+gate=0 均 **bit 级 0.0**）；真实激活 gate=0.05 max_abs=6.7e-3（cos=0.999984），快权重真实在线更新（w 0→20.6，968 token/16 chunks；非 persist 每请求 bit 级重开，persist 跨请求 20.6→27.1 且 episode 边界重置）；重建 loss finite；π0.5 state_dict 键集不受挂载污染；原生 flow-matching loss backward（loss=0.510 finite，梯度到 9/9 TTT 参数，AdamW 2 步后 TTT changed、base VLM/Expert 零改动、monitor 所有权证明）；save→新实例→load **bit 级 0.0**；extra_forward=0、耗时 ±1% 内、gpu_peak 9.6GB；修复两个真 bug（k 未归一→lr=0.1 快权重发散 w_end=inf/persist 模式 NaN 动作→k/q 单位归一+块均值梯度+lr∈(0,1] 构造约束；`_fast_state` 非注册 buffer→load 后 .to(device) 不搬→状态静默丢弃→forward 内跨设备搬移）（`scripts/pi05_ttt_functional_test.py`：23 项 GPU 检查 + `tests/test_ttt_layer.py`：14 项离线单测，含逐 token 参考递推等价/chunk 参考等价/delta-rule 误差收缩性质；回归 123 passed + 14 skipped，parity 基准 md5 更新为 2026-09-11 有意漂移——RTC+注意力捕获接线，注明日期与理由）；**上电后复跑**：server `--arch-ttt --ttt-active --ttt-gate 0.05 --monitor` + `scripts/piper_pi05_dry_run.py --mode dry_run --cycles 12 --skip-server-launch`（本次软件侧 7/8 门 PASS，仅 robot gate 失败） |

**已知的边界（不是 bug，是当前架构边界）**：

- **π0.5 server 边界（F5 起部分解除）**：Robo_int fork server（`robo_int/models/pi05/server.py`，
  端口 8012，与 reference 8011 并存）已提供 `vlm_prefix_mean` 特征通道（RLT 已用上）；
  **LoRA 已真实接入**（F8，PEFT + π0.5 官方默认 target，见 `scripts/pi05_lora_functional_test.py`）；
  action-query attention（B2）/ MEM（B3）/ TTT-Linear 快权重层（B4，`--arch-ttt`，k/q 逐 token 单位归一+块均值梯度的 mini-batch 快权重，gate=0 零效应）已接入——fork 现由本仓库维护，可自由加 hook，
  监测与参数所有权基建（F7/F8）已就绪。
  **Attention 门控已真实接入**（B2，AQV：expert 视觉 KV 的 V-side 按层门控，见 `scripts/pi05_aqv_functional_test.py`）；
  parity 由 `tests/test_pi05_server_parity.py` 守护（reference md5 基准 + wire metadata +
  可选 `ROBO_INT_PI05_PARITY=1` 数值级，当前 bit 级一致）。
- `vlm_prefix_mean` 是视觉+语言上下文表征（不含 state——π0.5 的 state 在 action expert 侧）；
  需要 state 的算法可自行拼接。
- SARM2 当前是 mock 级移植且**未训练**（叠衣服无示教/原语标注语料）：reward 管线可用，
  reward 数值不可用。缺口 = 带原语/进度标注的示教数据（哪怕几十条人工标注 episode）。
- generic 路径为同步推理（chunk 边界 ~150ms 停顿）；GUI 的异步 prefetch / RTC 未接入。
- dry-run episode 已标 `training_eligible=false`，只作链路验证，不进训练集。
- LeRobot 转换当前走 lerobot-piper-2 旧版 API（本机 Python 3.11 限制，molmoact2 新版需 ≥3.12），
  转换器已做双 API 兼容。

**复现**（先安装项目并按仓库根目录的 `.env.example` 填写本机路径）：

```bash
PY=python
export PYTHONPATH="$LEROBOT_PIPER_ROOT:$PWD" NO_PROXY=127.0.0.1,localhost

# F1/F2 录制（dry-run；F3 加 arch=identity）
$PY scripts/eval_on_robot.py robot=piper_dual_arm model=pi05_server task=fold_clothes \
    task.max_episode_steps=16 eval.ckpt=null eval.num_episodes=1 eval.save_episodes=true \
    data.data_dir=data/piper_dryrun
$PY scripts/score_and_clean.py task=fold_clothes data.data_dir=data/piper_dryrun
$PY scripts/convert_to_lerobot.py data.data_dir=data/piper_dryrun data.lerobot_out=data/lerobot_piper_dryrun
```

回归测试：`pytest tests/` → 123 passed + 14 skipped（全部离线，不依赖真机/GPU；真机链路由上命令人工触发）。
