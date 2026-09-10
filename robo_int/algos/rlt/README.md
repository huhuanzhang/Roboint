# RLT（RL Token）复现计划

> 论文：[RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/abs/2604.23073)
>（C. Xu, J. Springenberg, M. Equi, A. Amin, A. Esmail, S. Levine, L. Ke，2026，**Physical Intelligence (π)**）
> 项目页：[pi.website/research/rlt](https://www.pi.website/research/rlt)
> ~~注：归属曾误记为 Sakana AI，2026-09-02 查证修正为 Physical Intelligence。~~

## 代码现状（2026-09-02 查证，v2 全文已精读）

- **官方代码：无**（Physical Intelligence 未放出，openpi 仓库不含 RLT）；
- 非官方参考：
  - RLinf 内有 RLT 实现——**已知有 bug**（用户实测），只作对照不作依赖；
  - 社区复现 [Yyshadow/openpi-RLT](https://github.com/Yyshadow/openpi-RLT)（基于 openpi 的 π0.5）；
- → **本目录自复现**，以论文 v2 为准，差异全部标注在下表。

## 核心思想（按论文 §IV）

1. **RL token**（§IV-A）：在冻结 VLA 上加小 transformer encoder（输入 = VLA 最终层全部
   token embedding + 追加 `<rl>` token），取 `<rl>` 位置输出作为压缩读出 z_rl；
   decoder transformer 自回归重建原始 embedding（stop-gradient），构成信息瓶颈。
   **在示教数据上训练 2000–10000 梯度步后冻结**（可选同时 SFT VLA）。
2. **在线 RL**（§IV-B）：轻量 actor-critic 消费 x=(z_rl, 本体感觉 s^p)；
   actor **以 VLA 采样的参考 chunk ã 为条件输入**，输出对 chunk 的精修，
   损失 −Q + β‖a−ã‖²（向 VLA 动作的 BC/KL 式正则），训练时 50% 概率将参考置零
   （reference action dropout，防 actor 退化为复制 ã）。
3. **系统层**（§V）：warmup 用 VLA 参考策略预填 buffer；人工干预覆盖动作并存入 buffer；
   chunk stride-2 子采样提高样本效率；只在任务关键阶段（critical phase）训练 RL。
   真机四个精密操作任务，分钟–小时级数据，最高 3× 提速。

## 与论文的差异（复现标注，按论文 v2 校准）

| # | 论文（arXiv:2604.23073v2） | 本实现（mock 级） | 原因/计划 |
|---|---|---|---|
| 1 | RL token：transformer encoder（全部 token embedding + `<rl>`）+ decoder 自回归重建；**示教数据预训练 2k–10k 步后冻结** | MLP 自编码（encoder 投影 + decoder 重建），**RL 在线阶段联合训练**（critic 优化器内） | mock 特征无 token 序列结构；在线学 token 在 mock 上已验证有效（100% 成功率）。接 π0.5 时改为示教预训练 + 冻结，对齐论文 |
| 2 | 基模 π0.6；RL 状态 = (z_rl, 本体感觉 s^p) | MockVLA 冻结特征；仅 z_rl | mock 特征已含全部 state 信息；接 π0.5 时拼 proprio（对齐差异点见 models/pi05/） |
| 3 | actor 输出 **chunked 动作**（C=10×14 维），**以 VLA 采样参考 chunk 为条件输入**（pass-through）+ BC 正则 β‖a−ã‖² + 50% reference dropout | 单步动作；action_mode=direct/residual（residual 为本仓库设计选择）。**critic 侧已 Markov 化**（2026-09-02 修复）：residual 模式 Q 输入 = [token, executed, base]，act() 采样的 VLA 原动作经 observe() 落回放，target 端对 VLA(s') fresh 采样、actor 端与部署一致取 clip(base+残差)；direct 模式 critic 输入/构造完全不变 | **actor 侧条件化（参考 chunk 输入 + BC 正则 + chunked 动作）尚未实现**——论文消融中收益最大的组件，接真机前必须补（论文 §VI-C：去 BC 正则掉得最多，去 pass-through 显著变慢）。同源问题即 critic 的 base 上下文，已先修 |
| 4 | RL 算法 **TD3 风格**：双 Q 取 min + 目标网络、无熵正则（固定小 std Gaussian）、actor 每 2 次 critic 更新才 1 次、UTD=5、稀疏人工 ±1 奖励（chunk 内 reward 求和 + γ^C bootstrap） | **SAC**：双 Q 取 min + 目标网络 + 自动温度熵正则、每步更新 actor、UTD=2、dense reward | **算法家族不同（TD3 vs SAC）**。mock 单步 dense 任务下 SAC 是合理默认且已收敛；真机稀疏奖励 + chunk 场景按论文切 TD3 风格 |
| 5 | warmup：VLA 参考策略预填 buffer（N_warm 步） | 均匀随机 300 步（start_steps） | mock VLA 无任务能力；真机接 π0.5 后换 VLA warmup |
| 6 | 人工干预覆盖动作存 buffer、stride-2 子采样、critical-phase 训练/切换 | 无 | 系统级特性，归 infra（rollout runner / 干预接口），真机接入阶段实现 |
| 7 | 真机在线交互（50 Hz，episode 末人工标注成功/失败） | MockReachTask 进程内交互，dense reward | 先验证算法收敛性再上真机 |

## 实现笔记（自研复现中踩过的坑，论文未涉及）

- **联合 backward 导致 SAC 发散**：曾用单一优化器对 actor+critic+token_head 联合 backward，
  actor 梯度流入 critic/token_head，actor 靠搬动 token 表征「虚增 Q」而非改进动作，
  Q 值无锚定指数发散（debug 中 q_mean 0.886→2043）。修复：三段式更新
  （critic+token_head / actor（token detach）/ alpha 三个优化器）。
  论文的设定里 token_head 是冻结的，天然没有这条作弊通道——我们的在线学 token 变体
  必须显式切断（`rlt.py::_sac_step` 第 2 阶段）。
- **tanh log_prob 修正项失效**：a = margin·tanh(u) 时修正项须写
  log(margin·(1−tanh(u)²))；直接写 1−a²（a 已含 margin 缩放）会使修正项退化为
  近似常数，tanh 饱和惩罚消失，同样助长发散。见 `algos/common/nets.py`。
- **residual 模式 critic 缺 base 输入（2026-09-02 质量审查发现）**：residual 动作
  a = clip(VLA_base + 残差)，但回放里 critic 只见 (token, a)。真机 VLA flow-matching
  采样随机：同一 token 下 base 不同 → Q(token, a) 系统性模糊。MockVLA 确定性
  所以 mock 测不出（隐性 bug 的典型来源）。修复：act() 采样 base（预热步也采），
  observe() 缺省透传落回放（direct 模式零占位），critic 输入 concat base
  [token, executed, base]；target 端对 VLA(s') fresh 采样、actor 端取
  clip(base+残差)（与部署一致）。direct 模式构造序列/输入维度完全不变。
  同批顺带：外部批 update() 固定 utd=1（同批重复梯度无采样多样性）。

## 仓库锚点

| 论文组件 | 仓库落点 |
|---|---|
| VLA 内部表征 | `BaseVLA.extract_features()`（models/base.py） |
| RL token 头 | `rlt.py#RLTokenHead` |
| actor-critic | `rlt.py`（SAC） |
| 在线交互循环 | `infra/rollout.py` + `RLT.act()/observe()/update()` |

## 步骤（TODO）

1. [x] MockVLA + MockReachTask 上验证：RL token + SAC 在线学到高成功率（无示教数据，
      实测 3500 步 100%，Q 稳定在真实 return 量级）
2. [x] 精读论文 v2 方法节，校准差异表（2026-09-02，上表 1–7）
3. [ ] 对照 RLinf 实现抓 bug，记录差异清单
4. [ ] 接 π0.5：先补「示教数据预训练 token + 冻结」（差异 1），再补
      「参考动作直通 + BC 正则 + chunked 动作」（差异 3），最后按论文切 TD3 风格（差异 4）
5. [ ] 真机（Unitree 人形）小任务在线微调；系统层特性（干预/子采样/critical-phase）随真机落地（差异 5/6）
