# SARM2（多任务阶段感知奖励模型）复现

> 论文：[SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic Manipulation](https://arxiv.org/abs/2606.10305)
>（Q. Chen¹, H. Zheng¹, J. Yu², S. Huang¹, J. Sun¹, K. Goldberg², C. Wen³, P. Abbeel²,
> Y. Shentu²⁴, P. Wu⁴；¹Stanford ²Berkeley ³SJTU ⁴xdof.ai，2026）
> 项目页：[qianzhong-chen.github.io/sarm2.github.io](https://qianzhong-chen.github.io/sarm2.github.io/)

## 代码现状（2026-09-02 查证，论文全文 + 官方代码精读）

- **奖励模型本体官方已开源**：[xdofai/opensarm](https://github.com/xdofai/opensarm)
  （`models/action_estimator.py` 阶段估计器、`models/moe_deco_gate_reward_model.py` 多门 MoE 值模型、
  `config/sarm2.yaml`、`workspace/sarm2_ws.py` 训练循环；基于 LeRobot 数据集 + 冻结 SigLIP-2）；
- **SPIRAL 半边**（残差 RL，论文 §3.2）：[Qianzhong-Chen/openspiral](https://github.com/Qianzhong-Chen/openspiral)
  （openpi/JAX，依赖 π0/π0.5 + GPU，真机接入阶段再参考）；
- → **本目录以官方 opensarm 为准的 mock 级移植**（CPU 可跑）：结构/损失/训练流程对齐官方，
  视觉前端与规模 mock 化，全部差异见下表。

## 在本仓库的角色

SARM2 是**奖励基础设施**，不是策略优化器——同时服务两条线：
1. **数采打分**：`Sarm2Scorer` 实现 `data/scoring.py#BaseDataScorer`，给清洗管线提供学习型质量分（已落地）；
2. **RL 奖励**：给 RLT/DSRL 提供稠密/阶段化奖励（`StepResult.reward=None` 的延后打分闭环，TODO 7）。

## 核心思想（按论文 §3.1 + 官方代码）

1. **阶段估计器**（§3.1.1）：对最近 N=6 帧（间隔 Δ=30，~6s@30Hz）做时序分类，
   预测当前动作基元（21 原语 + null，K+1=22 类）；3 相机冻结 SigLIP-2 嵌入 + 本体感觉，
   4 层因果 Transformer + 双头（原语分类 + 语义组分类）。多任务混训（100 任务 200h 语料均衡采样 66h），
   实现 85.22% micro 精度并迁移到新任务。
2. **多门 MoE 值解码器**（§3.1.2）：6 层因果 Transformer，输入 = 首帧视觉锚 + 6 近帧 +
   最多 3 帧「回卷」（ReWiND 式隐式负样本，阻止退化为绝对时间索引预测器）+ e_task（任务指令嵌入）+
   e_prim（预测原语的嵌入）。K+1 原语按语义归成 M+1=8 组，**每组一个门网络**在共享专家池
   （E=10，各 3 隐层 MLP，专家即输出头）上做 top-k=2 稀疏路由；sigmoid 输出每步
   steps_to_go（=1−progress），MSE 稠密逐步回归 + MoE 辅助损失。
3. **SPIRAL**（§3.2，本仓库未实现）：DICE-RL 残差策略 + 混合 TD/MC critic（α=0.5）、
   TD3+BC（β=30）、best-of-κ（κ=4）部署、SARM2 奖励自标注的自治飞轮。
   80% value-MSE 降幅；叠短裤 58%→100%、擦白板 50%→90%。

## 与论文/官方代码的差异（复现标注）

**A. vs 官方 opensarm（实现级差异——官方为准，mock 化必要性标注）**

| # | 官方 opensarm | 本实现（mock 级） | 原因/计划 |
|---|---|---|---|
| 1 | 3 相机帧 → 冻结 SigLIP-2 嵌入；e_task/e_prim = SigLIP 文本嵌入（任务指令/原语名，soft 混合） | 原始状态向量（mock_pick_place 6 维）直接当「冻结视觉嵌入」；e_task/e_prim = 可学习 `nn.Embedding`，soft 混合路径同款（`compose_prim_embedding`） | mock 无视觉；接真机时换 SigLIP-2 前端 + 文本嵌入，接口不变（`state_seq` 换嵌入序列） |
| 2 | 每时间步 N+3 路 token（相机×N + e_prim + e_task + proprio），逐帧 concat 融合到 (N+3)·d | 每时间步 3 路 token（state + e_prim + e_task），同款 concat 到 3d | 流数随输入源缩放，融合机制一致 |
| 3 | d=512；阶段估计器 4 层 / 值模型 6 层；专家 3×256；30k 步（值）/16k 步（阶段预训练）；lr 5e-5 | d=64；各 2 层；专家 3×64；~1k 步；lr 5e-4（步数缩小 30× 故放大 lr） | CPU 可跑；超参比例保持（E=10、top-2、λ 同官方） |
| 4 | 辅助损失 `λ_bal·var(load)+λ_imp·var(importance)−λ_ent·H`，总损失 `mse+0.2·aux`（α_auxiliary） | **同款逐项实现**（含官方有而论文未提的 λ_imp=0.1 与 α_aux=0.2） | 对齐官方 |
| 5 | 阶段估计器先独立预训练（act_pri_ws）→ 值训练时联合微调（finetune_act_pri=True，CE 权重 0.25） | 同款两阶段（`phase="stage"` → `phase="value"` + `finetune_stage`），联合微调 CE 权重 0.25 | 对齐官方 |
| 6 | top-k 分数**重归一化**后加权；稀疏执行（只跑被选中的专家） | 重归一化同款；稠密计算 + gather（数学等价，mock 规模无性能顾虑） | 等价实现 |
| 7 | 数据集：LeRobot parquet + progress 列（标注管线 label_data/）；rewind 以 0.8 概率采样并倒序拼接 | `collect_demo`（脚本专家自带 GT 原语/进度）+ `build_stage/value_windows` 同款布局 `[anchor, mid×N, rewind 倒序, pad]`，rew_prob=0.8 | mock 数据平面；真机换 LeRobot 读取器时窗口逻辑复用 |
| 8 | 训练循环/日志/wandb/checkpoint（workspace 层） | `update(batch)`（BaseAlgo）+ 调度器 warmup+cosine、grad_clip 1.0 同款 | 仓库 infra 约定 |

**B. vs 论文（论文与官方代码不一致处，以官方为准并标注）**

| # | 论文（arXiv:2606.10305） | 官方 opensarm 实际 | 本实现 |
|---|---|---|---|
| 1 | 公式 (7) `L_entropy = −Σ p̄ log p̄` 印作**正项加入**总损失——最小化它会*降低*熵，与论文自述「防路由过早尖锐化」矛盾（**符号疑似笔误**） | `aux = … − λ_ent·H`（负号=最大化熵），印证意图 | 按官方：`−λ_ent·ent` |
| 2 | 公式 (6) `L_balance = E·Σ f_e·p̄_e`（Switch 风格 importance×load） | `var(expert_load)`（负载方差）+ 独立 `λ_imp·var(importance)`，λ_imp=0.1 论文未提 | 按官方 |
| 3 | 两模块「分别训练」（separately trained） | 阶段估计器预训练后**联合微调**（finetune_act_pri=True + CE 0.25） | 按官方 |
| 4 | 回归目标 r* = −(T−t)/T ∈ [−1,0]（π0.6 风格） | target = steps_to_go = 1−progress ∈ [0,1]，sigmoid 输出（仿射等价） | 按官方 [0,1] |
| 5 | 训练 10k 步（Table 3） | 配置 30k（值）/16k（阶段） | mock ~1k 步 |
| 6 | 辅助损失权重直接进总损失 | 多一层全局缩放 α_auxiliary=0.2 | 按官方 |

**C. mock 替身与论文实验设置的对应**

| 论文实验 | mock 对应 |
|---|---|
| 100 任务 200h 语料、66h 均衡采样 | `mock_pick_place` 3 任务变体（0/1 训练，2 held-out）× 各 8-12 条示教 |
| 21 原语 + null（K+1=22）、8 语义组（M+1） | 6 原语 + null、4 语义组（`PRIMITIVES`/`SEMANTIC_GROUPS`） |
| 10 任务基准（S1 经典/S2 非常规）+ S1-Var 迁移 | held-out 任务 2 的阶段估计迁移测试（实测 95%+） |
| MoE Density S_route：full 0.10 / w/o MG 0.42 / w/o SE 0.87 | `collapse_score`（公式同官方 moe_utils）；训练后 ≈0（均衡） |
| 消融 w/o SE / w/o MG | `use_primitives=False` / `single_gate=True` 开关 |
| 数采打分（episode 质量） | `Sarm2Scorer`（BaseDataScorer 接口，接入 data.cleaning） |
| SPIRAL RL 消费 dense reward | 未实现（接 π0.5 后参考 openspiral，见 TODO） |

## mock 级验证结果（2026-09-02 因果 mask 修复后重测，`tests/test_sarm2.py` @slow）

> 注：此前版本的值解码器存在 token 布局与 causal mask 错位导致的未来帧全量泄漏
> （见实现笔记 #7），旧数字（MSE 0.0012 / scorer 0.942 等）为污染值，已作废。

- 阶段估计器：val 精度 98.4%，**held-out 任务 94.1%**（论文真机 85.22%，正迁移主张成立）；
- 值解码器：val MSE 0.0017 = 常数基线 0.0904 的 1.9%（无 rewind 评测窗口）；
  预测 steps_to_go 沿 episode 单调下降（首帧 0.55 → 末帧 0.04，平均步差 −0.026）；
- 路由健康度：S_route 0.001（论文 full model 0.10，同为「均衡」区间，未塌缩）；
- 消融开关均可训练（mock 规模不做与论文 Table 1 的排序断言）；
- Scorer：完整 demo 0.962 vs 半截 demo 0.453（分离明显）；
- 因果性回归测试：帧 i 输出对帧 j>i 输入的梯度严格为零、同帧流间不屏蔽
  （`test_causal_no_future_leakage` / `test_same_frame_cross_stream_not_masked`，
  本次泄漏漏网的直接防线）。

## 实现笔记（mock 复现中踩过的坑，论文/官方未涉及）

官方代码只覆盖真机数据管线，以下 bug 全部出在 mock 替身与接线层——每一个都直接
污染训练真值或破坏推断一致性，记录于此防复发：

- **进度真值不单调**：进度按「到阶段目标的距离比例」插值，专家动作噪声会让距离
  单步回退 → 训练 target 倒挂。修复：阶段内改用最近距离包络（best-so-far）插值，
  物理语义即「进度只进不退」（`mock_pick_place.py#_progress`）。
- **grasp 阶段卡死**：初版专家在 grasp/release 只控夹爪不保位置，噪声把手推离抓取
  半径（0.06）后 `held` 判定迟迟不触发——grasp 从 ~5 步膨胀到 ~27 步/episode
  （手在物体旁随机游走等重逢）。修复：抓/放阶段对物体/放置点做位置保持。
- **transport 中途掉物**：抓取后夹爪停在 0.5~0.6 的临界开度，专家噪声加在 d_grip
  上使其随机游走跌破释放阈值 0.5 → 中途掉物 → 卡死 transport 直到截断
  （task 0 一度 4/8 成功、transport 均长 42 步）。修复：transport 持续闭紧，
  把夹爪钉在 1.0（`mock_pick_place.py#MockPickPlaceExpert.act`）。
- **PolicyFn 签名不符**：专家 act() 初版返回 `(action, primitive)` 元组，而
  factory/rollout 要求 `PolicyFn = dict -> np.ndarray`。修复：专家自带状态机
  （转移条件全部由观测量推断，与环境侧阶段机同构），act 只返回动作，标签改由
  `expert.phase` 暴露；episode 边界靠 `obs["steps"]==0` 自动复位。
- **YAML 裸 `null`**：原语表首元素 `null` 被 YAML 1.1 解析成 None，
  `build_hierarchy` 里 `None.split` 直接 AttributeError。修复：词汇表字符串
  一律引号（`configs/algo/sarm2.yaml` 注释已标注此坑）。
- **predict_value 窗口语义错位**：推断初版取序列前 N+1 帧当 `[anchor+mid]`，与
  训练窗口「anchor=首帧 + 以 t 结尾、间隔 frame_gap 的 mid」布局不一致，长 episode
  上属分布外推断。修复：逐帧滑窗重构同款布局（`sarm2.py#predict_value`）。
- **causal mask 与 token 布局错位（未来帧全量泄漏，2026-09-02 质量审查发现）**：
  forward 里 `torch.stack([s,p,t], dim=1)` 后直接 reshape 得到 **stream-major**
  布局 `[s_0..s_{T-1}, p_0.., t_0..]`，而 causal mask 的 `repeat_interleave` 构造
  是 **time-major** 语义（假设 `[s0,p0,t0, s1,p1,t1, ...]` 按帧分组）。错位下
  每帧融合输出都能经 task 流 token 的注意力段看到**全部未来帧**（T=10 实测帧 0
  可见 9 个未来帧）——非最后位置的 steps_to_go 回归可从未来帧直接读答案，
  训练信号污染（旧 MSE 0.0012 是污染数字），与官方 opensarm 因果结构不符。
  docstring 写的意图（每时间步 3 路 token、流间不屏蔽）本来就是 time-major，
  实现写错了。修复四件套：permute 到 time-major + first_pos 注入位改 flat 0 +
  padding mask 改 `repeat_interleave` + 融合取回改 `view(B,T,3,-1)`；
  并新增梯度因果性测试（帧 i 输出对帧 j>i 输入梯度为零）——此前测试只查
  值域/形状/可训练性，因果性漏网，这是本 bug 存活整整一轮复现的原因。

## 仓库锚点

| 论文/官方组件 | 仓库落点 |
|---|---|
| ActionTransformer（阶段估计器） | `sarm2.py#StageEstimator` |
| RewardTransformer（多门 MoE 值模型） | `sarm2.py#MMoEValueDecoder` |
| 两阶段训练（act_pri_ws / sarm2_ws） | `sarm2.py#SARM2`（`phase` 分发） |
| FrameGapLeRobotDataset 窗口布局 | `sarm2.py#build_stage_windows / build_value_windows` |
| moe_utils 塌缩度 | `sarm2.py#collapse_score` |
| 数采打分 | `scorer.py#Sarm2Scorer`（`data/scoring.py` 接口） |
| mock 基准任务 | `tasks/mock_pick_place.py` |
| 配置 | `configs/algo/sarm2.yaml` + `configs/task/mock_pick_place.yaml` |

## 步骤（TODO）

1. [x] 查证官方代码（opensarm=奖励模型半边开源、openspiral=RL 半边开源）（2026-09-02）
2. [x] mock 级移植：阶段估计器 + 多门 MoE 值模型 + 辅助损失 + S_route + 消融开关（对齐官方实现）（2026-09-02）
3. [x] mock_pick_place 多阶段基准 + held-out 迁移验证 + Sarm2Scorer 接入打分管线（2026-09-02）
4. [ ] 接真机视觉前端：SigLIP-2 冻结嵌入替换原始状态（差异 A1/A2），LeRobot 数据读取器替换窗口构建（A7）
5. [ ] RM 自适应（论文 4 阶段管线的第 2 步：~100 条 rollout 人工标注 {fast/slow progress, adjust, mistake} 混合预训练子集微调）
6. [ ] SPIRAL 残差 RL（参考 openspiral；依赖 π0.5 基模接入，TD/MC 混合 critic + TD3+BC + best-of-κ）
7. [ ] RL 奖励消费侧：RLT/DSRL 的 dense reward 供给接口（StepResult.reward=None 的延后打分闭环）
