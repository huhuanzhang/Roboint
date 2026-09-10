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

## mock 级验证结果（2026-09-02，`tests/test_sarm2.py` @slow）

- 阶段估计器：val 精度 95.5%，**held-out 任务 97.5%**（论文真机 85.22%，正迁移主张成立）；
- 值解码器：val MSE 0.0012（常数基线 0.0864 的 1.4%）；预测 steps_to_go 沿 episode 单调下降
  （首帧 0.74 → 末帧 0.09）；无 rewind 窗口 MSE 0.0014（ rewind 非必需但一致）；
- 路由健康度：S_route ≈ 0.0（论文 full model 0.10，同为「均衡」区间，未塌缩）；
- 消融开关均可训练（mock 规模不做与论文 Table 1 的排序断言）；
- Scorer：完整 demo 0.942 vs 半截 demo 0.319（分离明显）。

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
