# In-Context（示教作为上下文）

> 两条路线，**落地成本不同，分开对齐**（2026-09-03 审查修正：曾把 RICL 误记为
> 免训练路线——其摘要明确是 post-hoc finetuning 注入 ICL 能力，免训练的只是
> 部署侧「新任务不再更新参数」的接口）：
>
> - **免训练线（mock 第一步对齐）**：[Keypoint Action Tokens Enable In-Context
>   Imitation Learning in Robotics](https://roboticsconference.org/2024/program/papers/96/)
>   （RSS 2024，免训练 in-context 模仿）、[Retrieval-VLA: Training-Free In-Context
>   Adaptation for Vision-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_Retrieval-VLA_Training-Free_In-Context_Adaptation_for_Vision-Language-Action_Models_CVPRF_2026_paper.pdf)
>   （CVPR 2026 Findings，training-free 检索拼接）
> - **post-train 线（接 π0.5 阶段对齐）**：[RICL: Adding In-Context Adaptability to
>   Pre-Trained Vision-Language-Action Models](https://arxiv.org/abs/2508.02062)
>   （UPenn，Sridhar et al.；对 π₀-FAST 用小规模示教数据事后微调注入 ICL 能力，
>   强调 "without training from scratch" 而非免训练；部署时新任务 10–20 条示教
>   拼上下文即无参数更新。**已放出代码与 RICL-π₀-FAST 权重**）

## 思路（共同机制 + 两条线）

- 共同机制：把「相关示教片段」检索进 VLA 的输入上下文，靠 in-context 能力
  适配新任务/新环境；
- **免训练线**：预训练 VLA 直接拼上下文用（Keypoint Action Tokens / Retrieval-VLA）。
  mock 第一步只做「检索 + 拼接」，零训练预算——但真机大模型上 ICL 能力是否
  足够是经验问题；
- **post-train 线（RICL）**：模仿学习预训练的 VLA **不会自然获得 ICL 能力**
  （RICL 摘要明确指出），需要先用交错示教数据事后微调。接 π0.5 时若免训练
  路线效果不足，走这条——**需准备示教数据 + 训练预算**，接入阶段再对齐
  （RICL 有官方代码可抄 recipe）。
- 与 MEM 的分工：MEM 是**参数化/记忆库式**历史（写入-读出，隐藏态或
  检索库），in-context 是**显式拼上下文**；检索器可与 MEM 长时语义库
  共用——两条路线同任务 A/B（与 TTT 的显式/参数化对比同一框架）。

## 本仓库的实现锚点

| 组件 | 仓库落点 |
|---|---|
| 检索器（示教库 → 相关片段） | 本目录（与 arch/mem 长时库检索共用接口） |
| 上下文注入 | `BaseAttentionVariant.key_value_transform`（片段作额外 KV）或 `query_transform`；跨挂件经 `HookContext` |
| 示教库 | `data/`（episode + LeRobot 格式，复用打分/清洗管线，Sarm2Scorer 可过滤低质片段） |
| post-train 数据（RICL 线） | 交错示教格式（观测-动作片段交织），接 π0.5 时定义；复用 SFT 训练循环 |
| 对照组 | `arch/mem`（参数化记忆）、`arch/ttt` 同任务对比 |

## 步骤（TODO，分路线）

**免训练线（先行，零训练预算）**：
1. [ ] MockVLA 加「上下文条件」通路（观测序列前拼示教片段特征）
2. [ ] 检索器 mock 版（状态距离 / 特征相似度）
3. [ ] mock 任务「few-shot 新任务适配」评测（复用 mock_pick_place held-out 变体 2）

**π0.5 阶段（分路线决策）**：
4. [ ] 先按 Keypoint Action Tokens / Retrieval-VLA 免训练注入（检索粒度 / 截断长度 / 上下文位置）
5. [ ] 若 π0.5 上下文能力不足，按 RICL 准备交错示教数据 post-train（训练预算另计，官方 recipe 可参考）
