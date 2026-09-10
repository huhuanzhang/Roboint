# DSRL（带 MEM）

> 论文：[Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799)（DSRL）
> 用户构思：**DSRL × MEM 结合**——在 latent-noise steering 基础上引入
> [MEM 多尺度具身记忆](https://arxiv.org/abs/2603.03596)，面向长时序灵巧手任务。

## DSRL 核心思想

- 底座是 BC 训练的扩散策略（diffusion policy），**冻结不动的黑盒**；
- RL 不改扩散网络本身，而是学习在初始噪声（latent-noise）空间上做 steering——
  即学一个从观测到初始噪声偏置的策略；
- 样本高效、真机友好，论文在 Adroit 灵巧手任务上验证。

## 与 MEM 的结合点（待细化）

候选方案（实现前先做 A/B）：
1. **记忆条件化 steering**：steering 策略的输入从当前观测扩展为「观测 + MEM 记忆 token」，
   长时序任务中提供历史上下文；
2. **记忆检索式扩散条件**：MEM 的长期语义库检索结果作为扩散条件的额外分支；
3. TTT 隐藏状态作为在线记忆（与 arch/ttt 组合）。

## 依赖关系

- 需要一个扩散/流策略底座：π0.5 的 flow-matching action expert 天然适配；
- MEM 模块来自 `arch/mem`；
- SAC 组件可与 RLT 共用（抽到 algos/common/）。

## 步骤（TODO）

1. [ ] MockVLA 加噪声多步采样头，模拟"扩散策略"（最小可验证 DSRL 设定）
2. [ ] DSRL 主体：latent steering 策略 + SAC
3. [ ] 接 π0.5 flow expert + MEM 挂件
4. [ ] Revo2 灵巧手任务真机验证
