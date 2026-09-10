# 轨迹 Prompt（用户构思方向）

> **用户自研构思**：用**轨迹段**（而非语言/图像）作为 prompt 条件策略——
> 「照着这条（成功/部分成功）轨迹的样式做」。暂无定名对口论文，候选参考：
> - [PromptDT: Prompting Decision Transformer for Few-Shot Policy Generalization](https://arxiv.org/abs/2206.13499)（ICML 2022，
>   **trajectory prompt 概念出处**：few-shot 示教轨迹段编码任务信息引导策略，
>   [官方代码](https://github.com/mxu34/prompt-dt)）；
> - [Prompt Tuning with Diffusion for Few-Shot Pre-trained Policy Learning](https://arxiv.org/abs/2411.01168)
>   （扩散**生成** prompt 轨迹加速 few-shot 适应）；
> - RICL（`arch/in_context/`）：检索示教片段进上下文，思路相邻（检索 vs 生成）。

## 与其它挂件的关系

- **DSRL**：steering 作用在初始噪声空间，轨迹 prompt 作用在条件空间——可组合
  （轨迹 prompt 提供任务条件，DSRL 学残差噪声偏置）；
- **in-context**：in-context 是「拼接进输入序列」，轨迹 prompt 是「编码成
  条件向量/token 注入」（可学习、可压缩），两者同任务 A/B；
- **MEM**：长时语义库里的历史成功轨迹天然是 prompt 候选来源（检索→编码→注入）；
- **SARM2**：值解码器可给候选轨迹段打分（预期进度收益），作为 prompt 选择器。

## 步骤（TODO）

1. [ ] 定义轨迹 prompt 编码（轨迹段 → 固定维条件向量/token，编码器可学习）
2. [ ] MockVLA 条件通路 + mock 任务 A/B（同任务：有/无 prompt、好/坏 prompt）
3. [ ] 与 DSRL / in-context 的组合实验（灵巧手长时序任务优先，配 Revo2）
