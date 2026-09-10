# MEM：多尺度具身记忆挂件

> 论文：[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596)
> （Physical Intelligence，Marcel Torne*、Karl Pertsch* 等，2026-03）
> 项目页：[pi.website/research/memory](https://www.pi.website/research/memory)

## 官方开源状态（2026-09 查证）

**无官方代码仓库。** [Awesome-Memory-for-Robotics](https://github.com/Everloom-129/Awesome-Memory-for-Robotics) 列表中 MEM 只有 [Paper][Project] 标签、没有 [Code]；Physical Intelligence 的 GitHub（openpi 所在）尚未放出。→ **需要自复现**，以论文为准，盯紧官方仓库避免重复劳动。

## 论文架构（复现依据）

- **短时记忆（video-based）**：最近历史以视频形式压缩——video encoder + ViT 上的 temporal attention；
- **长时记忆（text-based）**：跨更长时间尺度的文本化语义记忆；
- 混合模态注入 VLA（论文基于 Gemma 3-4B 级 VLA）；
- 目标任务：10-15 分钟级长时序（如做晚饭中途记住"已经加过盐"）。

同路线参考（检索确认）：MemoryVLA（感知-认知记忆，arXiv:2508.19236）、
HiMe（层级具身记忆）、Native Memory Compression（arXiv:2607.06678）、
FM-VLA（力觉记忆，arXiv:2607.18231）。

## 与其它模块的关系

- **DSRL（带MEM）**：steering 策略以记忆为条件（algos/dsrl/README.md 方案 1）；
- **TTT（RoboTTT）**：显式记忆 vs 参数化记忆的对照组——同一实验框架下 A/B；
- **attention 通用接口**：MEM 把记忆 token 放进 `HookContext.memory`，
  attention 变体经 `key_value_transform` 消费——两挂件即插即组合。

## 实现为 `BaseArchModule`

- `attach(model)`：挂视频编码支路（短时）+ 文本记忆库（长时），
  记忆 token 注入动作头上下文；
- `detach()`：完整撤销（A/B 实验正确性依赖）。

## 步骤（TODO）

1. [ ] MEM-short 最小复现：视频短时记忆支路（MockVLA 上验证）
2. [ ] 文本长时记忆库：写入/检索/遗忘策略
3. [ ] 长时序 mock 任务（"需要记住 N 步前事件"的合成任务）上对照增益
4. [ ] π0.5 / GR00T 上挂载验证；与 TTT 挂件同任务对比
