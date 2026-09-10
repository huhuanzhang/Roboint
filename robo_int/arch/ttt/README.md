# TTT Layer（Test-Time Training）

> **主要参考：[RoboTTT: Context Scaling for Robot Policies](https://research.nvidia.com/labs/gear/robottt/)**（NVIDIA Research GEAR Lab）
> 理论源头：Learning to (Learn at Test Time) 系列（TTT-Linear / TTT-MLP，arXiv:2407.04620）

## RoboTTT 做了什么（我们的对齐目标）

- **TTT 层嵌入 GR00T N1.7 级 VLA**——与我们的基模选择直接对口，这是选它当主要参考的决定性原因；
- 视觉-运动上下文扩展到 **8K 时间步**（约 5 分钟"肌肉记忆"），比先前工作高三个数量级；
- 固定尺寸的 fast weights 在每个 visuomotor token 上做自监督梯度更新
  ——隐藏状态即记忆，推理开销恒定；
- 对应 MEM 的显式记忆路线，RoboTTT 是**参数化记忆**路线——两条路线在我们仓库里同框架 A/B。

## 本仓库的实现锚点

| RoboTTT 组件 | 仓库落点 |
|---|---|
| TTT 层（替换 attention 的序列建模层） | 本目录 `ttt_layer.py`（先独立实现 TTT-Linear） |
| 嵌入 VLA 的位置 | `BaseArchModule.attach()`：挂到基模 visuomotor token 流上 |
| 快权重自监督目标 | 视频帧重建/对比目标（按论文细节校准） |
| 对照组 | `arch/mem`（显式记忆）同任务对比 |

## 步骤（TODO）

1. [ ] TTT-Linear 独立实现 + 单元测试（长序列记忆玩具任务：需要回看 N 步前信息的合成任务）
2. [ ] 精读 RoboTTT 论文/开源（若有代码放出），校准：层插入位置、自监督目标、fast weights 尺寸
3. [ ] MockVLA 序列通路挂载（mock 任务加"需要记忆"的变体验证增益）
4. [ ] GR00T N1.7 上复现（同款基模，路线直接可抄）
