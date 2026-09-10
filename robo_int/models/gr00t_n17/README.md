# GR00T N1.7 接入路线

> NVIDIA Isaac GR00T N1.7（[GitHub](https://github.com/Nvidia/Isaac-GR00T)）：
> 开放人形 VLA，双系统架构（System-1 快动作 + System-2 慢推理），跨本体设计，
> 与本仓库「人形 VLA」任务天然对口。

## 接入步骤（TODO）

1. [ ] **确认访问与依赖链**：early access 权重 + HF 模型卡；
   依赖（flash-attn 等 Linux-only）全部放 `[gr00t]` extra，不进主依赖
2. [ ] **加载与推理**：官方微调脚本最小跑通 -> 封装 `BaseVLA`
   - `forward_actions`：action head（离散 token 或 DiT，按 N1.7 实际结构）
   - `extract_features`：System-2 表征（RLT 状态）
   - `compute_loss`：动作头损失（SFT）
3. [ ] **本体适配**：GR00T 官方 embodiment 配置加入 Unitree 人形 + Revo2 手
   （跨本体设计的预期用法，重点是 action normalizer 与关节映射表）
4. [ ] **LoRA/dropout**：同 pi05，走统一 infra 选项
5. [ ] 与 π0.5 做 A/B（同任务、同数据、同训练配置）

## 风险

- early access 的依赖/许可变动：接入时再验证，隔离在 `[gr00t]` extra 后面；
- 若 N1.7 动作头为离散 token，RLT/DSRL 的 log_prob 路线与 π0.5 不同，
  算法侧做好双分支。
