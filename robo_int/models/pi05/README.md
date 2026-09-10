# π0.5 接入路线

> 开放权重 VLA（Physical Intelligence）：VLM backbone + flow-matching action expert。
> 仓库定位：与 GR00T N1.7 并行的首选基模之一（用户确认**双基模并行接入**）。

## 接入步骤（TODO）

1. [ ] **加载与推理**：openpi / LeRobot 生态加载 HF 权重，封装成 `BaseVLA`：
   - `forward_actions` = flow-matching 采样（去噪步数可配）
   - `extract_features` = VLM backbone 的 pooled/last-token 表征（RLT 用）
   - `compute_loss` = flow-matching 条件流损失（SFT 用）
2. [ ] **图像管线对齐**：相机帧 -> 模型预处理（分辨率/归一化与预训练一致）
3. [ ] **本体对齐**：动作空间映射到 Unitree 人形 + Revo2 手（delta EE pose + 手指关节）
4. [ ] **LoRA + dropout**：peft 挂 backbone；dropout 从 `infra/config.py` 注入
5. [ ] **数据**：`data/lerobot_io.py` 的双向转换（自采数据 <-> LeRobot 数据集）
6. [ ] **依赖落定**：openpi/lerobot 实际包名写进 `pyproject.toml [pi05]` extra

## 风险

- flow expert 的 log_prob：DSRL 需要，flow-matching 下用概率流 ODE 的
  变换公式近似——实现前先在 MockVLA 上验证数值稳定性。
