# Action-Query → Vision 注意力变体（通用接口 + 四个参考工作）

> 用户点名参考：VLA-Trace / AtVLA / GazeVLA / PelicanVLA（均已检索查证，见下表）
> 通用接口实现在本目录 `base.py`（`BaseAttentionVariant` + `HookContext`）。

## 参考工作

| 工作 | 出处 | 借鉴点 | 在接口里的对应 |
|---|---|---|---|
| [AtVLA: Look Where It Matters](https://arxiv.org/html/2608.02197v1) | arXiv:2608.02197 | 可学习 register token 注入视觉编码器 + 注意力矫正 + 不确定性门控局部精化 + 注意力引导裁剪 | `key_value_transform`（register/crop）、`post_attention`（uncertainty gating） |
| [GazeVLA](https://arxiv.org/abs/2604.22615) | arXiv:2604.22615 | 大规模第一人称人类数据预训练 gaze/intention；衍生 [Gaze2Act](https://arxiv.org/html/2605.30282v1) 的 gaze-regularized attention | `query_transform`（gaze embedding 进 query）、`attention_bias`（gaze 先验偏置） |
| [Pelican-VLA 0.5](https://arxiv.org/html/2607.06655v1) | arXiv:2607.06655 | "Attending before acting"：统一理解 + 未来帧生成 + 动作预测，动作前先显式注意/生成 | `pre_attention`（动作前生成/注意阶段）、`key_value_transform` |
| [VLA-Trace](https://arxiv.org/abs/2605.30117) | arXiv:2605.30117 | **诊断框架而非机制**：CKA 表征追踪 + attention knockout（因果干预）+ 行为测试，已在 π0 上应用 | 不进挂件接口，做成本目录 `diagnostics.py` 的分析工具（变体验证/归因） |

相关：[Inference-Time Attention Steering for VLAs](https://arxiv.org/pdf/2608.17095)（推理时注意力转向，建立在 VLA-Trace 的 knockout 上）。
同名注意：**AT-VLA**（触觉注入，arXiv:2605.07308，北大）是另一篇——若灵巧手要触觉模态再纳入。

## 通用接口（base.py）

四个钩子覆盖四篇工作的机制落点，挂件实现任意子集：

```python
class BaseAttentionVariant(BaseArchModule):
    def pre_attention(self, context) -> None        # 动作前阶段（Pelican 式生成/注意）
    def query_transform(self, queries, context)     # action query 变换（GazeVLA 式）
    def key_value_transform(self, keys, values, context)  # 视觉 KV 变换（AtVLA 式 register/crop）
    def attention_bias(self, bias, context)         # logits 偏置（gaze 先验/矫正 mask）
    def post_attention(self, output, context)       # 输出后处理（不确定性门控精化）
```

`HookContext` 统一携带 observation / instruction / memory（MEM 挂件写入记忆 token，
attention 变体消费——**MEM × attention 即插即组合**）/ uncertainty 等。
基模侧约定：wrapper 暴露 action↔vision 交叉注意力的接入点（π0.5/GR00T 接入时落实
`get_attention_hook_points()`）。

## 实验矩阵（TODO）

全部走同一评测协议（同数据同预算 + VLA-Trace 式诊断归因）：

- [ ] 基线：均匀 self-attention
- [ ] v1 gaze-bias：`attention_bias` 注入 gaze 先验（GazeVLA 路线）
- [ ] v2 register+crop：`key_value_transform`（AtVLA 路线）
- [ ] v3 attend-before-act：`pre_attention` 未来生成（Pelican 路线）
- [ ] v4 memory-query：query 同时消费 `context.memory`（与 MEM 联动，本仓库原创组合）
- [ ] 诊断：`diagnostics.py` 的 attention knockout 定位各变体真正用到的视觉区域
- [ ] 度量：SFT 收敛速度 / 真机成功率 / 注意力图可视化 / 推理延迟开销

## 步骤（TODO）

1. [x] 通用接口 `base.py`（钩子 + HookContext + 串行编排器）
2. [ ] diagnostics.py：VLA-Trace 式 CKA + attention knockout
3. [ ] MockVLA 加视觉输入后实现 v1/v2 两个变体打样
4. [ ] π0.5 / GR00T 接入点打通后全矩阵跑
