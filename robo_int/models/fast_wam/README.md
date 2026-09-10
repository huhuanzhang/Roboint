# Fast-WAM 接入路线

> 论文：[Fast-WAM: Do World Action Models Need Test-time Future...](https://arxiv.org/abs/2603.16666)
>（[项目页](https://yuantianyuan01.github.io/FastWAM/)）
> WAM（World Action Model）路线：视频扩散/世界模型骨干显式建模未来，再映射动作，
> 作为 VLA 路线的对照组接入。

## 定位

第三基模，**研究对照优先**：同一任务下与 π0.5/GR00T 对比
「显式未来建模」与「隐式动作头」的差异。优先级排在两个 VLA 之后。

## 接入步骤（TODO）

1. [ ] 确认官方代码是否放出（当前未见开源实现，大概率自复现）
2. [ ] 自复现最小版：视频 backbone（冻结）+ 轻量动作头
3. [ ] 封装 `BaseVLA`；`extract_features` 给未来表征（RLT 在 WAM 上的行为
   是个有趣实验：RL token 压缩的是"未来想象"而非当前感知）
4. [ ] 与 TTT 层结合点：视频 backbone 的长序列正适合 TTT/记忆类挂件

## 风险

- 自复现工作量大，先砍到最小可对比版本（不含完整视频生成，只留表征）。
