# Robo_int

真机机器人 **VLA / RL 后训练研究基础设施**（对标 [RLinf / RLinf-USER](https://github.com/RLinf/RLinf)），服务于**灵巧手任务**与**人形 VLA** 的算法验证。

核心诉求：**能跑优先** —— 自研算法构思能最快上真机闭环看效果。

## 研究矩阵

| 层 | 内容 |
|---|---|
| 任务 | 灵巧手操作（强脑 Revo2）、人形 VLA（Unitree 人形） |
| 基模 | [π0.5](https://www.physicalintelligence.company/)、[GR00T N1.7](https://github.com/Nvidia/Isaac-GR00T)、[Fast-WAM](https://arxiv.org/html/2603.16666v1)（World Action Model 路线） |
| 架构改进 | [MEM](https://arxiv.org/html/2603.03596v1) 多尺度具身记忆、TTT layer、action-query→vision 注意力、LoRA |
| RL | [RLT](https://arxiv.org/html/2604.23073v2)（RL Token，干净复现，RLinf 实现有 bug）、[DSRL](https://arxiv.org/abs/2506.15799)（扩散策略 latent steering，带 MEM）、SARM2（stage-aware 奖励模型） |
| Infra | RLinf 式分布式组件（单机多进程起步）、数采打分/清洗、训练正则化选项（dropout 等） |

## 目录结构

```
configs/          # Hydra 实验配置（robot/task/model/algo/infra 五组）
robo_int/
  robots/         # 硬件抽象层：BaseRobot + Unitree 人形 / 强脑 Revo2 / Mock
  tasks/          # 任务环境：BaseTask + 灵巧手/人形任务 + Mock
  models/         # 基模接入：BaseVLA + pi05 / gr00t_n17 / fast_wam / Mock
  arch/           # 架术改进挂件：MEM / TTT / attention / LoRA
  algos/          # RL 算法：SFT(可跑) / RLT / DSRL / SARM2
  data/           # 数采管线：episode 格式、打分、清洗、LeRobot 转换
  infra/          # 训练循环、rollout、配置 schema
  utils/          # 日志等
scripts/          # 入口：teleop_collect / train_sft / train_rl / eval_on_robot / score_and_clean
tests/            # 不依赖真机/GPU 的测试（mock 全链路）
```

## 快速开始（Mock 全链路，无需真机与 GPU）

```bash
pip install -e ".[dev]"

# 1. 用脚本"专家"在 mock 任务上采数据
python scripts/teleop_collect.py data.num_episodes=5

# 2. 打分 + 清洗
python scripts/score_and_clean.py

# 3. SFT 训练（MockVLA，CPU 可跑）
python scripts/train_sft.py

# 4. 真机(现在是 mock)评测
python scripts/eval_on_robot.py
```

## 设计原则

1. **算法即插即用**：`algos/` 只依赖统一接口，自研算法修改不碰框架代码。
2. **arch/ 是挂件不是底座**：MEM/TTT/注意力变体都以可挂到任意 `BaseVLA` 的模块实现，方便 A/B。
3. **一切先过 mock**：任何新组件先在 mock 任务上端到端跑通，再上真机。
4. **重依赖惰性导入**：SDK/大模型依赖只在真正用到时 import，保证骨架轻量可测。

## 路线图

- [x] 仓库骨架 + 核心接口 + mock 全链路
- [ ] Unitree 人形 + Revo2 灵巧手 robot client（真机数据闭环）
- [ ] π0.5 与 GR00T N1.7 双基模加载/推理
- [ ] SFT（LoRA + dropout 选项）+ 真机 eval 闭环
- [ ] RLT 复现（冻结 VLA + RL token + actor-critic）
- [ ] DSRL（带 MEM）、SARM2 奖励模型
- [ ] 分布式化（多机并行，接口已预留）

## 运行环境

- 开发：Windows（仅编写代码）
- 运行：Linux + CUDA（训练与真机部署）
