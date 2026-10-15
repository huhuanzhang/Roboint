# Robo_int

真机机器人 **VLA / RL 后训练研究基础设施**（对标 [RLinf / RLinf-USER](https://github.com/RLinf/RLinf)），服务于**灵巧手任务**与**人形 VLA** 的算法验证。

核心诉求：**能跑优先** —— 自研算法构思能最快上真机闭环看效果。

## 研究矩阵

| 层 | 内容 |
|---|---|
| 任务 | 灵巧手操作（强脑 Revo2）、人形 VLA（Unitree 人形） |
| 基模 | [π0.5](https://www.physicalintelligence.company/)、[GR00T N1.7](https://github.com/Nvidia/Isaac-GR00T)、[Fast-WAM](https://arxiv.org/html/2603.16666v1)（World Action Model 路线） |
| 架构改进 | [MEM](https://arxiv.org/html/2603.03596v1) 多尺度具身记忆、TTT layer、action-query→vision 注意力、LoRA |
| RL | [RLT](https://arxiv.org/html/2604.23073v2)（RL Token、冻结 VLA 与 actor-critic）、[DSRL](https://arxiv.org/abs/2506.15799)（扩散策略 latent steering，带 MEM）、SARM2（stage-aware 奖励模型） |
| Infra | RLinf 式分布式组件（单机多进程起步）、数采打分/清洗、训练正则化选项（dropout 等） |

## 目录结构

```
configs/          # Hydra 实验配置（robot/task/model/arch/algo/infra/data/eval 八组）
robo_int/
  robots/         # 硬件抽象层：BaseRobot + Unitree 人形 / 强脑 Revo2 / Mock
  tasks/          # 任务环境：BaseTask + 灵巧手/人形任务 + Mock
  models/         # 基模接入：BaseVLA + pi05 / gr00t_n17 / fast_wam / Mock
  arch/           # 架构改进挂件：MEM / TTT / attention / LoRA
  algos/          # RL 算法：SFT(可跑) / RLT / DSRL / SARM2
  data/           # 数采管线：episode 格式、打分、清洗、LeRobot 转换
  infra/          # 训练循环、rollout、配置 schema
  utils/          # 日志等
scripts/          # 入口：teleop_collect / train_sft / train_rl / eval_on_robot / score_and_clean
tests/            # 不依赖真机/GPU 的测试（mock 全链路）
```

## 快速开始（Mock 全链路，无需真机与 GPU）

```bash
git clone https://github.com/huhuanzhang/Roboint.git
cd Roboint

# 在仓库根目录执行，推荐使用 Python 3.10+ 虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

# 1. 用脚本"专家"在 mock 任务上采数据
python scripts/teleop_collect.py data.num_episodes=5

# 2. 打分 + 清洗
python scripts/score_and_clean.py

# 3. SFT 训练（MockVLA，CPU 可跑）
python scripts/train_sft.py

# 4. Mock 评测
python scripts/eval_on_robot.py
```

## 设计原则

1. **算法即插即用**：`algos/` 只依赖统一接口，自研算法修改不碰框架代码。
2. **arch/ 是挂件不是底座**：MEM/TTT/注意力变体都以可挂到任意 `BaseVLA` 的模块实现，方便 A/B。
3. **一切先过 mock**：任何新组件先在 mock 任务上端到端跑通，再上真机。
4. **重依赖惰性导入**：SDK/大模型依赖只在真正用到时 import，保证骨架轻量可测。

## 路线图

- [x] 仓库骨架 + 核心接口 + mock 全链路
- [x] Piper X 双臂真机接入 + π0.5 推理链路（验证范围见历史记录）
- [ ] Unitree 人形 + Revo2 灵巧手 robot client（真机数据闭环）
- [ ] GR00T N1.7 基模接入；π0.5 本地进程内加载（B 类挂件/训练的前提）
- [ ] SFT（LoRA + dropout 选项）+ 真机 eval 闭环
- [x] RLT 功能链路（冻结 VLA + RL token + actor-critic；真机效果待验证）
- [x] SARM2 mock 规模奖励模型与奖励管线（任务数据训练待完成）
- [ ] DSRL（带 MEM）
- [ ] 分布式化（多机并行，接口已预留）

## 当前实现与限制

| 模块 | 当前状态 |
|---|---|
| Mock 数采 → 清洗 → SFT → 评测 | CPU 可运行，默认入口使用 Mock |
| Piper 双臂 + π0.5 server | 已有适配器与 dry-run 入口；需要外部硬件、SDK、客户端和 checkpoint |
| RLT | replay、actor/critic、训练入口与 checkpoint 功能已实现；不代表真机算法效果已验证 |
| SARM2 | mock 规模奖励模型与管线已实现；未训练的输出不可作为有效任务奖励 |
| AQV / LoRA / MEM / TTT | 已有实现与离线单测；真实 π0.5 验证需要兼容的 LeRobot 环境和 GPU |
| Unitree / Revo2 / GR00T / Fast-WAM / DSRL | 接口或设计占位，尚未接入 |

详细的历史实验条件、数值结果与已知边界见 [真机验证记录](docs/VALIDATION_HISTORY.md)。

## 测试与真机配置

```bash
python -m pytest tests/ -q
```

核心测试只需要基础依赖与 CPU。LeRobot、Piper SDK、参考服务、真实 episode 或 GPU 不可用时，相关测试按前置条件跳过；本仓库不分发实验数据和模型权重。
GitHub Actions 会运行 CPU 测试、Mock 脚本全链路和打包检查。

真机相关路径通过环境变量配置，示例见 [.env.example](.env.example)：

```bash
cp .env.example .env
# 编辑 .env，填写本机实际路径；程序不会自动加载该文件。
set -a
source .env
set +a
```

请使用自己的相机序列号修改 `configs/robot/piper_dual_arm.yaml`。Piper 入口默认采用 dry-run。
π0.5 server 使用外部 LeRobot 实现；这些硬件/模型依赖不在基础安装范围内。

GitHub 上传命令与本次准备范围见 [上传说明](docs/GITHUB_UPLOAD.md)。当前项目许可标记为 `Proprietary`。

## 运行环境

- 核心框架与 Mock：Python >=3.10，CPU 可运行。
- 真机部署与真实 VLA 训练：Linux + CUDA，并安装对应硬件/模型依赖。
