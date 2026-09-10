# CLAUDE.md

真机机器人 VLA/RL 后训练研究仓库。对标 RLinf。核心诉求"能跑优先"：让用户自研算法构思最快上真机（Unitree 人形 + 强脑 Revo2 灵巧手）闭环验证。

## 本地开发环境（Windows）

- 测试/运行 Python：`D:\conda_envs\ai\python.exe`（3.11，torch 2.12 CPU，numpy）
- 跑测试：`D:/conda_envs/ai/python.exe -m pytest tests/ -q`
- 冒烟训练：`D:/conda_envs/ai/python.exe scripts/train_sft.py`（默认全 mock，CPU 可跑）
- 目标运行环境是 Linux + CUDA；真机 SDK、pi05/gr00t 等重依赖都是惰性导入，禁止在核心接口文件顶部 import 它们

## 约定

- 注释/docstring 用中文；数据平面用 numpy，模型平面用 torch
- 新组件必须先在 mock 链路（mock robot/task/VLA）跑通再接真机
- 算法（algos/）、基模（models/）、架构挂件（arch/）只依赖 `base.py` 里的抽象接口，互相不直接 import
- 配置统一走 `configs/`（Hydra + OmegaConf structured schema 在 `robo_int/infra/config.py`）
- 训练正则化选项（dropout、weight_decay、grad_clip 等）是 infra 一等公民，加在 `infra/config.py` 的 schema 里而不是散落各算法
- 测试不依赖真机/GPU；需要的话用 `@pytest.mark.robot` / `@pytest.mark.gpu` 标记

## 仓库地图

- `robo_int/robots/` 硬件抽象（BaseRobot/CompositeRobot + unitree/ brainco_revo2/ stub）
- `robo_int/tasks/` 任务环境（BaseTask + mock_reach / mock_pick_place 可跑示例；任务族路线见 tasks/README.md）
- `robo_int/models/` 基模接入（BaseVLA + pi05/ gr00t_n17/ fast_wam/ 接入路线 README）
- `robo_int/arch/` 架构改进挂件（attention 接口已实现、LoRA 工具就位；MEM/TTT/in-context/轨迹prompt 为 stub + 论文链接）
- `robo_int/algos/` SFT 已实现；rlt/ dsrl/ sarm2/ 是设计笔记 + stub
- `robo_int/data/` episode 格式、打分(scoring)、清洗(cleaning)、LeRobot 转换
- `robo_int/infra/` TrainLoop、rollout runner、配置 schema
