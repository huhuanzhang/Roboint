# 任务环境路线

任务分两层：**mock 任务**（CPU 可跑，算法验证）→ **真机任务**（Unitree 人形 +
强脑 Revo2 灵巧手）。新算法一律先在 mock 过一遍再接真机（CLAUDE.md 约定）；
真机相关测试标 `@pytest.mark.robot`，不进默认测试。

## 已有

| 任务 | 用途 |
|---|---|
| `mock_reach.py` | 单目标 dense reward，RLT 在线 RL 基准（3500 步 100% 成功率） |
| `mock_pick_place.py` | 多阶段（6 原语 + 进度 GT）、3 任务变体 + held-out，SARM2 阶段/值模型基准与打分验证 |

## 计划任务族（TODO）

### 灵巧手（强脑 Revo2）

- [ ] mock 版：多指状态（6-9 维关节角）+ in-hand 任务（物体翻转 / 旋转 /
      多指抓握），保持 BaseTask 接口与阶段/进度 GT 标注（SARM2 直接可用）；
- [ ] 真机版：`robots/brainco_revo2` 接入 + 遥操示教采集 → `data/` 打分清洗管线；
- [ ] 长时序变体（配 MEM / TTT / in-context 的记忆评测）。

### 人形（Unitree）

- [ ] mock 版：双臂协同任务（一手持物一手操作 / 双臂搬运，双 6-DoF 臂抽象）；
- [ ] 真机版：`robots/unitree` 接入；长时序全身任务优先（记忆类挂件的主战场）。

## 约定

- 任务实现 `BaseTask`（reset/step/observation），观测 dict 含 `state`，
  `StepResult.reward` 可为 None（延后打分，SARM2 用法）；
- 配置走 `configs/task/*.yaml`，factory 注册名字后 Hydra 可组合；
- 脚本专家（示教采集用）实现 PolicyFn：`obs dict -> np.ndarray`，
  GT 标签经专家自身属性暴露（见 `mock_pick_place.MockPickPlaceExpert.phase`）。
