# Unitree 人形接入

> 硬件：Unitree 人形（G1 系列为主，具体型号接入时确认）。
> SDK：unitree_sdk2（DDS 通信）的 Python 绑定，惰性导入。

## 接入步骤（TODO）

1. [ ] 确认型号与 SDK 版本；`[unitree]` extra 落定包名
2. [ ] `UnitreeHumanoid(BaseRobot)`：
   - connect/disconnect：DDS channel 初始化
   - get_state：关节状态 + IMU（低状态接口）
   - send_action：关节位置目标（含安全限幅/限速，RL 探索期间的保护带）
   - 使能/去使能/阻尼模式切换
3. [ ] 上位控制模式选型：整机关节空间 vs 手臂 delta EE（阻抗/导纳在上位机做，
   参考RLinf-USER 的 10Hz 目标 + 底层 500Hz-1kHz 追踪分层）
4. [ ] 与 Revo2 组成 `CompositeRobot`（人形手臂 + 灵巧手 = 灵巧手任务本体）
5. [ ] teleop 数据采集：space mouse / 键盘 / 主从臂，产出 EpisodeStore

## 安全红线（实现 review 时逐条核对）

- 动作限幅/限速在 send_action 内部强制，不依赖调用方
- 训练异常（进程被杀/断连）时自动进阻尼模式
- workspace bounding box：任务配置给出，机器人侧再裁剪一次
