# B3 MEM 问题清单

> 阶段:B3 多尺度具身记忆(MEM)功能验证,2026-09-11。
> 范围:本阶段开发/验证过程中**实际遇到**的问题、根因、修复与验证结果,外加当前仍存在的已知边界。
> 结论:所列代码问题已全部闭环;最终 22/22 GPU 检查 + 9 项单测 PASS,pytest 113 passed + 2 skipped + 8 error(环境门所致,非代码回归)。

## 一、模块真实 bug(已修,按严重度)

### 1. attach 把整个 pi0.5 注册成子模块(P0,最严重)
- 现象:save->新实例 load 报 `RuntimeError: Error(s) in loading state_dict for MEMModule: Unexpected key(s): "_attached_model.paligemma..."`(数千键);save 出的文件含全部 pi0.5 权重。
- 根因:`MEMModule` 继承 `nn.Module`,`self._attached_model = model` 被 `nn.Module.__setattr__` 当作子模块注册进 `_modules`,于是 `state_dict()` / `parameters()` / `.to()` 全部把 pi0.5 算进来。
- 连带影响:
  - "optimizer 只持有 MEM 参数"检查在修复前是**空真**(两侧集合都被污染成全模型,相等恒成立);
  - 训练段 `before = [p.clone() ...]` 会克隆整个基座(显存浪费);
  - checkpoint 体积 GB 级,违背"不触碰 pi0.5 150000"的边界精神。
- 修复:`object.__setattr__(self, "_attached_model", model)` 绕过子模块注册,引用只进实例 `__dict__`(见 `robo_int/arch/mem/memory.py` attach)。
- 验证:修复后 save 文件 134,436,439 B(约 134MB)= MEM 33.6M 参数 x 4B;load 无 unexpected keys;`policy.mem.parameters()` 恰为 11 个参数张量;optimizer 持有集与 MEM 参数集严格相等(不再空真)。
- 防再犯:任何"挂件持有宿主模型引用"的属性都必须 `object.__setattr__`;自查手段——看 `len(list(module.parameters()))` 与 state_dict 键数。

### 2. detach 无法用 `is` 恢复原方法(bound-method identity)(P1)
- 现象:单测 `test_detach_restores_original_embed_prefix` FAILED。
- 根因:`model.embed_prefix` 每次属性访问都会新建一个 bound method 对象,`original is model.embed_prefix` 恒 False,也无法据此判断"挂载前是否存在实例属性遮蔽"。
- 修复(改模块,不是给测试放水):attach 记录 `_shadowed_instance_attr`(挂载前 `embed_prefix` 是否已在 `model.__dict__`);`_restore_embed_prefix()` 据此恢复——shadow 场景写回原值,否则 `del model.__dict__["embed_prefix"]` 回落到类方法。
- 验证:单测全过;detach 后再推理确认无注入。

### 3. `write_memory` 没把编码结果传进 bank(P1,开发期自查)
- 现象:自查发现 `write_memory(text, key)` 调 `bank.write` 时漏传 `content_emb`,且残留占位方法 `write_memory_with_emb`。
- 修复:重写为正确传 emb 的一次性编码写入,删除占位方法。
- 验证:GPU 测试"真实文本库写入 bank_size=4"与"文本库 reload 后检索一致" PASS。

### 4. server.py 的 MEM monitor 记录块误锚进 AQV 块内(P1)
- 现象:run1 输出 `this_request=0 total=0 HOOK INACTIVE`,随后测试侧 `np.isfinite(None)` TypeError。
- 根因:补丁锚点选了 AQV 的 per-camera 循环,`if self.mem is not None:` 被嵌进 `if self.aqv is not None:` 内部;aqv=None 时 MEM 记录永远不执行。测试侧又没对空统计做防御,把问题掩盖成第二个报错。
- 修复:MEM 记录块 dedent 到与 AQV 块平级;测试对 `mean is None` 先判再 `isfinite`。
- 验证:重跑 hook `this_request=1 total=7 ACTIVE`、mem_tokens 统计 finite PASS;live probe 中 hook total 1->4。

## 二、测试脚本 bug(已修)

### 5. 原生 loss 入口用错对象
- 现象:`AttributeError: 'SmolVLAPiperPolicy' object has no attribute 'forward'`。
- 根因:原生训练入口是 `PI05Policy.forward`(modeling_pi05.py 1243 行起,非 no_grad 装饰);server 包装类没有同名方法。
- 修复:两处(初始 loss 与 optimizer 循环)改为 `policy.policy.forward(batch)`。

### 6. 训练 batch 设备不对齐
- 现象:`RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0`(action_in_proj 处)。
- 根因:`PI05Policy.forward` 不搬 batch(正式训练由数据加载/accelerate 保证已在 GPU);noise/time 是随 `actions.device` 采样的,因此只要把 batch 整体对到模型设备即可。
- 修复:构造 batch 后 `{k: v.to(dev) for ...}` 一次对齐。

### 7. save/load "同输入"用了过期历史快照
- 现象:run5 唯一 FAIL:`save->new instance->load 同输入同输出(bit 级) max_diff=7.643e-03`。
- 根因:纯测试逻辑错——前面性能 bench 的 3 次 `policy.infer()` 会像真实服务一样 push 当前 obs 特征,`deque(maxlen=4)` 被冲刷;比较时 `policy.mem` 的真实历史已不是 `hist_a`,而 fresh 灌的还是旧 `hist_a`,两边输入根本不同。
- 修复:fresh 改灌 `policy.mem._history` 当前内容;检索 query 同步改。
- 验证:bit 级 0.000e+00 PASS。
- 教训:断言"同输入同输出"前必须先证明输入相同,不能假设中间步骤没有副作用。

### 8. probe 三连错(环境+解析)
- a) `ModuleNotFoundError: No module named 'lerobot'`——probe 忘带 PYTHONPATH。
- b) `ImportError: cannot import name 'RTCAttentionSchedule' from 'lerobot.configs'`——lerobot-piper-2 的 lerobot 没有该符号;导入 fork server 模块(只为拿 Packer/unpackb)必须用 venv-smolvla 解释器 + `PYTHONPATH=$PWD`。
- c) `TypeError: unsupported format string passed to NoneType.__format__`——monitor report 的 tensor 条目是统计 dict(mean/min/max/l2_norm/shape/n_samples...),没有 `values` 键;REQ0 空记忆不注入时统计为空。改防御性解析后通过。

## 三、环境/操作问题(非代码)

### 9. GPU 被外来 server 占满 -> 首次 GPU 测试 OOM
- 现象:CUDA OOM;nvidia-smi 见 pid 2685039 占 36.8GB。
- 根因:zhengdongchen 当晚新起的 openpi JAX server(8012)。用户早前"8012 我不用了"只覆盖当时的旧实例,不构成杀新实例的授权。
- 处置:不杀,报 BLOCKED 等确认;用户回复"server 退出了"后核实 GPU 空闲再继续。

### 10. 8 个 pytest error:环境门被外来 server "打穿"
- 现象:全量回归 113 passed + 2 skipped + 8 errors(0 FAILED)。
- 根因:rlt_native x4 + sarm2_real x4 的门是"8012 端口关 -> skip";外来 server 占端口后门变成"开但元数据缺 `required_policy_cameras` -> error"。基线 104 项全保持,9 项新 MEM 单测全过,非代码回归。
- 处置:如实记录,不改代码、不放宽门。

### 11. `pkill -f` 自匹配自杀 + 模式 kill 被权限拦截
- 现象:收尾 `pkill -9 -f 'pi05/server.py'` 直接把 Monitor 任务自己杀掉(exit 1 无输出)——bash -c 命令行本身含该字符串。另一次同类操作被权限分类器拒绝(模式匹配会波及机器上所有 pi05 server,含外来 8012)。
- 连带:更早一次 `SERVER_STILL_RUNNING` 也是 pgrep 自匹配的**误报**(stderr 已显示"已终止")。
- 规范:收尾只 kill 本任务跟踪的 SERVER_PID;readiness 用端口 connect 判断(本阶段一贯如此,未依赖 grep 日志行)。

## 四、确认过的"非问题"(防再误读)

### 12. REQ0 不注入 = 正确时序语义
- server 在**推理之后**才 push 当前 obs 特征(过去观测影响当前动作)。因此新 session 的第一个请求空记忆、不注入、monitor `n_samples=0`,这是设计而非失效;REQ1 起 history 1->2->3、hook total 递增、token 统计出现。
- 元数据里 `bank_size=0` 同理:connect 时尚未写入任何文本。

### 13. smoke 阶段 TypeError
- 是冒烟脚本自己的调用传参错(直接调 wrapped `embed_prefix` 没给全参数),模块本身正确;用正确参数复跑即过。

## 五、当前已知边界(未修,如实声明)

1. **MEM 参数未训练**:本阶段只验链路(forward/backward/save/load/dry-run);初始化参数下的影响数字(impact cos 0.9555 等)不构成任何效果结论,效果实验被本轮规则明确禁止。
2. **论文级复现是 TODO**:MEM 论文(arXiv:2603.03596)无官方代码;当前实现是 repo README 设计下的最小真实版本(短时=vlm_prefix_mean 历史交叉注意力,而非论文的视频 encoder 压缩;长时=文本库)。
3. **长时文本来源在客户端**:推理时服务器不自动产生文本,依赖 observation 的 `mem_texts` 字段带入(GPU 测试与 probe 均用真实夹爪测量文本演示)。
4. **bank 不持久**:server 进程内存态,重启即清;`MEMModule.save/load` 已验证可用,但 server CLI 尚无 `--mem-state` 自动恢复入口(不在本阶段范围)。
5. **短时表征信息量有限**:vlm_prefix_mean 是 2048D 均值池化(不含 robot state),这是 extra_forward=0 的代价——当前 obs 不会再跑一遍 pi0.5。

## 六、防再犯 checklist(B4 TTT 起逐条对照)

- [ ] 挂件持有宿主模型引用 -> `object.__setattr__`;验证 `len(list(module.parameters()))`
- [ ] attach/detach 用 shadow 标记,不比较 bound method 身份
- [ ] monitor 记录块与既有挂件块平级(不嵌套)
- [ ] 手工训练 batch -> `.to(next(model.parameters()).device)`
- [ ] 原生 loss -> `PI05Policy.forward`(经 `policy.policy`)
- [ ] "同输入同输出"断言前先对齐真实输入(bench 有副作用)
- [ ] 导入 fork server 模块 -> venv-smolvla python + `PYTHONPATH=$PWD`
- [ ] 收尾只 kill 自己跟踪的 PID;readiness 只看端口
- [ ] 外来 GPU 进程一律不杀,先报告

## 附:最终验证状态(修复后)

- GPU functional test(`scripts/pi05_mem_functional_test.py`):22/22 PASS(TEST_EXIT=0)
- 单测(`tests/test_mem_module.py`):9/9;全量 pytest **113 passed + 2 skipped + 8 error**(环境门)
- 关键数字:双零效应 parity bit 级 0.0(disabled 与 active+空记忆);同 obs 不同真实历史 token diff 1.5077e-02 / action diff 2.2668e-02;impact mean 3.7308e-02 max 2.3900e-01 l2 1.6378e+00 cos 0.955530;原生 loss 1.684845、梯度 11/11 MEM 参数、base VLM/Expert 零改动;save/load bit 级 0.0;extra_forward=0;+1.5% 推理耗时;Piper dry-run 12 cycles exit=0 command_sent=false。
