"""架构改进挂件（arch）抽象。

定位：MEM / TTT / attention 变体 / LoRA 都是「可挂到任意 BaseVLA 上的模块」，
不是新基模。目标是 A/B 对比时零侵入——换挂件不换底座代码。

挂载模型（attach/detach）而非继承，原因：
- 同一挂件要能配 π0.5 也能配 GR00T；
- 训练脚本按配置组装 model + arch_modules，卸载后基模保持原样便于对照。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from robo_int.models.base import BaseVLA


class BaseArchModule(ABC):
    """架构改进模块的统一接口。

    实现方约定：
    - attach 后对模型的修改必须能被 detach 完整撤销（A/B 实验的正确性依赖这一点）；
    - 挂件自身的可训练参数用 parameters() 暴露，交给训练循环统一优化；
    - 挂件不能改变 BaseVLA 的对外接口签名（predict_actions / compute_loss /
      extract_features 的输入输出结构不变，只改内部信息流）。
    """

    name: str = "arch_module"

    @abstractmethod
    def attach(self, model: BaseVLA) -> None:
        """挂载到基模：替换/包裹内部组件、注册 forward hook 等。"""

    @abstractmethod
    def detach(self) -> None:
        """卸载：完整撤销 attach 造成的影响。"""

    def parameters(self):
        """挂件自身的可训练参数（默认空，子类按需重写）。"""
        return iter(())
