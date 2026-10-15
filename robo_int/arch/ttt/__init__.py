"""TTT（Test-Time Training）层挂件。

最小真实实现见 ttt_layer.py：TTT-Linear（arXiv:2407.04620 的线性快权重形式）
——prefix token 流上按 mini-batch 在线更新快权重 W（自监督重建目标
‖W k_t − v_t‖²），z_t = W_c q_t 经 wo 后按 gate 残差注入；gate=0/inactive
时 bit 级零效应。π0.5 注入与 MEM 同面（embed_prefix 包裹，server CLI
--arch-ttt）。RoboTTT 论文效果级复现（GR00T 基模、8K 上下文、层插入位置
精调）仍是 README TODO，当前验证目标是功能链路。
"""

from robo_int.arch.ttt.ttt_layer import TTTLinearLayer

TTTLayer = TTTLinearLayer  # README 设计文档中的旧类名

__all__ = ["TTTLinearLayer", "TTTLayer"]
