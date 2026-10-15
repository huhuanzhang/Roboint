"""MEM 多尺度具身记忆挂件（arXiv:2603.03596）。

最小真实实现见 memory.py：短时特征历史 temporal cross-attention + 长时文本
记忆库（write/retrieve/forget），记忆 token 经 embed_prefix 包裹注入 π0.5
prefix 上下文（Action Expert 与 VLM 自然 attend）。π0.5 server 接线见
robo_int/models/pi05/server.py（--arch-mem）。论文效果级复现（视频编码支路、
长时序任务增益）仍是 README TODO，当前验证目标是功能链路。
"""

from robo_int.arch.mem.memory import MEMModule, TextMemoryBank

__all__ = ["MEMModule", "TextMemoryBank"]
