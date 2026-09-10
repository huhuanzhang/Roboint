"""RLT（RL Token）：冻结 VLA + RL token 读出 + actor-critic。

论文 arXiv:2604.23073（Physical Intelligence，无官方代码）。
实现与论文的差异表见本目录 README.md。
"""

from robo_int.algos.rlt.rlt import RLT, RLTokenHead

__all__ = ["RLT", "RLTokenHead"]
