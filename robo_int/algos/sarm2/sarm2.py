"""SARM2：多任务阶段感知奖励模型（阶段估计器 + 多门 MoE 值解码器）。

论文：SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic
Manipulation（Chen et al., 2026, arXiv:2606.10305）。
**官方代码已开源**（[xdofai/opensarm](https://github.com/xdofai/opensarm)，SPIRAL 半边在
[Qianzhong-Chen/openspiral](https://github.com/Qianzhong-Chen/openspiral)）——本实现为
**官方代码的 mock 级移植**：结构/损失/训练流程对齐 opensarm，视觉编码器（冻结 SigLIP-2）
由原始状态向量扮演、规模缩小到 CPU 可跑。完整差异表见本目录 README.md，要点：

| 官方 opensarm | 本实现（mock 级） |
|---|---|
| 3 相机 SigLIP-2 冻结嵌入 + e_prim/e_task 文本嵌入 | 原始状态向量 + 可学习 nn.Embedding（task/prim） |
| 阶段估计器预训练 → 与值模型联合微调（单优化器） | 同款两阶段 update(phase) 分发（差异表 #5） |
| 辅助损失 λ_bal·var(load)+λ_imp·var(imp)−λ_ent·H | 同款（论文公式 (7) 符号有误，官方代码为准，README 差异 #4） |
| target = steps_to_go = 1−progress，sigmoid 输出 | 同款 |
| d=512, 6 层值模型 / 4 层阶段估计器, E=10×3×256, 30k 步 | d=64, 2 层, E=10×3×64, ~1k 步 |

用法（离线更新型算法，BaseAlgo）::

    algo = SARM2(primitives=..., groups=..., state_dim=6, num_tasks=3)
    algo.phase = "stage"                 # 阶段一：预训练阶段估计器
    for batch in stage_batches: algo.update(batch)
    algo.phase = "value"                 # 阶段二：联合训练值解码器（+微调估计器）
    for batch in value_batches: algo.update(batch)
    r_hat = algo.predict_value(states, task_id=0)   # 奖励/进度推断
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from robo_int.algos.base import BaseAlgo


# ---------------------------------------------------------------- 层次结构工具


def build_hierarchy(primitives: list[str], groups: list[str]) -> np.ndarray:
    """由语义分组字符串解析「原语 -> 组」索引表。

    官方 SARM2Workspace._build_hierarchy 同款机制：组描述形如
    ``"approach or retreat"``，按 " or " 拆词与原语名匹配。
    返回 (num_primitives,) int64 数组，未匹配到的原语归入第 0 组（null）。
    """
    prim_to_group = np.full(len(primitives), 0, dtype=np.int64)
    for g, desc in enumerate(groups):
        for name in desc.split(" or "):
            name = name.strip()
            if name in primitives:
                prim_to_group[primitives.index(name)] = g
    return prim_to_group


def collapse_score(avg_gate_probs: np.ndarray | torch.Tensor, top_k: int) -> float:
    """MoE 路由塌缩度 S_route（论文 Eq.6 / 官方 moe_utils._format_report）。

    r = E · mean(top-k 平均路由概率)，S = (r−1)/(E/k−1) ∈ [0,1]，
    0 = 均衡负载，1 = 完全塌缩到 top-k。与 N/k 无关，可直接对比不同规模。
    """
    p = np.asarray(avg_gate_probs, dtype=np.float64)
    E = p.shape[-1]
    r = E * np.sort(p)[-top_k:].mean()
    denom = E / top_k - 1.0
    return float(np.clip((r - 1.0) / denom, 0.0, 1.0)) if denom > 0 else 0.0


def _gate_mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    """门网络：LayerNorm -> Linear -> GELU -> Dropout -> Linear（官方 _make_mlp 同构）。"""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def _expert_mlp(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    """专家网络（= 输出头）：LN -> (Linear-GELU-Dropout)×3 -> Linear(1)（官方 3 隐层）。"""
    layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
    prev = in_dim
    for _ in range(3):
        layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- 阶段估计器


class StageEstimator(nn.Module):
    """动作基元阶段估计器（官方 ActionTransformer 的 mock 替身）。

    输入 mid 窗口状态序列 (B, N, S)，输出 (原语 logits (B, K), 组 logits (B, G))。
    - 状态投影 + 学习式时间位置编码（官方 time_pos_emb；无视觉流，mock 单流）
    - 因果 TransformerEncoder（官方 nn.TransformerEncoder + causal mask 同款）
    - 逐帧融合 MLP（LN->Linear->ReLU）后取**最后有效帧**（官方 fused_final）
    - 双头：原语分类头 + 组分类头（官方 task_head / class_head）
    """

    def __init__(
        self,
        state_dim: int,
        num_primitives: int,
        num_groups: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_window: int = 32,
    ):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, max_window, d_model))
        nn.init.normal_(self.time_pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU()
        )
        self.prim_head = nn.Linear(d_model, num_primitives)
        self.group_head = nn.Linear(d_model, num_groups)

    def forward(self, state_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """state_seq: (B, N, S) 固定长度 N（官方 act_pri 恒定输入 n_obs_steps）。"""
        B, N, _ = state_seq.shape
        x = self.state_proj(state_seq) + self.time_pos_emb[:, :, :N].squeeze(1)  # (B, N, D)
        causal = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(x, mask=causal, is_causal=True)  # (B, N, D)
        fused_last = self.fusion(h[:, -1])  # 最后帧（长度恒定，无需 gather）
        return self.prim_head(fused_last), self.group_head(fused_last)


# ---------------------------------------------------------------- 值解码器


class MMoEValueDecoder(nn.Module):
    """多门 MoE 值（进度）解码器（官方 RewardTransformer 的 mock 替身）。

    每个时间步 3 路 token：state / e_prim / e_task（官方为 N 相机 + e_prim + e_task +
    proprio 的 mock 对应）。序列布局 [anchor, mid×N, rewind×r, pad]，
    rewind 帧以倒序拼在尾部（官方 FrameGapLeRobotDataset 的 ReWiND 式隐式负样本，
    阻止模型退化为绝对时间索引预测器）。

    - 位置信息只加在 state 流首帧（官方 first_pos「视觉锚」，防泄漏绝对时间）
    - token 序列 time-major：每时间步 3 路 token 相邻（[s,p,t]×T），帧间因果、
      同帧流间不屏蔽——布局与 causal mask 语义必须一致（见 forward 注释）
    - 因果 Transformer 后逐帧 concat 融合 -> (B, T, 3d)，MoE 作用在**每个有效时间步**
      （官方稠密逐步回归，不是只回归最后一帧）
    - 多门：每组一个门网络，gate_idx 按样本外部指定（官方：预测原语的语义组）
    - top-k=2 稀疏路由，**top-k 分数重归一化**后加权（官方 weights=topk/Σtopk）
    - 辅助损失（官方形式，README 差异 #4）::
          aux = λ_bal·var(expert_load) + λ_imp·var(importance) − λ_ent·H(gate)
    - sigmoid 输出 r ∈ (0,1)，target 为 steps_to_go = 1 − progress
    """

    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        num_primitives: int,
        num_groups: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        num_experts: int = 10,
        top_k: int = 2,
        gate_hidden: int = 64,
        expert_hidden: int = 64,
        lambda_balance: float = 2.5,
        lambda_entropy: float = 0.5,
        lambda_importance: float = 0.1,
        max_window: int = 16,
    ):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.lambda_balance, self.lambda_entropy = lambda_balance, lambda_entropy
        self.lambda_importance = lambda_importance
        # 官方：e_task/e_prim 为冻结 SigLIP 文本嵌入；mock 用可学习 Embedding 扮演（差异 #2）
        self.task_emb = nn.Embedding(num_tasks, d_model)
        self.prim_emb = nn.Embedding(num_primitives, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.first_pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.moe_in = 3 * d_model  # state + e_prim + e_task 三流逐帧 concat
        self.gate_nets = nn.ModuleList(
            [
                _gate_mlp(self.moe_in, gate_hidden, num_experts, dropout)
                for _ in range(num_groups)
            ]
        )
        self.experts = nn.ModuleList(
            [_expert_mlp(self.moe_in, expert_hidden, dropout) for _ in range(num_experts)]
        )

    def compose_prim_embedding(self, prim_probs: torch.Tensor) -> torch.Tensor:
        """e_prim = P(原语) @ 原语嵌入表（官方 soft 混合路径，finetune 时保持可微）。"""
        return prim_probs @ self.prim_emb.weight  # (B, K) @ (K, D) -> (B, D)

    def forward(
        self,
        state_seq: torch.Tensor,     # (B, T, S) 含 anchor/mid/rewind/pad
        task_idx: torch.Tensor,      # (B,)
        prim_vec: torch.Tensor,      # (B, D) e_prim（外部组合好，soft/hard/置零皆可）
        lengths: torch.Tensor,       # (B,) 每样本有效长度
        gate_idx: torch.Tensor,      # (B,) 每样本选用的门（语义组）
    ):
        """返回 (r (B,T) ∈ (0,1), aux_loss 标量, info dict)。"""
        B, T, _ = state_seq.shape
        device = state_seq.device
        eps = 1e-9

        # --- 三路 token：state / e_prim / e_task，各自沿时间广播 ---
        s = self.state_proj(state_seq)                                   # (B, T, D)
        p = prim_vec.unsqueeze(1).expand(B, T, prim_vec.shape[-1])       # (B, T, D)
        t = self.task_emb(task_idx).unsqueeze(1).expand(B, T, s.shape[-1])
        # time-major 布局 [s0,p0,t0, s1,p1,t1, ...]（flat 位 3t+a = 帧 t 的流 a），
        # 必须与下方 repeat_interleave 版 causal mask 的 time-major 语义一致。
        # 曾误用 stream-major（stack 后直接 reshape），mask 与布局错位导致每帧
        # 输出都能读到全部未来帧，steps_to_go 回归被污染（README 实现笔记 #7）
        x = torch.stack([s, p, t], dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, -1)

        # 官方：仅 state 流首帧加位置偏置（视觉锚；防绝对时间泄漏）。
        # time-major 下首帧 state 流就是 flat 位 0
        x[:, 0, :] = x[:, 0, :] + self.first_pos

        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        causal = causal.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)  # 同帧流间不屏蔽
        pad = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        key_pad = pad.repeat_interleave(3, dim=1)  # 每帧 3 个 token 共享该帧 pad 值
        h = self.transformer(x, mask=causal, src_key_padding_mask=key_pad, is_causal=False)
        h = h.view(B, T, 3, -1).reshape(B, T, self.moe_in)  # 逐帧融合（帧内 [s,p,t] 顺序保持）

        # --- 有效 token 上做多门 MoE（官方 valid_idx 同款） ---
        valid = ~pad
        valid_idx = valid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        Nv = valid_idx.numel()
        b_idx = valid_idx // T
        fused = h.reshape(B * T, self.moe_in)[valid_idx]                  # (Nv, moe_in)
        gate_sel = gate_idx[b_idx]                                       # (Nv,)

        gate_logits = torch.empty(Nv, self.num_experts, device=device)
        for g, net in enumerate(self.gate_nets):
            rows = (gate_sel == g).nonzero(as_tuple=False).squeeze(1)
            if rows.numel():
                gate_logits[rows] = net(fused[rows])
        gate_probs = F.softmax(gate_logits, dim=-1)                       # (Nv, E)

        topk_scores, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        # 官方：top-k 分数重归一化后加权（README 差异 #3）
        weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + eps)

        expert_out = torch.cat([e(fused) for e in self.experts], dim=-1)  # (Nv, E)
        picked = expert_out.gather(1, topk_idx)                           # (Nv, k)
        y_valid = (picked * weights).sum(dim=-1)                          # (Nv,)

        y = torch.zeros(B * T, device=device)
        y[valid_idx] = y_valid
        r = torch.sigmoid(y).view(B, T)

        # --- 辅助损失（官方 moe_deco_gate_reward_model.py 同款） ---
        one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # (Nv, k, E)
        expert_load = one_hot.sum(dim=(0, 1)) / (float(Nv * self.top_k) + eps)
        load_loss = expert_load.var(unbiased=False)
        ent = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
        importance = gate_probs.sum(dim=0)
        importance = importance / (importance.sum() + eps)
        importance_loss = importance.var(unbiased=False)
        aux_loss = (
            self.lambda_balance * load_loss
            + self.lambda_importance * importance_loss
            - self.lambda_entropy * ent  # 负号 = 最大化路由熵（论文公式 (7) 符号有误，README #4）
        )
        info = {
            "gate_entropy": float(ent.detach()),
            "expert_load": expert_load.detach(),
            "avg_gate_probs": gate_probs.mean(dim=0).detach(),
        }
        return r, aux_loss, info


# ---------------------------------------------------------------- 窗口构建（数据平面）


def build_stage_windows(
    demos: list[dict],
    n_obs_steps: int = 6,
    frame_gap: int = 2,
    anchor_stride: int = 2,
) -> dict:
    """示教 demo -> 阶段估计器训练窗口（numpy，数据平面）。

    每个窗口 = mid 帧序列（以 t 结尾、间隔 frame_gap、长度 n_obs_steps）+ t 处原语标签。
    返回 {"state_seq": (M, N, S), "primitive": (M,)}。
    """
    seqs, labels = [], []
    need = (n_obs_steps - 1) * frame_gap + 1
    for demo in demos:
        states, prims = demo["states"][:-1], demo["primitives"]
        for t in range(need - 1, len(prims), anchor_stride):
            idx = np.arange(t - (n_obs_steps - 1) * frame_gap, t + 1, frame_gap)
            seqs.append(states[idx])
            labels.append(prims[t])
    if not seqs:
        raise ValueError("demo 太短，切不出任何 mid 窗口")
    return {
        "state_seq": np.stack(seqs).astype(np.float32),
        "primitive": np.asarray(labels, dtype=np.int64),
    }


def build_value_windows(
    demos: list[dict],
    n_obs_steps: int = 6,
    frame_gap: int = 2,
    max_rewind_steps: int = 3,
    rew_prob: float = 0.8,
    anchor_stride: int = 2,
    seed: int = 0,
) -> dict:
    """示教 demo -> 值解码器训练窗口（官方 FrameGapLeRobotDataset 布局的 mock 版）。

    序列布局 ``[start] + mid(n_obs_steps) + rewind(r)``，r 以 rew_prob 概率
    在 1..max_rewind_steps 内随机（官方 rewind_flag 概率 0.8）；rewind 取 t 之前
    间隔 frame_gap 的帧并**倒序**拼接（ReWiND 式隐式负样本）。不足额补零。

    每个有效位置的 target = steps_to_go = 1 − progress（官方 sarm2.yaml
    use_future_step=True 路径）。返回::

        {"state_seq": (M, 1+N+R, S), "task": (M,), "primitive": (M,),
         "target": (M, 1+N+R), "lengths": (M,)}
    """
    rng = np.random.default_rng(seed)
    total = 1 + n_obs_steps + max_rewind_steps
    need = (n_obs_steps - 1) * frame_gap + 1
    seqs, tasks, labels, targets, lengths = [], [], [], [], []
    for demo in demos:
        states, prims, prog = demo["states"][:-1], demo["primitives"], demo["progress"]
        steps_to_go = 1.0 - prog
        for t in range(need - 1, len(prims), anchor_stride):
            mid = np.arange(t - (n_obs_steps - 1) * frame_gap, t + 1, frame_gap)
            r = 0
            if rng.random() < rew_prob:
                r = int(rng.integers(1, min(max_rewind_steps, t // frame_gap) + 1))
            rew = np.arange(t - r * frame_gap, t, frame_gap)[::-1] if r > 0 else np.array([], int)
            idx = np.concatenate([[0], mid, rew])  # anchor=episode 首帧 + mid + rewind(倒序)
            L = len(idx)
            seq = np.zeros((total, states.shape[-1]), dtype=np.float32)
            tgt = np.zeros(total, dtype=np.float32)
            seq[:L] = states[idx]
            tgt[:L] = steps_to_go[idx]
            seqs.append(seq)
            targets.append(tgt)
            tasks.append(demo["task_id"])
            labels.append(prims[t])
            lengths.append(L)
    if not seqs:
        raise ValueError("demo 太短，切不出任何值窗口")
    return {
        "state_seq": np.stack(seqs),
        "task": np.asarray(tasks, dtype=np.int64),
        "primitive": np.asarray(labels, dtype=np.int64),
        "target": np.stack(targets),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }


# ---------------------------------------------------------------- 算法主体


class SARM2(BaseAlgo):
    """SARM2 奖励模型（离线更新型：两阶段 update(batch)）。

    阶段 ``phase="stage"``：仅训阶段估计器（原语 CE + 组 CE）。
    阶段 ``phase="value"``：训值解码器；``finetune_stage=True`` 时估计器联合微调
    （官方 finetune_act_pri 同款，total = reward + w·(CE_prim + CE_group)）。

    消融开关（论文 Table 1 / 官方对应实现）：
    - ``single_gate=True``：w/o MG——所有样本走 0 号门（单门 MoE）
    - ``use_primitives=False``：w/o SE——e_prim 置零、门固定 null 组，
      退化为「无阶段感知」的多门 MoE（门失去语义来源）

    Args:
        primitives: 原语名列表（首元素约定为 null/dummy）。
        groups: 语义分组描述列表（"a or b" 形式，含 null 组）。
        state_dim: 状态维度（mock 为任务状态；真机为冻结视觉嵌入维度）。
        num_tasks: 任务数（e_task 嵌入表大小）。
    """

    def __init__(
        self,
        *,
        primitives: list[str],
        groups: list[str],
        state_dim: int,
        num_tasks: int,
        d_model: int = 64,
        stage_layers: int = 2,
        value_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_obs_steps: int = 6,
        frame_gap: int = 2,
        max_rewind_steps: int = 3,
        num_experts: int = 10,
        top_k: int = 2,
        gate_hidden: int = 64,
        expert_hidden: int = 64,
        lambda_balance: float = 2.5,
        lambda_entropy: float = 0.5,
        lambda_importance: float = 0.1,
        alpha_auxiliary: float = 0.2,
        finetune_stage: bool = True,
        stage_loss_weight: float = 0.25,
        single_gate: bool = False,
        use_primitives: bool = True,
        lr: float = 5e-4,
        weight_decay: float = 5e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float = 1.0,
        warmup_steps: int = 0,
        total_steps: int = 2000,
        seed: int = 0,
    ):
        super().__init__()
        self.primitives = list(primitives)
        self.groups = list(groups)
        self.num_primitives = len(primitives)
        self.num_groups = len(groups)
        self.num_tasks = num_tasks
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.single_gate = single_gate
        self.use_primitives = use_primitives
        self.finetune_stage = finetune_stage
        self.stage_loss_weight = stage_loss_weight
        self.alpha_auxiliary = alpha_auxiliary
        self.grad_clip = grad_clip
        self.warmup_steps, self.total_steps = warmup_steps, total_steps
        self._torch_rng = torch.Generator().manual_seed(seed)

        self.prim_to_group = torch.tensor(build_hierarchy(primitives, groups))
        self.stage_estimator = StageEstimator(
            state_dim,
            self.num_primitives,
            self.num_groups,
            d_model=d_model,
            n_layers=stage_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_window=n_obs_steps + 1,
        )
        self.value_decoder = MMoEValueDecoder(
            state_dim,
            num_tasks,
            self.num_primitives,
            self.num_groups,
            d_model=d_model,
            n_layers=value_layers,
            n_heads=n_heads,
            dropout=dropout,
            num_experts=num_experts,
            top_k=top_k,
            gate_hidden=gate_hidden,
            expert_hidden=expert_hidden,
            lambda_balance=lambda_balance,
            lambda_entropy=lambda_entropy,
            lambda_importance=lambda_importance,
            max_window=1 + n_obs_steps + max_rewind_steps,
        )
        self.phase = "stage"
        self._optimizer = None
        self._scheduler = None
        self._opt_phase = None
        self._total_updates = 0

    # ---------------------------------------------------------------- 优化器

    def _build_optimizer(self) -> None:
        """按当前 phase 建优化器+调度器（切换 phase 时重建）。官方单 AdamW 同款超参。"""
        if self.phase == "stage":
            params = list(self.stage_estimator.parameters())
        else:
            params = list(self.value_decoder.parameters())
            if self.finetune_stage:
                params += list(self.stage_estimator.parameters())
            else:
                for p_ in self.stage_estimator.parameters():
                    p_.requires_grad_(False)
        self._optimizer = AdamW(params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        schedulers = []
        milestones: list[int] = []
        if self.warmup_steps > 0:
            schedulers.append(
                LinearLR(self._optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup_steps)
            )
            milestones.append(self.warmup_steps)
        schedulers.append(
            CosineAnnealingLR(self._optimizer, T_max=max(self.total_steps - self.warmup_steps, 1), eta_min=self.lr * 0.01)
        )
        self._scheduler = (
            SequentialLR(self._optimizer, schedulers, milestones=milestones)
            if len(schedulers) > 1
            else schedulers[0]
        )

    # ---------------------------------------------------------------- 更新

    def update(self, batch: dict) -> dict:
        """一次梯度步。batch 结构由 phase 决定（见 build_stage/value_windows）。"""
        if self._optimizer is None or self._opt_phase != self.phase:
            self._build_optimizer()
            self._opt_phase = self.phase
        if self.phase == "stage":
            metrics = self._update_stage(batch)
        elif self.phase == "value":
            metrics = self._update_value(batch)
        else:
            raise ValueError(f"未知 phase={self.phase}；可选 stage / value")
        self._optimizer.step()
        self._scheduler.step()
        self._total_updates += 1
        return metrics

    def _update_stage(self, batch: dict) -> dict:
        state_seq = torch.as_tensor(batch["state_seq"])
        prim = torch.as_tensor(batch["primitive"])
        group = self.prim_to_group[prim]
        prim_logits, group_logits = self.stage_estimator(state_seq)
        ce_prim = F.cross_entropy(prim_logits, prim)
        ce_group = F.cross_entropy(group_logits, group)
        loss = ce_prim + ce_group
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.stage_estimator.parameters() if p.requires_grad], self.grad_clip
        )
        return {
            "loss": float(loss.detach()),
            "ce_prim": float(ce_prim.detach()),
            "ce_group": float(ce_group.detach()),
            "prim_acc": float((prim_logits.argmax(-1) == prim).float().mean()),
        }

    def _update_value(self, batch: dict) -> dict:
        state_seq = torch.as_tensor(batch["state_seq"])
        task = torch.as_tensor(batch["task"])
        prim_gt = torch.as_tensor(batch["primitive"])
        target = torch.as_tensor(batch["target"])
        lengths = torch.as_tensor(batch["lengths"])

        # 阶段估计（mid 窗口 = 序列 [1, 1+N)，官方 _slice_act_pri_inputs 同款切片）
        mid = state_seq[:, 1 : 1 + self.n_obs_steps]
        prim_logits, group_logits = self.stage_estimator(mid)
        prim_probs = F.softmax(prim_logits, dim=-1)

        # e_prim：联合微调用 soft 混合（可微），冻结时 hard（官方两分支同款）
        if not self.use_primitives:
            prim_vec = torch.zeros(
                prim_probs.shape[0],
                self.value_decoder.prim_emb.weight.shape[1],
                device=prim_probs.device,  # GPU 上与后续 stack 的设备一致（CPU 下无差别）
            )
        elif self.finetune_stage:
            prim_vec = self.value_decoder.compose_prim_embedding(prim_probs)
        else:
            prim_vec = self.value_decoder.prim_emb.weight[prim_logits.argmax(-1)]

        gate_idx = self.prim_to_group[prim_logits.argmax(-1)]
        if not self.use_primitives or self.single_gate:
            gate_idx = torch.zeros_like(gate_idx)  # 消融：固定 null 组 / 单门

        r, aux, info = self.value_decoder(state_seq, task, prim_vec, lengths, gate_idx)

        # 稠密逐步回归（只算有效位置），total = mse + α_aux·aux + w·(CE)（官方同款）
        valid = (
            torch.arange(target.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        ).float()
        mse = ((r - target) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)
        reward_loss = mse + self.alpha_auxiliary * aux
        loss = reward_loss
        metrics = {
            "loss": float(loss.detach()),
            "mse": float(mse.detach()),
            "aux_loss": float(aux.detach()),
            "gate_entropy": info["gate_entropy"],
            "s_route": collapse_score(info["avg_gate_probs"], self.value_decoder.top_k),
        }
        if self.finetune_stage:
            ce_prim = F.cross_entropy(prim_logits, prim_gt)
            ce_group = F.cross_entropy(group_logits, self.prim_to_group[prim_gt])
            loss = reward_loss + self.stage_loss_weight * (ce_prim + ce_group)
            metrics["ce_prim"] = float(ce_prim.detach())
            metrics["prim_acc"] = float((prim_logits.argmax(-1) == prim_gt).float().mean())
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in (self.value_decoder, self.stage_estimator) for p in g.parameters() if p.requires_grad],
            self.grad_clip,
        )
        metrics["loss"] = float(loss.detach())
        return metrics

    # ---------------------------------------------------------------- 推断

    @torch.no_grad()
    def predict_stage(self, mid_window: np.ndarray) -> np.ndarray:
        """mid 窗口 (N, S) -> 原语概率 (K,)。"""
        was_training = self.stage_estimator.training
        self.stage_estimator.eval()
        logits, _ = self.stage_estimator(torch.as_tensor(mid_window[None], dtype=torch.float32))
        if was_training:
            self.stage_estimator.train()
        return F.softmax(logits, dim=-1)[0].numpy()

    @torch.no_grad()
    def predict_value(self, states: np.ndarray, task_id: int) -> np.ndarray:
        """整条状态序列 (T, S) -> 每帧预测值 r̂ (T,)（steps_to_go，越小越接近完成）。

        与训练窗口同构：对每个可评估帧 t（凑齐 mid 历史），构造
        ``[anchor=首帧, mid(以 t 结尾、间隔 frame_gap)]`` 无 rewind 窗口，
        取最后有效位置输出。t 不足历史的头部位置填 NaN。
        内部自动跑阶段估计取 e_prim / gate（推断路径，hard 选择）。
        所有窗口一次批量前向（曾是每帧单独前向 O(T) 次网络调用，
        打分管线逐 episode 评分的瓶颈）。
        """
        # 两个模块的训练态分别记录恢复：只按 stage_estimator 记录会把
        # 原本 eval 的 value_decoder 误切回 train（dropout 被打开）
        was_training_stage = self.stage_estimator.training
        was_training_value = self.value_decoder.training
        self.stage_estimator.eval()
        self.value_decoder.eval()
        try:
            states = np.asarray(states, dtype=np.float32)
            T, _ = states.shape
            g, N = self.frame_gap, self.n_obs_steps
            need = (N - 1) * g + 1
            r_hat = np.full(T, np.nan, dtype=np.float32)
            if T >= need:
                ts = np.arange(need - 1, T)
                # 批量构造所有窗口：(M, 1+N, S)，anchor=首帧 + mid(以 t 结尾)
                seqs = np.stack(
                    [
                        np.concatenate([states[:1], states[t - (N - 1) * g : t + 1 : g]])
                        for t in ts
                    ]
                )
                lengths = np.full(len(ts), 1 + N, dtype=np.int64)
                r_hat[ts] = self._forward_windows(seqs, task_id, lengths)
        finally:
            if was_training_stage:
                self.stage_estimator.train()
            if was_training_value:
                self.value_decoder.train()
        return r_hat

    def _forward_windows(self, seqs: np.ndarray, task_id: int, lengths: np.ndarray) -> np.ndarray:
        """批量窗口推断（anchor+mid），返回每窗口最后有效位置的 r̂ (M,)。"""
        seq_t = torch.as_tensor(seqs, dtype=torch.float32)
        logits, _ = self.stage_estimator(seq_t[:, 1:])
        probs = F.softmax(logits, dim=-1)
        prim_vec = self.value_decoder.compose_prim_embedding(probs)
        gate_idx = self.prim_to_group[logits.argmax(-1)]
        if not self.use_primitives or self.single_gate:
            gate_idx = torch.zeros_like(gate_idx)
        r, _, _ = self.value_decoder(
            seq_t,
            torch.full((len(seqs),), int(task_id), dtype=torch.long),
            prim_vec,
            torch.as_tensor(lengths),
            gate_idx,
        )
        last = torch.as_tensor(np.asarray(lengths) - 1)
        return r[torch.arange(len(seqs)), last].numpy()

    # ---------------------------------------------------------------- 存取

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage_estimator": self.stage_estimator.state_dict(),
                "value_decoder": self.value_decoder.state_dict(),
                "phase": self.phase,
                "total_updates": self._total_updates,
            },
            path,
        )

    def load_state(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.stage_estimator.load_state_dict(payload["stage_estimator"])
        self.value_decoder.load_state_dict(payload["value_decoder"])
        self.phase = payload["phase"]
        self._total_updates = payload["total_updates"]
