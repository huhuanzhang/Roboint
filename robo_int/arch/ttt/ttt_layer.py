"""TTT-Linear 层挂件（arch/ttt/README.md 设计的最小真实实现）。

理论源头 arXiv:2407.04620（Learning to (Learn at Test Time): TTT-Linear）；
工程对齐目标 RoboTTT（NVIDIA GEAR，TTT 层嵌入 VLA 的 visuomotor token 流）。
本文件是可移植核心，仅依赖 torch：

- TTTLinearLayer：序列建模层——快权重（fast weights）W ∈ R^{d_model×latent}
  在序列上做 mini-batch 在线梯度更新（自监督重建目标 ℓ(W; k_t, v_t) =
  ‖W k_t − v_t‖²，联想记忆语义：学 key→value 映射），输出 z_t = W_c q_t
  （c = t 所在 chunk 起点，即论文的 mini-batch TTT：块内查询共享块起点的 W）；
- 残差注入：out = x + gate · norm_out(wo(z))；gate=0 / inactive 时 x 原对象
  原样返回（bit 级零效应，与 AQV/MEM 的零效应门槛同口径）；
- 更新规则 W ← W − lr · (1/C_c) Σ_{t∈c}(W k̂_t − v_t)k̂_tᵀ（mini-batch GD
  取块均值；ℓ 对 W 的梯度本带 2 因子，已吸收进 lr）；k/q 做 per-token
  单位归一（k̂/q̂，尺度可被可学习投影吸收）：真实 prefix token 的
  ‖k‖² ~ O(10²)，块求和梯度范数 O(10⁴)，未归一时 lr=0.1 即发散
  （实测 w_end=inf）；归一后每块收缩因子 (1 − lr/C) < 1，lr ≤ 1 时
  构造性不发散（构造函数强制）；
- fast weights 是推理期状态不是可训练参数：persist=False 时每个请求从
  fast_init（可学习 W₀，零初始化）重开（论文 per-sequence 语义）；
  persist=True 时跨请求保留（RoboTTT「上下文扩展」风味），session 边界
  仍由 reset_episode 重置；梯度不跨请求回传（截断，如实记录的简化）；
- fp32 内部计算（π0.5 推理是 bf16，快权重循环在 fp32 做数值稳定），
  输出残差 cast 回输入 dtype。

π0.5 注入（与 MEM 同面）：attach 包裹 model.embed_prefix，对 prefix token
embedding 序列（(B, L, d_model)，L=prefix_len）做上述变换；sample_actions
与训练 forward 同路径（原生 flow-matching loss 可经残差分支反传到本层）。

参数状态：本模块参数随机初始化、未经训练——本仓库阶段目标是功能链路验证
（forward/backward/save-load/dry-run），不是 RoboTTT 论文效果复现（GR00T
基模、8K 时间步上下文、门控/尺寸精调都是 README TODO）。
"""

from __future__ import annotations

import torch
from torch import nn

from robo_int.arch.base import BaseArchModule


class TTTLinearLayer(nn.Module, BaseArchModule):
    """TTT-Linear 序列层：快权重在线自监督更新 + 残差注入。

    effective 行为由 ``active`` 开关与 ``gate`` 共同决定（server CLI 注入）：
    - active=False 或 gate=0：forward_tokens 返回输入原对象（零效应基础设施）；
    - active=True 且 gate>0：prefix token 流经 TTT 变换后残差注入。
    """

    name = "ttt"

    def __init__(self, d_model: int = 2048, latent_dim: int = 128, lr: float = 0.1,
                 gate: float = 0.0, chunk_size: int = 64, persist: bool = False):
        nn.Module.__init__(self)
        if d_model < 1 or latent_dim < 1:
            raise ValueError(f"d_model={d_model} / latent_dim={latent_dim} 必须 >= 1")
        if chunk_size < 1:
            raise ValueError(f"chunk_size={chunk_size} 必须 >= 1")
        if not 0.0 < lr <= 1.0:
            raise ValueError(f"ttt lr={lr} 需在 (0, 1]（k̂/q̂ 单位归一下的稳定域）")
        if not 0.0 <= gate <= 1.0:
            raise ValueError(f"gate={gate} 需在 [0, 1]（0=零效应）")
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.gate = float(gate)
        self.chunk_size = int(chunk_size)
        self.persist = bool(persist)
        self.active = False
        self._fast_state: torch.Tensor | None = None  # (B, D, K)，detached 推理期状态
        self._original_embed_prefix = None
        self._attached_model = None
        self._shadowed_instance_attr = False
        self.last_stats: dict = {}

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)
        self.wq = nn.Linear(self.d_model, self.latent_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.latent_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fast_init = nn.Parameter(torch.zeros(self.d_model, self.latent_dim))

    # ---------------------------------------------------------------- 快权重状态

    def reset_episode(self) -> None:
        """session/episode 边界：清持久快权重（per-sequence 语义的兜底）。"""
        self._fast_state = None
        self.last_stats = {}

    @property
    def fast_state_persisted(self) -> bool:
        return self._fast_state is not None

    # ---------------------------------------------------------------- TTT 前向

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """对 (B, L, d_model) 序列做 TTT-Linear 变换。

        返回 (out, recon_loss)：recon_loss = Σ_c Σ_{t∈c} ‖W_c k_t − v_t‖²
        （自监督重建目标，训练期可反传；推理路径不消费）。零效应时 (x, None)。
        """
        if not self.active or self.gate == 0.0:
            self.last_stats = {
                "applied": False,
                "reason": "inactive" if not self.active else "zero_gate",
            }
            return x, None
        batch, seq_len, _ = x.shape
        dtype_in = x.dtype
        device = x.device
        # fp32 内部计算；关 autocast 防混入半精度
        with torch.autocast(device_type=device.type, enabled=False):
            h = self.norm_in(x.float())
            # k/q 单位归一：数值稳定（见类 docstring），尺度由可学习投影吸收
            q = torch.nn.functional.normalize(self.wq(h), dim=-1)  # (B, L, K)
            k = torch.nn.functional.normalize(self.wk(h), dim=-1)  # (B, L, K)
            v = self.wv(h)  # (B, L, D)
            reused_state = False
            # _fast_state 不是注册 buffer（.to() 不会搬它），此处跨设备搬移，
            # 否则 load()->to(device) 顺序下状态会被静默丢弃从零重开
            if (self.persist and self._fast_state is not None
                    and tuple(self._fast_state.shape) == (batch, self.d_model, self.latent_dim)):
                w = self._fast_state.detach().to(device=device, dtype=torch.float32).clone()
                reused_state = True
            else:
                w = self.fast_init.unsqueeze(0).expand(batch, -1, -1).clone()
            w_start_norm = float(w.detach().norm())
            chunk = self.chunk_size
            outs: list[torch.Tensor] = []
            recon = torch.zeros((), device=device, dtype=torch.float32)
            for start in range(0, seq_len, chunk):
                end = min(start + chunk, seq_len)
                qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
                outs.append(torch.einsum("bdk,bck->bcd", w, qc))  # 块内查询共享块起点 W
                err = torch.einsum("bdk,bck->bcd", w, kc) - vc   # (B, C, D)
                recon = recon + err.pow(2).sum()
                # mini-batch GD：块均值梯度（1/C_c）
                grad = torch.einsum("bcd,bck->bdk", err, kc) / (end - start)
                w = w - self.lr * grad
            z = self.wo(torch.cat(outs, dim=1))                  # (B, L, D)
            delta = self.gate * self.norm_out(z)
            out = x + delta.to(dtype_in)
            if self.persist:
                self._fast_state = w.detach()
        with torch.no_grad():
            out_d = out.detach()
            self.last_stats = {
                "applied": True,
                "seq_len": int(seq_len),
                "batch": int(batch),
                "chunks": (seq_len + chunk - 1) // chunk,
                "chunk_size": int(chunk),
                "reused_state": reused_state,
                "w_start_norm": w_start_norm,
                "w_end_norm": float(w.detach().norm()),
                "w_delta_norm": float((w.detach() - self.fast_init.detach().unsqueeze(0)).norm()),
                "recon_loss": float(recon.detach()),
                "recon_per_token": float(recon.detach()) / max(batch * seq_len, 1),
                "out_delta_l2": float((out_d.float() - x.detach().float()).norm()),
                "out_nan_count": int(torch.isnan(out_d).sum()),
                "out_inf_count": int(torch.isinf(out_d).sum()),
                "persistent": self.persist,
            }
        return out, recon

    # ---------------------------------------------------------------- 生命周期（BaseArchModule）

    def attach(self, model) -> None:
        """包裹 model.embed_prefix（π0.5 系接口：sample_actions/训练 forward 都经
        self.embed_prefix 属性查找，实例级遮蔽生效）。"""
        if self._original_embed_prefix is not None:
            return
        if not callable(getattr(model, "embed_prefix", None)):
            raise ValueError("TTT.attach 需要模型暴露 embed_prefix（π0.5 系接口）")
        # 记录原值来自实例 __dict__（可能是别人的包裹，如 MEM）还是类方法，
        # detach 按来源恢复：类方法 -> del 实例遮蔽；实例值 -> 原样写回。
        self._shadowed_instance_attr = "embed_prefix" in model.__dict__
        original = model.embed_prefix
        ttt = self

        def wrapped_embed_prefix(images, img_masks, tokens, masks):
            embs, pad_masks, att_masks = original(images, img_masks, tokens, masks)
            if not ttt.active or ttt.gate == 0.0:
                return embs, pad_masks, att_masks
            new_embs, _recon = ttt.forward_tokens(embs)
            return new_embs, pad_masks, att_masks

        model.embed_prefix = wrapped_embed_prefix
        self._original_embed_prefix = original
        # object.__setattr__ 绕过 nn.Module 的子模块注册：被挂载的 pi0.5 不得
        # 泄漏进 state_dict()/parameters()（B3 的教训）
        object.__setattr__(self, "_attached_model", model)

    def _restore_embed_prefix(self) -> None:
        """把 embed_prefix 恢复到 attach 前的解析状态（类方法或原实例值）。"""
        model = self._attached_model
        if self._shadowed_instance_attr:
            model.__dict__["embed_prefix"] = self._original_embed_prefix
        elif "embed_prefix" in model.__dict__:
            del model.__dict__["embed_prefix"]  # 撤销实例遮蔽，回落类方法

    def detach(self) -> None:
        """完整撤销 attach（A/B 实验正确性依赖）；快权重状态保留。"""
        if self._original_embed_prefix is not None:
            self._restore_embed_prefix()
            self._original_embed_prefix = None
        self._attached_model = None

    # ---------------------------------------------------------------- 存取

    def save(self, path) -> dict:
        """保存参数 + 持久快权重（不触碰 π0.5 checkpoint）。"""
        payload = {
            "state_dict": self.state_dict(),
            "fast_state": (self._fast_state.detach().cpu()
                           if self._fast_state is not None else None),
            "config": {
                "d_model": self.d_model, "latent_dim": self.latent_dim,
                "lr": self.lr, "gate": self.gate,
                "chunk_size": self.chunk_size, "persist": self.persist,
            },
        }
        torch.save(payload, path)
        return payload["config"]

    def load(self, path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(payload["state_dict"])
        state = payload.get("fast_state")
        self._fast_state = state.to(self.fast_init.device) if state is not None else None
