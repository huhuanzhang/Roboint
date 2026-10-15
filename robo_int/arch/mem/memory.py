"""MEM 多尺度具身记忆挂件（arch/mem/README.md 设计的最小真实实现）。

论文 arXiv:2603.03596（无官方代码，README 记录查证）：短时 video-based 记忆 +
长时 text-based 记忆，混合模态注入 VLA。本文件是可移植核心，仅依赖 torch/numpy：

- TextMemoryBank：长时文本记忆库（写入 / 余弦相似度检索 / 容量 FIFO 遗忘；
  跨 episode 保留——长时语义记忆的 scope 语义）；
- MEMModule：记忆 token 计算——
  * 短时支路：episode 内最近 K 步的 vlm_prefix_mean(2048D，视觉+语言 prefix
    均值，**不含 robot state**) 上做 learned-query temporal cross-attention。
    这些特征是过去请求已计算的（fork server feature capture 缓存），MEM 不为
    当前观测额外跑一次 π0.5；
  * 长时支路：检索到的文本条目 embedding 上做 cross-attention。文本 embedding
    由调用方注入 encode_text 回调产生（π0.5 上 = checkpoint tokenizer + 冻结
    embed 层，不训练、无 transformer forward）；
  * 融合：LayerNorm(query + short + long) -> (B, num_tokens, d_model) 记忆 token。

π0.5 注入（token 级）：attach 包裹 model.embed_prefix，把记忆 token 追加到
(embs, pad_masks, att_masks) 末尾（pad=1 真实 token、att=0 与 prefix 同块双向）。
prefix pass 与 Action Expert 全部 denoise 步经既有 mask 机制自然 attend，
不改 denoise/mask 代码；训练 forward 同路径（原生 flow-matching loss 可反传）。

零效应保证：active=False 或无可记忆内容（history 空 且 bank 空）时
compute_tokens 返回 None、embed_prefix 原对象原样返回 -> bit 级 parity。

参数状态：本模块的参数是随机初始化、未经训练的——本仓库阶段目标是功能链路
验证（forward/backward/save-load/dry-run），不是论文效果复现。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from robo_int.arch.base import BaseArchModule


class TextMemoryBank:
    """长时文本记忆库：容量 FIFO 遗忘 + 特征余弦相似度检索。

    条目 = (text, context_key, content_emb)：context_key 是写入时刻的观测特征
    （检索 query 同空间），content_emb 是文本内容嵌入（无 encode_text 回调时
    为 None，长时支路如实跳过该条目）。
    """

    def __init__(self, capacity: int = 16):
        if capacity < 1:
            raise ValueError("bank capacity 必须 >= 1")
        self.capacity = capacity
        self._entries: deque[tuple[str, np.ndarray, np.ndarray | None]] = deque()

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, text: str, context_key: np.ndarray,
              content_emb: np.ndarray | None = None) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text 必须是非空字符串")
        key = np.asarray(context_key, dtype=np.float32).reshape(-1)
        if len(self._entries) >= self.capacity:  # 遗忘最旧
            self._entries.popleft()
        self._entries.append((text.strip(), key, content_emb))

    def retrieve(self, query: np.ndarray, k: int = 1) -> list[tuple[str, float]]:
        """按与 query 的余弦相似度取 top-k 文本（稳定排序，零向量相似度记 0）。"""
        if not self._entries:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        scored: list[tuple[str, float]] = []
        for text, key, _emb in self._entries:
            kn = float(np.linalg.norm(key))
            sim = float(np.dot(q, key) / (qn * kn)) if qn > 0 and kn > 0 else 0.0
            scored.append((text, sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, int(k))]

    def embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """取给定文本集合的 content_emb（缺失/无嵌入的条目跳过，如实返回现存者）。"""
        wanted = {t for t in texts}
        return [emb for t, _k, emb in self._entries if t in wanted and emb is not None]

    def clear(self) -> None:
        self._entries.clear()

    def dump(self) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
        return list(self._entries)

    def load_entries(self, entries: list[tuple[str, np.ndarray, np.ndarray | None]]) -> None:
        self._entries.clear()
        self._entries.extend(entries[-self.capacity:])


class MEMModule(nn.Module, BaseArchModule):
    """多尺度具身记忆：短时特征历史 + 长时文本库 -> 记忆 token。

    effective 行为由 ``active`` 开关决定（server CLI 注入）：
    - active=False：attach 后 embed_prefix 仍原样返回（零效应基础设施）；
    - active=True 且有可记忆内容：追加 num_tokens 个记忆 token 进 prefix。
    """

    name = "mem"

    def __init__(self, d_model: int = 2048, history_len: int = 4, num_tokens: int = 4,
                 bank_capacity: int = 16, num_heads: int = 8,
                 encode_text=None, retrieve_k: int = 3):
        nn.Module.__init__(self)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 不能被 num_heads={num_heads} 整除")
        self.d_model = int(d_model)
        self.history_len = int(history_len)
        self.num_tokens = int(num_tokens)
        self.retrieve_k = int(retrieve_k)
        self.active = False
        self.encode_text = encode_text  # Callable[[str], np.ndarray(d_model,)] | None
        self.bank = TextMemoryBank(bank_capacity)
        # episode 内短时记忆（连接/session 级，reset_episode 清空）
        self._history: deque = deque(maxlen=self.history_len)
        self._original_embed_prefix = None
        self._attached_model = None
        self._shadowed_instance_attr = False
        self.last_stats: dict = {}

        self.mem_queries = nn.Parameter(0.02 * torch.randn(self.num_tokens, self.d_model))
        self.short_attn = nn.MultiheadAttention(self.d_model, num_heads, batch_first=True)
        self.long_attn = nn.MultiheadAttention(self.d_model, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(self.d_model)

    # ---------------------------------------------------------------- 记忆状态

    def reset_episode(self) -> None:
        """episode 边界：清短时历史；长时文本库保留（跨 episode 语义）。"""
        self._history.clear()
        self.last_stats = {}

    def push_history(self, feature: torch.Tensor) -> None:
        """推入一步已计算的观测特征（vlm_prefix_mean，2048D，不含 robot state）。"""
        self._history.append(feature.detach().to(torch.float32))

    @property
    def history_len_current(self) -> int:
        return len(self._history)

    def write_memory(self, text: str, context_key: np.ndarray) -> None:
        """写一条长时文本记忆；content_emb 经 encode_text 回调即时计算（一次性）。"""
        emb = None
        if self.encode_text is not None:
            emb = np.asarray(self.encode_text(text), dtype=np.float32).reshape(-1)
        self.bank.write(text, context_key, emb)

    # ---------------------------------------------------------------- 记忆 token

    def compute_tokens(self) -> torch.Tensor | None:
        """(1, num_tokens, d_model) 记忆 token；无可记忆内容时返回 None。

        无内容（history 与 bank 皆空）时不注入——"没有记住任何东西"不该产生
        伪装的记忆 token，也保证 active-but-empty 的 bit 级零效应。
        """
        device = self.mem_queries.device
        history = list(self._history)
        if not history and len(self.bank) == 0:
            self.last_stats = {"injected": False, "reason": "empty_memory"}
            return None

        queries = self.mem_queries.unsqueeze(0)  # (1, M, D)
        ctx = queries
        used_short = False
        if history:
            hist = torch.stack(history).unsqueeze(0).to(device)  # (1, K, D)
            short, _ = self.short_attn(queries, hist, hist, need_weights=False)
            ctx = ctx + short
            used_short = True

        query_vec = history[-1].cpu().numpy() if history else np.zeros(self.d_model, np.float32)
        retrieved = self.bank.retrieve(query_vec, k=min(self.retrieve_k, len(self.bank)))
        text_embs = self.bank.embeddings([t for t, _s in retrieved])
        used_long = False
        if text_embs:
            texts_t = torch.from_numpy(np.stack(text_embs)).unsqueeze(0).to(device)
            long_ctx, _ = self.long_attn(queries, texts_t, texts_t, need_weights=False)
            ctx = ctx + long_ctx
            used_long = True

        tokens = self.out_norm(ctx)  # (1, M, D)
        with torch.no_grad():
            t = tokens.detach().float()
            self.last_stats = {
                "injected": True,
                "num_tokens": int(tokens.shape[1]),
                "history_len": len(history),
                "bank_size": len(self.bank),
                "retrieved_texts": len(retrieved),
                "used_short": used_short,
                "used_long": used_long,
                "token_l2_norm": float(t.norm()),
                "token_mean": float(t.mean()),
                "token_std": float(t.std()) if t.numel() > 1 else 0.0,
                "token_nan_count": int(torch.isnan(t).sum()),
                "token_inf_count": int(torch.isinf(t).sum()),
            }
        return tokens

    # ---------------------------------------------------------------- 生命周期（BaseArchModule）

    def attach(self, model) -> None:
        """包裹 model.embed_prefix（π0.5 系接口：sample_actions/训练 forward 都经
        self.embed_prefix 属性查找，实例级遮蔽生效）。"""
        if self._original_embed_prefix is not None:
            return
        if not callable(getattr(model, "embed_prefix", None)):
            raise ValueError("MEM.attach 需要模型暴露 embed_prefix（π0.5 系接口）")
        # 记录原值来自实例 __dict__（可能是别人的包裹）还是类方法，
        # detach 按来源恢复：类方法 -> del 实例遮蔽；实例值 -> 原样写回。
        self._shadowed_instance_attr = "embed_prefix" in model.__dict__
        original = model.embed_prefix
        mem = self

        def wrapped_embed_prefix(images, img_masks, tokens, masks):
            embs, pad_masks, att_masks = original(images, img_masks, tokens, masks)
            if not mem.active:
                return embs, pad_masks, att_masks
            mem_tokens = mem.compute_tokens()
            if mem_tokens is None:
                return embs, pad_masks, att_masks
            batch, device = pad_masks.shape[0], pad_masks.device
            m = mem_tokens.shape[1]
            embs = torch.cat([embs, mem_tokens.to(device=device, dtype=embs.dtype)], dim=1)
            pad_masks = torch.cat(
                [pad_masks, torch.ones(batch, m, dtype=pad_masks.dtype, device=device)], dim=1
            )
            att_masks = torch.cat(
                [att_masks, torch.zeros(batch, m, dtype=att_masks.dtype, device=device)], dim=1
            )
            return embs, pad_masks, att_masks

        model.embed_prefix = wrapped_embed_prefix
        self._original_embed_prefix = original
        # object.__setattr__ 绕过 nn.Module 的子模块注册：被挂载的 pi0.5 不得
        # 泄漏进 state_dict()/parameters()（否则 save 会带出整个 checkpoint）
        object.__setattr__(self, "_attached_model", model)

    def _restore_embed_prefix(self) -> None:
        """把 embed_prefix 恢复到 attach 前的解析状态（类方法或原实例值）。"""
        model = self._attached_model
        if self._shadowed_instance_attr:
            model.__dict__["embed_prefix"] = self._original_embed_prefix
        elif "embed_prefix" in model.__dict__:
            del model.__dict__["embed_prefix"]  # 撤销实例遮蔽，回落类方法

    def detach(self) -> None:
        """完整撤销 attach（A/B 实验正确性依赖）；记忆数据（history/bank）保留。"""
        if self._original_embed_prefix is not None:
            self._restore_embed_prefix()
            self._original_embed_prefix = None
        self._attached_model = None

    # ---------------------------------------------------------------- 存取

    def save(self, path) -> dict:
        """保存参数 + 文本库（不触碰 π0.5 checkpoint）。"""
        payload = {
            "state_dict": self.state_dict(),
            "bank": [(t, k, e) for t, k, e in self.bank.dump()],
            "config": {
                "d_model": self.d_model, "history_len": self.history_len,
                "num_tokens": self.num_tokens, "bank_capacity": self.bank.capacity,
            },
        }
        torch.save(payload, path)
        return payload["config"]

    def load(self, path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(payload["state_dict"])
        self.bank.load_entries(payload["bank"])
