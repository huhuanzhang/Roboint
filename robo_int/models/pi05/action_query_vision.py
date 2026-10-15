"""Action-query Vision Attention Gate（B 类 model-internal 模块）。

审计结论（modeling_pi05.py 精读）：
- π0.5 的 Action Expert **本来就**通过 prefix KV cache attend 全部 968 个 prefix
  token（3×256 视觉 + 200 语言）——这是 π0.5 的基座机制，本模块不重复实现它；
- 本模块做的是这一通路上的**可学习门控干预**：按层放大/衰减视觉 token 在
  expert 交叉注意力中的 value 贡献（V-side 调制，改的是「视觉信息注入多少」，
  不改 softmax 分布；K-side 变体可另行实现）。

接入点（唯一 junction）：`paligemma_with_expert.forward` 的 prefix-only 调用
（inputs_embeds=[prefix, None], use_cache=True）——返回的 past_key_values 即
expert 全部 denoise 步要 attend 的 KV。此处按层变换视觉切片，denoise 循环
每步 deepcopy 后使用，无需逐层逐步打补丁。

零效应保证：所有 gate==0 时直接返回原 KV 对象（无任何算术）→ bit 级 parity。

prefix token 布局（embed_prefix 拼接顺序）：[cam_0 的 256 tokens, cam_1 的 256,
cam_2 的 256, 语言 200]；相机顺序 = pi05.config.image_features 顺序。
本模块**不硬编码**相机切片：由 server 按 image_features 注册 per-camera 统计；
若 256×n_cams + lang_len != 实测 seq_len 则拒绝切相机（防错误映射）。
"""

from __future__ import annotations

import torch
from torch import nn


class ActionQueryVisionGate(nn.Module):
    """按层门控 Action Expert 对视觉 prefix KV 的 value 贡献。

    effective_gate(layer) = layer_gates[layer] + fixed_gate
    - layer_gates：可学习（零初始化），训练时唯一可更新参数；
    - fixed_gate：推理期固定偏置（CLI 注入，如 0.05 做 functional 激活）。

    V'_vision = (1 + g) · V_vision（每层同一切片）；K 不动。
    """

    name = "action_query_vision"

    def __init__(self, paligemma_with_expert, num_layers: int, vision_tokens: int,
                 fixed_gate: float = 0.0, camera_names: list[str] | None = None):
        super().__init__()
        self.pwe = paligemma_with_expert
        self.num_layers = num_layers
        self.vision_tokens = vision_tokens
        self.fixed_gate = float(fixed_gate)
        self.camera_names = list(camera_names or [])
        self.layer_gates = nn.Parameter(torch.zeros(num_layers))
        self._original_forward = None
        # 每次 prefix 变换的统计（monitor 读取）
        self.last_stats: dict = {}
        self.hook_call_count = 0

    # ---------------------------------------------------------------- 生命周期

    def attach(self) -> None:
        """包裹 pwe.forward（molmoact2 直接调 .forward()，hook 机制不适用）。"""
        if self._original_forward is not None:
            return
        pwe = self.pwe
        original = pwe.forward
        gate = self

        def wrapped(attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, use_cache=None, adarms_cond=None):
            outputs = original(
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, adarms_cond=adarms_cond,
            )
            # prefix-only pass：inputs_embeds=[prefix, None] 且要 KV cache
            if (inputs_embeds is not None and len(inputs_embeds) == 2
                    and inputs_embeds[1] is None and use_cache):
                outputs = (outputs[0], gate._transform_kv(outputs[1]))
            return outputs

        pwe.forward = wrapped
        self._original_forward = original

    def detach(self) -> None:
        if self._original_forward is not None:
            self.pwe.forward = self._original_forward
            self._original_forward = None

    # ---------------------------------------------------------------- 变换

    def effective_gates(self) -> torch.Tensor:
        """报告用（detach）的门控值。"""
        return self.layer_gates.detach() + self.fixed_gate

    def _transform_kv(self, pkv):
        """变换视觉切片的 V（K 不动）。gate 全 0 时原样返回（无算术 -> bit 级 parity）。

        KV 容器双兼容（本机 transformers 实测，DynamicCache.__iter__ yield
        (keys, values, sliding) 三元组，直接解包 (k, v) 会 ValueError）：
        - DynamicCache：逐层读写 ``layer.values``，原地更新后返回同一对象；
        - legacy tuple[(k, v), ...]：构造新 tuple 返回。

        门控用**不 detach** 的 layer_gates 参与运算：推理（no_grad）下无额外开销，
        手动复现采样路径时可对 layer_gates 反传（functional 训练测试用）。
        """
        gates = self.layer_gates + self.fixed_gate
        if bool((gates == 0).all()):
            self.hook_call_count += 1
            self.last_stats = {"transformed": False}
            return pkv
        stats_v_mass: dict = {}
        kv_shapes = None
        layers = getattr(pkv, "layers", None)
        if layers is not None:  # transformers DynamicCache
            for idx, layer in enumerate(layers):
                k, v = layer.keys, layer.values
                if kv_shapes is None:
                    kv_shapes = {"k": tuple(k.shape), "v": tuple(v.shape), "layers": len(layers)}
                g = gates[min(idx, len(gates) - 1)]
                if float(g.detach()) == 0.0:
                    continue
                v2 = v.clone()
                v2[:, :, : self.vision_tokens, :] = v[:, :, : self.vision_tokens, :] * (1.0 + g)
                layer.values = v2
                if not stats_v_mass:
                    with torch.no_grad():
                        per_cam = self._camera_mass(v)
                        if per_cam is not None:
                            stats_v_mass = per_cam
            self.hook_call_count += 1
            self.last_stats = {
                "transformed": True,
                "container": "dynamic_cache",
                "kv_shapes": kv_shapes,
                **({"camera_v_mass": stats_v_mass} if stats_v_mass else {}),
            }
            return pkv
        new_kv = []
        for layer, (k, v) in enumerate(pkv):  # legacy tuple[(k, v), ...]
            if kv_shapes is None:
                kv_shapes = {"k": tuple(k.shape), "v": tuple(v.shape), "layers": len(pkv)}
            g = gates[min(layer, len(gates) - 1)]
            if float(g.detach()) == 0.0:
                new_kv.append((k, v))
                continue
            v2 = v.clone()
            v2[:, :, : self.vision_tokens, :] = v[:, :, : self.vision_tokens, :] * (1.0 + g)
            new_kv.append((k, v2))
            if not stats_v_mass:
                with torch.no_grad():
                    per_cam = self._camera_mass(v)
                    if per_cam is not None:
                        stats_v_mass = per_cam
        self.hook_call_count += 1
        self.last_stats = {
            "transformed": True,
            "container": "legacy_tuple",
            "kv_shapes": kv_shapes,
            **({"camera_v_mass": stats_v_mass} if stats_v_mass else {}),
        }
        return tuple(new_kv)

    def _camera_mass(self, v: torch.Tensor) -> dict[str, float] | None:
        """按 image_features 顺序切视觉 KV，报每相机 mean|V|；语言段单独一段。

        仅当 vision_tokens 恰好整除相机数时才切（否则映射不可靠，返回 None，
        由上层如实报告 camera token mapping unavailable，绝不硬切猜测）。
        """
        n = len(self.camera_names)
        if n == 0 or self.vision_tokens % n != 0:
            return None
        per = self.vision_tokens // n
        with torch.no_grad():
            mass = {}
            for i, cam in enumerate(self.camera_names):
                mass[cam] = float(v[:, :, i * per:(i + 1) * per, :].abs().float().mean())
            mass["language"] = float(v[:, :, self.vision_tokens:, :].abs().float().mean())
        return mass
