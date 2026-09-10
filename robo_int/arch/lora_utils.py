"""LoRA 挂载工具（peft 惰性导入）。

统一入口：所有基模（π0.5/GR00T/Mock）的 LoRA 微调都走这里，
保证 target_modules/rank/alpha 的配置语义跨基模一致。
"""

from __future__ import annotations


def attach_lora(model, target_modules: list[str], r: int = 16, alpha: int = 32):
    """给基模挂 LoRA。dropout 由 peft 自身的 lora_dropout 承接。

    model: torch.nn.Module（基模或其子模块，如 VLM backbone）
    返回 peft 的 PeftModel。
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("LoRA 需要 peft：pip install peft") from e

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        # dropout 选项统一走 infra 配置注入（见 RegularizationConfig.dropout）
        lora_dropout=0.0,
    )
    return get_peft_model(model, config)
