"""
DeBERTa encoder 的运行方式：选择性梯度检查点（README_v3 第 11 节）与 torch.compile（第 12 节）。

梯度检查点只决定激活是保存还是在反向时重算，不改变计算结果（transformers 5 默认
use_reentrant=False，并保留随机数状态，重算时的 dropout mask 与前向相同）。
显存够时让前 skip_layers 层不做检查点，可以省掉这些层的重算。

torch.compile 会改变数值（算子融合），所以它属于实验协议，在配置指纹中。
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# 每层的编译版本：冻结 / 可训练 × 训练 / 评估 × 首次静态形状 / 之后的动态形状，实测共 9 个，
# 不随序列长度增加。默认上限 8 会让多出来的版本静默退回 eager，所以调高。
_RECOMPILE_LIMIT = 64


def enable_gradient_checkpointing(encoder, skip_layers: int = 0) -> int:
    """
    Checkpoint every layer of `encoder` (a HF DeBERTa model) except the first `skip_layers`.

    The first layers are skipped because they include the frozen ones, whose activations are
    cheaper to keep. Returns the number of layers that still use checkpointing.
    """
    encoder.gradient_checkpointing_enable()
    layers = encoder.encoder.layer
    n_skip = max(0, min(int(skip_layers), len(layers)))
    for layer in layers[:n_skip]:
        if not hasattr(layer, "gradient_checkpointing"):
            raise RuntimeError(
                f"{type(layer).__name__} has no per-layer gradient_checkpointing flag in this "
                "transformers version; set the *gradient_checkpointing_skip_layers settings to 0"
            )
        layer.gradient_checkpointing = False
    return len(layers) - n_skip


def compile_encoder_layers(encoder, fallback_random: bool = True) -> int:
    """
    torch.compile every layer of `encoder` (dynamic shapes, for dynamic padding). Returns the layer count.

    fallback_random=True makes compiled dropout use the same RNG ops as eager PyTorch, so dropout
    masks match eager ones; with Inductor's own RNG the regularisation measurably weakened
    (README_v3 11.3).
    """
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    inductor_config.fallback_random = bool(fallback_random)
    dynamo_config.recompile_limit = max(int(dynamo_config.recompile_limit), _RECOMPILE_LIMIT)
    layers = encoder.encoder.layer
    for layer in layers:
        layer.compile(dynamic=True)
    return len(layers)


def uncompiled_frames() -> int:
    """Frames dynamo tried but did not compile, e.g. after hitting the recompile limit (should be 0)."""
    from torch._dynamo.utils import counters

    return int(counters["frames"]["total"] - counters["frames"]["ok"])
