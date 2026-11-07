"""
tapt.py (README_v3 13.5)
========================
TAPT（task-adaptive pre-training，Gururangan et al. 2020）：在一个 run 的无标签池上用 MLM 继续预训练
detector 的 encoder，再由 train_detector.py 的方法 AT 按 A 的设置微调。只用池中的文本，不用任何标签。

- 起点是 models.nli_model（A 也从它初始化，两者才可比），用与 detector 相同的 AutoModel 加载。
- DeBERTa-v3 的官方权重用 RTD 预训练，没有 MLM 头，所以新建一个：dense → GELU → LayerNorm →
  与输入词向量共享的解码矩阵 + bias（BERT / DeBERTa 的标准结构）。只在被遮盖的位置算 logits。
- 输入与 detector 完全相同（ClaimEvidenceDataset：claim 为第一段、证据为第二段，max_length，动态补齐）。
- 遮盖：每个非特殊 token 以 mask_prob 被选中，其中 80% 换成 [MASK]、10% 换成随机 token、10% 不变。
- 两个阶段：
    1. 冻结 encoder（含词向量），只训练新的 MLM 头 head_steps 步：随机初始化的头一开始梯度很大，
       直接全部一起训练会冲坏 NLI 权重；
    2. 全部参数一起训练 epochs 个 epoch，学习率 warmup_steps 步线性升到 lr 后保持不变。学习率不衰减，
       所以第 k 个 epoch 结束时的快照就等价于"只训练 k 个 epoch"，可行性测试可以一次比较多个步数。
- 每个 epoch 结束时在验证集的文本上算 MLM loss（固定的遮盖，不用标签）。--save_epochs 中的 epoch
  把 encoder 存成 save_pretrained 目录（<out_dir>/epoch<k>/），附 tapt_info.json。

这些设置不进配置文件（否则配置指纹改变），默认值见 common/paths.py 的 TAPT_SETTINGS；
可行性测试（tools/tapt_probe.sh）用命令行参数逐个指定。

Usage:
    python training/tapt.py --config configs/config_fc.yaml --run_dir runs_fc/r0.10_s42 --seed 42 \
        --lr 1e-5 --mask_prob 0.15 --epochs 30 --save_epochs 3,10,30 --out_dir /tmp/tapt_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ClaimEvidenceDataset, DynamicPaddingCollator  # noqa: E402
from common.paths import TAPT_SETTINGS, RunPaths, cfg_path, labeled_split_path, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tapt")
for noisy_name in ("httpx", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(noisy_name).setLevel(logging.WARNING)

# 不在可行性测试中变化的设置（README_v3 13.5）
FIXED = {
    "batch_size": 32,
    "head_steps": 200,
    "head_lr": 1e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}


class MLMHead(nn.Module):
    """dense → GELU → LayerNorm → tied decoder (the encoder's word embeddings) + bias."""

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float, init_std: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.dense.weight, std=init_std)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(self.norm(F.gelu(self.dense(hidden))), word_embeddings, self.bias)


def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: torch.Tensor, mask_id: int,
                n_random: int, mask_prob: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking (80% [MASK] / 10% random / 10% unchanged) over non-special, non-padding tokens."""
    labels = input_ids.clone()
    prob = torch.full(input_ids.shape, float(mask_prob))
    prob.masked_fill_(torch.isin(input_ids, special_ids) | (attention_mask == 0), 0.0)
    selected = torch.bernoulli(prob, generator=gen).bool()
    labels[~selected] = -100
    masked = input_ids.clone()
    to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=gen).bool() & selected
    masked[to_mask] = mask_id
    to_random = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=gen).bool() & selected & ~to_mask
    masked[to_random] = torch.randint(n_random, input_ids.shape, generator=gen)[to_random]
    return masked, labels


def mlm_loss(encoder, head: MLMHead, input_ids, attention_mask, labels, device, use_amp: bool) -> torch.Tensor:
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        selected = labels != -100
        logits = head(hidden[selected], encoder.embeddings.word_embeddings.weight)
    return F.cross_entropy(logits.float(), labels[selected])


@torch.no_grad()
def dev_loss(encoder, head, dev_batches, device, use_amp: bool) -> float:
    encoder.eval()
    head.eval()
    total, n = 0.0, 0
    for input_ids, attention_mask, labels in dev_batches:
        k = int((labels != -100).sum())
        total += float(mlm_loss(encoder, head, input_ids, attention_mask, labels, device, use_amp)) * k
        n += k
    encoder.train()
    head.train()
    return total / max(n, 1)


def parse_args() -> argparse.Namespace:
    defaults = TAPT_SETTINGS or {}
    p = argparse.ArgumentParser(description="TAPT: MLM continued pre-training on the run's unlabeled pool (README_v3 13.5).")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=defaults.get("lr"))
    p.add_argument("--mask_prob", type=float, default=defaults.get("mask_prob"))
    p.add_argument("--epochs", type=int, default=defaults.get("epochs"))
    p.add_argument("--save_epochs", type=str, default=None, help="Comma-separated epochs to save (default: the last one)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to save the encoder (default: <run>/checkpoints/tapt/encoder, the input of method AT)")
    args = p.parse_args()
    missing = [k for k in ("lr", "mask_prob", "epochs") if getattr(args, k) is None]
    if missing:
        p.error(f"--{', --'.join(missing)} required: TAPT_SETTINGS in common/paths.py is not set yet (README_v3 13.5)")
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir)
    train_cfg = cfg.get("training", {})
    settings = {"lr": args.lr, "mask_prob": args.mask_prob, "epochs": args.epochs, **FIXED}
    save_epochs = sorted({int(e) for e in args.save_epochs.split(",")}) if args.save_epochs else [args.epochs]
    if not save_epochs or save_epochs[0] < 1 or save_epochs[-1] > args.epochs:
        raise SystemExit(f"--save_epochs must lie in [1, {args.epochs}], got {save_epochs}")
    default_out = args.out_dir is None
    out_root = paths.tapt_encoder if default_out else Path(args.out_dir)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and bool(train_cfg.get("use_bf16", False))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("use_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("use_tf32", True))

    cache_dir = str(cfg_path(cfg, "model_cache_dir"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["deberta_base"], cache_dir=cache_dir)
    start_model = cfg["models"]["nli_model"]
    encoder = AutoModel.from_pretrained(start_model, dtype=torch.float32, cache_dir=cache_dir).to(device)
    ecfg = encoder.config
    head = MLMHead(ecfg.hidden_size, ecfg.vocab_size, ecfg.layer_norm_eps, ecfg.initializer_range).to(device)

    max_length = int(train_cfg.get("max_length", 512))
    ds_kwargs = {"max_length": max_length, "dynamic_padding": True, "require_label": False}
    pool = ClaimEvidenceDataset(str(paths.unlabeled_pool), tokenizer, **ds_kwargs)
    dev = ClaimEvidenceDataset(str(labeled_split_path(cfg, "dev")), tokenizer, **ds_kwargs)
    collate = DynamicPaddingCollator(tokenizer.pad_token_id)
    gen_order = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(pool, batch_size=FIXED["batch_size"], shuffle=True, drop_last=True, generator=gen_order,
                        collate_fn=collate, num_workers=int(train_cfg.get("num_workers", 0)))
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids)))
    n_random = len(tokenizer)
    gen_mask = torch.Generator().manual_seed(args.seed + 1)
    gen_dev = torch.Generator().manual_seed(0)  # 验证集的遮盖固定，各设置、各 seed 可比
    dev_batches = []
    for batch in DataLoader(dev, batch_size=FIXED["batch_size"], shuffle=False, collate_fn=collate):
        ids, labels = mask_tokens(batch["input_ids"], batch["attention_mask"], special_ids, tokenizer.mask_token_id,
                                  n_random, args.mask_prob, gen_dev)
        dev_batches.append((ids, batch["attention_mask"], labels))

    steps_per_epoch = len(loader)
    log.info("run=%s seed=%d start=%s | pool %d pairs (%d steps/epoch), dev %d pairs | settings %s | save epochs %s -> %s",
             paths.root.name, args.seed, start_model, len(pool), steps_per_epoch, len(dev), settings, save_epochs, out_root)

    def batches():
        while True:
            for batch in loader:
                ids, labels = mask_tokens(batch["input_ids"], batch["attention_mask"], special_ids,
                                          tokenizer.mask_token_id, n_random, args.mask_prob, gen_mask)
                yield ids, batch["attention_mask"], labels

    stream = batches()
    history = [{"phase": "start", "epoch": 0, "step": 0, "dev_mlm_loss": dev_loss(encoder, head, dev_batches, device, use_amp)}]
    log.info("dev MLM loss with the untrained head: %.4f", history[-1]["dev_mlm_loss"])
    t0 = time.time()

    # ---- phase 1: head only ----
    encoder.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=FIXED["head_lr"], weight_decay=FIXED["weight_decay"])
    run_loss = 0.0
    for step in tqdm(range(1, FIXED["head_steps"] + 1), desc="head", unit="step", leave=False, dynamic_ncols=True):
        loss = mlm_loss(encoder, head, *next(stream), device, use_amp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), FIXED["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
    history.append({"phase": "head", "epoch": 0, "step": FIXED["head_steps"], "train_mlm_loss": run_loss / FIXED["head_steps"],
                    "dev_mlm_loss": dev_loss(encoder, head, dev_batches, device, use_amp)})
    log.info("[head] %d steps: train MLM loss %.4f, dev %.4f (%.1f min)", FIXED["head_steps"],
             history[-1]["train_mlm_loss"], history[-1]["dev_mlm_loss"], (time.time() - t0) / 60)

    # ---- phase 2: everything, constant LR after warmup ----
    encoder.requires_grad_(True)
    params = [*encoder.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=FIXED["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / FIXED["warmup_steps"]))
    step = 0
    for epoch in range(1, args.epochs + 1):
        run_loss = 0.0
        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}", unit="step", leave=False, dynamic_ncols=True):
            loss = mlm_loss(encoder, head, *next(stream), device, use_amp)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite MLM loss at epoch {epoch} step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, FIXED["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            step += 1
        row = {"phase": "full", "epoch": epoch, "step": step, "lr": sched.get_last_lr()[0],
               "train_mlm_loss": run_loss / steps_per_epoch, "dev_mlm_loss": dev_loss(encoder, head, dev_batches, device, use_amp),
               "minutes": round((time.time() - t0) / 60, 2)}
        history.append(row)
        log.info("[epoch %d/%d] train MLM loss %.4f, dev %.4f (%.1f min)", epoch, args.epochs,
                 row["train_mlm_loss"], row["dev_mlm_loss"], row["minutes"])
        if not math.isfinite(row["dev_mlm_loss"]):
            raise SystemExit("non-finite dev MLM loss")
        if epoch in save_epochs:
            target = out_root if default_out else out_root / f"epoch{epoch}"
            target.mkdir(parents=True, exist_ok=True)
            encoder.save_pretrained(target)
            info = {"run": paths.root.name, "seed": args.seed, "start_model": start_model, "settings": settings,
                    "epoch": epoch, "full_steps": step, "pool_size": len(pool), "history": history}
            with open(target / "tapt_info.json", "w", encoding="utf-8") as fh:
                json.dump(info, fh, indent=2, ensure_ascii=False)
            log.info("saved encoder after epoch %d -> %s", epoch, target)

    if default_out:  # 方法 AT 的 TAPT：训练记录放进 outputs/（随结果提交），权重在 checkpoints/（跑完删除）
        paths.tapt_outputs.mkdir(parents=True, exist_ok=True)
        with open(paths.tapt_outputs / "tapt_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"run": paths.root.name, "seed": args.seed, "start_model": start_model, "settings": settings,
                       "pool_size": len(pool), "history": history}, fh, indent=2, ensure_ascii=False)
    log.info("TAPT done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
