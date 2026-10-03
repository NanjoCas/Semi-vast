# 方向一致性 c 的设计分析（ratio 0.1，3 个 seed）：对 train 全部句对计算 NLI 三类概率（缓存到 analysis/nli_train_probs.npz），
# 然后离线比较多种伪标签过滤 / 融合方案的规模、准确率、各类精度和"平衡后准确率"（平衡采样器下 detector 实际看到的期望准确率）。
# 运行：python v2/analysis/c_design_probe.py（任意目录）。第一次运行需要 GPU 约几分钟，之后读缓存。
"""Offline comparison of direction-consistency (c) designs for pseudo-label filtering."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
V2 = Path("/root/autodl-tmp/Semi-vast/v2"); sys.path.insert(0, str(V2))
from common.data_utils import direction_consistency

CACHE = V2 / "analysis/nli_train_probs.npz"
SEEDS = (42, 43, 44)
SRC = ("climatecheck", "climate_fever", "scifact")
LAB = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
ld = lambda p: [json.loads(l) for l in open(p)]
train = ld(V2 / "processed/sci/labeled/train.jsonl")

# ---- NLI probabilities in pseudo-label order (SUP=entail, REF=contradict, NEI=neutral), same premise as extractor.py ----
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True); ids, probs = list(z["ids"]), z["probs"]
else:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "cross-encoder/nli-deberta-v3-large"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(V2.parent / "model_cache"))
    model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=str(V2.parent / "model_cache")).cuda().eval()
    l2i = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    order_cols = [l2i["entailment"], l2i["contradiction"], l2i["neutral"]]
    prem = [" ".join((r.get("evidence") or [])[:5]) for r in train]; hyp = [r["claim"] for r in train]
    ids = [str(r["id"]) for r in train]; probs = np.zeros((len(train), 3), dtype=np.float32)
    order = np.argsort([len(p) for p in prem])
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([prem[i] for i in idx], [hyp[i] for i in idx], truncation=True, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            probs[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, order_cols].cpu().numpy()
    np.savez(CACHE, ids=np.array(ids), probs=probs)
nli = dict(zip(ids, probs))


def load_seed(seed):
    rd = V2 / f"runs_sci/r0.10_s{seed}"
    gold = {r["id"]: LAB.index(r["label"]) for r in ld(rd / "data/unlabeled_gold.jsonl")}
    P = ld(rd / "pseudo/pseudo_pool.jsonl")
    d = dict(
        g=np.array([gold[p["id"]] for p in P]), pl=np.array([p["pseudo_label"] for p in P]),
        conf=np.array([p["confidence"] for p in P]), pe=np.array([p["probs"] for p in P]),
        ls=np.array([p["logic_score"] for p in P]), src=np.array([p["source"] for p in P]),
        pn=np.array([nli[str(p["id"])] for p in P]),
    )
    d["c"] = np.array([direction_consistency(a, b) for a, b in zip(d["pl"], d["ls"])])
    d["cp"] = d["pn"][np.arange(len(P)), d["pl"]]          # NLI 对伪标签类别的概率
    return d


def describe(name, keep, label, g):
    k = keep.sum()
    if k == 0:
        return f"| {name} | 0 | — | — | — |"
    ok = label[keep] == g[keep]
    prec = [ok[label[keep] == c].mean() if (label[keep] == c).any() else np.nan for c in range(3)]
    cnt = [int((label[keep] == c).sum()) for c in range(3)]
    return (f"| {name} | {k} | {ok.mean():.3f} | {np.nanmean(prec):.3f} | "
            + " / ".join(f"{n}:{p:.2f}" if n else "0" for n, p in zip(cnt, prec)) + " |")


rows = {}
for seed in SEEDS:
    d = load_seed(seed); g, pl, conf, c, cp, pn, pe, src = (d[k] for k in ("g", "pl", "conf", "c", "cp", "pn", "pe", "src"))
    # sanity: stored LogicScore == P(entail) - P(contradict) recomputed here
    r = np.corrcoef(d["ls"], pn[:, 0] - pn[:, 1])[0, 1]
    ok = pl == g
    print(f"\n######## seed {seed}  (n={len(g)}, 整池准确率 {ok.mean():.3f}, LS 与重算值相关系数 {r:.4f})")

    print("\n各伪标签类别内部，信号区分对错的 AUROC（c 只在类内比较才不受类别构成影响）：")
    print("| 伪标签类别 | 条数 | 精度 | 置信度 | c | NLI 对该类的概率 cp |")
    print("|---|---|---|---|---|---|")
    for k in range(3):
        m = pl == k
        a = lambda s: roc_auc_score(ok[m], s[m]) if 0 < ok[m].mean() < 1 else np.nan
        print(f"| {LAB[k]} | {m.sum()} | {ok[m].mean():.3f} | {a(conf):.3f} | {a(c):.3f} | {a(cp):.3f} |")

    b = conf >= 0.7
    L = b & (c > 0.3)
    nL = L.sum()
    topn = np.zeros_like(b); topn[np.argsort(-conf)[:nL]] = True
    # NEI 的 c 改为用 NLI 的中立概率：pn[:,2] > t
    c_nei_p = np.where(pl == 2, pn[:, 2], c)
    # 分类别分位数：每类内按 c 取前一半，避免 NEI 自动通过
    half = np.zeros_like(b)
    for k in range(3):
        idx = np.where(b & (pl == k))[0]
        if len(idx):
            half[idx[np.argsort(-c[idx])[: max(1, len(idx) // 2)]]] = True
    # 融合：p = (1-α)·p_extractor + α·p_nli，按融合后的 argmax 作为伪标签、融合后的最大概率作为置信度
    fused = {}
    for alpha in (0.3, 0.5):
        pf = (1 - alpha) * pe + alpha * pn
        fused[alpha] = (pf.argmax(1), pf.max(1))
    # 按来源设置 α：SciFact 上 NLI 更准，ClimateCheck 上 extractor 更准
    a_src = np.select([src == "scifact", src == "climate_fever"], [0.6, 0.4], 0.2)[:, None]
    pf_src = (1 - a_src) * pe + a_src * pn
    fl_src, fc_src = pf_src.argmax(1), pf_src.max(1)

    print("\n过滤 / 融合方案（\"平衡后准确率\"= 各伪标签类别精度的平均，平衡采样器下 detector 实际看到的期望准确率）：")
    print("| 方案 | 条数 | 准确率 | 平衡后准确率 | 各类 条数:精度（SUP / REF / NEI） |")
    print("|---|---|---|---|---|")
    lines = [
        ("B：置信度 ≥ 0.7", b, pl),
        ("L（现行）：B 且 c > 0.3", L, pl),
        (f"B 取置信度前 {nL} 条（与 L 同规模）", topn, pl),
        ("L′：NEI 改用 P_nli(中立) > 0.5", b & (c_nei_p > np.where(pl == 2, 0.5, 0.3)), pl),
        ("L″：只对 SUP/REF 用 c > 0.3，NEI 按 P_nli(中立) > 0.7", b & np.where(pl == 2, pn[:, 2] > 0.7, c > 0.3), pl),
        ("Q：B 且每类内按 c 取前 50%", half, pl),
        ("A1：B 且 NLI argmax 与伪标签一致", b & (pn.argmax(1) == pl), pl),
        ("F0.3：融合 α=0.3，融合置信度 ≥ 0.7", fused[0.3][1] >= 0.7, fused[0.3][0]),
        ("F0.5：融合 α=0.5，融合置信度 ≥ 0.6", fused[0.5][1] >= 0.6, fused[0.5][0]),
        ("Fsrc：按来源 α（SF 0.6 / CF 0.4 / CC 0.2），融合置信度 ≥ 0.7", fc_src >= 0.7, fl_src),
    ]
    for name, keep, lab in lines:
        line = describe(name, keep, lab, g); print(line)
        rows.setdefault(name, []).append(line)
    # L 的 REFUTES 错误来自哪类金标签（NLI 对无关 claim 也判矛盾）
    m = L & (pl == 1) & ~ok
    print(f"\nL 中 REFUTES 伪标签的错误 {m.sum()} 条，金标签分布 SUP/REF/NEI = {[int((g[m] == k).sum()) for k in range(3)]}；"
          f"按来源 {dict((s, int((src[m] == s).sum())) for s in SRC)}")
    m = b & (pl == 2)
    print(f"B 中 NEI 伪标签 {m.sum()} 条，c > 0.3 的通过率 {np.mean(c[m] > 0.3):.2f}；|LS| < 0.01 的比例 {np.mean(np.abs(d['ls'][m]) < 0.01):.2f}")
