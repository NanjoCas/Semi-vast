# 方向一（logic-aware teacher）的离线验证（README_v3 第 10.5、10.6 节）。不训练模型，不改配置。
#   第 1 部分：v2 seed 42–44 的伪标签池上，比较 extractor、NLI 零样本、两者融合作为 teacher 的伪标签质量（金标签只用于评估）。
#   第 2 部分：对照——不用伪标签，只在测试时把 detector 与 NLI 融合（seed 42 的 v3 试跑结果），看 NLI 本身能带来多少。
# 运行：cd v3 && python analysis/teacher_fusion_probe.py。第一次运行在 GPU 上计算 dev/test 的 NLI 概率（约几分钟），之后读缓存。
"""Offline check of an extractor + NLI fused teacher, and of test-time NLI fusion as a no-SSL control."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

V3 = Path(__file__).resolve().parent.parent
V2 = V3.parent / "v2"
SEEDS = (42, 43, 44)
SRC = ("climatecheck", "climate_fever", "scifact")
LAB = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
ALPHAS = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)
ld = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]


# ---- NLI 概率，按伪标签的类别顺序（SUP = 蕴含，REF = 矛盾，NEI = 中立）；premise / hypothesis 与 models/extractor.py 相同 ----
def nli_probs(split: str) -> dict:
    cache = V3 / f"analysis/nli_{split}_probs.npz"
    if not cache.exists() and split == "train":
        cache = V2 / "analysis/nli_train_probs.npz"            # v2 已经算好（train 与 v3 完全相同）
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(list(z["ids"]), z["probs"]))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = ld(V3 / f"processed/labeled/{split}.jsonl")
    name = "cross-encoder/nli-deberta-v3-large"
    kw = dict(cache_dir=str(V3.parent / "model_cache"))
    tok = AutoTokenizer.from_pretrained(name, **kw)
    model = AutoModelForSequenceClassification.from_pretrained(name, **kw).cuda().eval()
    l2i = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    cols = [l2i["entailment"], l2i["contradiction"], l2i["neutral"]]
    prem = [" ".join((r.get("evidence") or [])[:5]) for r in rows]
    hyp = [r["claim"] for r in rows]
    ids = [str(r["id"]) for r in rows]
    probs = np.zeros((len(rows), 3), dtype=np.float32)
    order = np.argsort([len(p) for p in prem])
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([prem[i] for i in idx], [hyp[i] for i in idx], truncation=True, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            probs[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, cols].cpu().numpy()
    np.savez(cache, ids=np.array(ids), probs=probs)
    return dict(zip(ids, probs))


def fuse(pe, pn, alpha, kind="lin"):
    """lin: (1-α)·p_e + α·p_n；geo: p_e^(1-α)·p_n^α 归一化。"""
    if kind == "geo":
        lp = (1 - alpha) * np.log(pe + 1e-8) + alpha * np.log(pn + 1e-8)
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)
    return (1 - alpha) * pe + alpha * pn


def quality(lab, g, keep=None):
    """准确率、平衡后准确率（各伪标签类别精度的平均 = 平衡采样器下 detector 看到的期望准确率）、对金标签的 macro-F1、各类条数。"""
    keep = np.ones(len(g), bool) if keep is None else keep
    l, gg = lab[keep], g[keep]
    ok = l == gg
    prec = [ok[l == c].mean() if (l == c).any() else np.nan for c in range(3)]
    return dict(n=int(keep.sum()), acc=ok.mean(), bal=np.nanmean(prec), f1=f1_score(gg, l, average="macro", labels=[0, 1, 2]),
                prec=prec, cnt=[int((l == c).sum()) for c in range(3)])


def top_frac(conf, frac=0.3):
    keep = np.zeros(len(conf), bool)
    keep[np.argsort(-conf, kind="stable")[: int(round(frac * len(conf)))]] = True
    return keep


def part1():
    nli = nli_probs("train")
    print("# 第 1 部分：teacher 的伪标签质量（v2 伪标签池，金标签只用于评估）\n")
    print("平衡后准确率 = 各伪标签类别精度的平均。A 在 seed 42 测试集上的参照：平衡后准确率（macro recall）0.508，"
          "ClimateCheck 0.496 / Climate-FEVER 0.579 / SciFact 0.531。\n")
    teachers = [("extractor", None, None)] + [(f"融合 lin α={a}", a, "lin") for a in (0.2, 0.3, 0.5, 0.7)] \
        + [("融合 geo α=0.5", 0.5, "geo"), ("NLI 零样本", 1.0, "lin")]
    summary = {}
    for seed in SEEDS:
        rd = V2 / f"runs_sci/r0.10_s{seed}"
        gold = {r["id"]: LAB.index(r["label"]) for r in ld(rd / "data/unlabeled_gold.jsonl")}
        P = ld(rd / "pseudo/pseudo_pool.jsonl")
        g = np.array([gold[p["id"]] for p in P])
        pe = np.array([p["probs"] for p in P], dtype=np.float64)
        pn = np.array([nli[str(p["id"])] for p in P], dtype=np.float64)
        src = np.array([p["source"] for p in P])
        assert (pe.argmax(1) == np.array([p["pseudo_label"] for p in P])).mean() > 0.99
        print(f"## seed {seed}（池 {len(g)} 条）\n")
        print("| teacher | 整池 准确率 / 平衡后 / F1 | ClimateCheck 准确率 / 平衡后 | Climate-FEVER | SciFact | 置信度前 30%：条数 / 准确率 / 平衡后 | 前 30% 各类 条数:精度 SUP / REF / NEI |")
        print("|---|---|---|---|---|---|---|")
        for name, a, kind in teachers:
            pf = pe if a is None else fuse(pe, pn, a, kind)
            lab, conf = pf.argmax(1), pf.max(1)
            q = quality(lab, g)
            per = [quality(lab, g, src == s) for s in SRC]
            t = quality(lab, g, top_frac(conf))
            cls = " / ".join(f"{n}:{p:.2f}" for n, p in zip(t["cnt"], t["prec"]))
            print(f"| {name} | {q['acc']:.3f} / {q['bal']:.3f} / {q['f1']:.3f} | "
                  + " | ".join(f"{x['acc']:.3f} / {x['bal']:.3f}" for x in per)
                  + f" | {t['n']} / {t['acc']:.3f} / {t['bal']:.3f} | {cls} |")
            summary.setdefault(name, []).append((q["acc"], q["bal"], t["acc"], t["bal"], *[x["acc"] for x in per]))
        print()
    print("## 3 个 seed 平均\n")
    print("| teacher | 整池准确率 | 整池平衡后 | 前 30% 准确率 | 前 30% 平衡后 | ClimateCheck 准确率 | Climate-FEVER | SciFact |")
    print("|---|---|---|---|---|---|---|---|")
    for name, v in summary.items():
        m = np.mean(v, 0)
        print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in m) + " |")
    print()


def part2():
    nli = nli_probs("test")
    test = {r["id"]: r for r in ld(V3 / "processed/labeled/test.jsonl")}
    run = V3 / "runs/r0.10_s42/outputs/detector"
    print("# 第 2 部分：对照——不用伪标签，测试时把 detector 与 NLI 融合（seed 42，v3 试跑）\n")
    print("α = 0 为 detector 本身，α = 1 为 NLI 零样本。注意：这里的 α 网格是在测试集上看的，只用于诊断，不能用来选 α。\n")
    print("| detector | α | 测试 macro-F1 | 来源平均 | ClimateCheck | Climate-FEVER | SciFact |")
    print("|---|---|---|---|---|---|---|")
    for tag in ("A", "A_t1042", "B", "B_t1042", "O"):
        f = run / tag / "test_predictions.jsonl"
        if not f.exists():
            continue
        R = ld(f)
        g = np.array([r["gold"] for r in R])
        pd_ = np.array([r["probs"] for r in R], dtype=np.float64)
        pn = np.array([nli[str(r["id"])] for r in R], dtype=np.float64)
        src = np.array([test[r["id"]]["source"] for r in R])
        for a in ALPHAS:
            if tag != "A" and a not in (0.0, 0.3, 0.5):
                continue
            lab = fuse(pd_, pn, a).argmax(1)
            per = [f1_score(g[src == s], lab[src == s], average="macro", labels=[0, 1, 2]) for s in SRC]
            print(f"| {tag} | {a} | {f1_score(g, lab, average='macro'):.4f} | {np.mean(per):.4f} | " + " | ".join(f"{x:.4f}" for x in per) + " |")
    print()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    part2()
    part1()
