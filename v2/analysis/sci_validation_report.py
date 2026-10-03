# 新数据集低成本验证的分析报告（runs_sci/r0.10_s42）：伪标签准确率、各信号 AUROC、B+L 过滤、detector 分来源 F1、配对 bootstrap、只看 claim 的基线。
# 运行：python v2/analysis/sci_validation_report.py [runs_sci/r0.10_s43]（任意目录；默认 runs_sci/r0.10_s42）。
"""Analysis of the low-cost validation run on the new datasets (runs_sci/r0.10_s42)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
V2 = Path("/root/autodl-tmp/Semi-vast/v2"); sys.path.insert(0, str(V2))
from evaluation.aggregate_results import load_predictions, paired_bootstrap

RUN = V2 / (sys.argv[1] if len(sys.argv) > 1 else "runs_sci/r0.10_s42")
L = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
ld = lambda p: [json.loads(l) for l in open(p)]
SRC = ("climatecheck", "climate_fever", "scifact")

ext = json.load(open(RUN / "outputs/extractor/extractor_train_metrics.json"))
best = max(ext, key=lambda h: h.get("val_macro_f1", -1)) if isinstance(ext, list) else ext
print("== extractor ==", {k: round(v, 4) for k, v in best.items() if isinstance(v, float) and "val" in k})

gold = {r["id"]: L.index(r["label"]) for r in ld(RUN / "data/unlabeled_gold.jsonl")}
P = ld(RUN / "pseudo/pseudo_pool.jsonl")
g = np.array([gold[p["id"]] for p in P]); src = np.array([p["source"] for p in P])
pl = np.array([p["pseudo_label"] for p in P]); conf = np.array([p["confidence"] for p in P])
ls = np.array([p["logic_score"] for p in P]); disc = np.array([p["discourse_score"] for p in P]); w = np.array([p["weight"] for p in P])
ok = pl == g
cons = np.where(pl == 0, ls, np.where(pl == 1, -ls, 1 - np.abs(ls)))
auc = lambda m, s: roc_auc_score(ok[m], s[m]) if 0 < ok[m].mean() < 1 else float("nan")

print("\n== 伪标签（关闭先验校正）==")
print("| 来源 | 条数 | 准确率 | 精度 SUP/REF/NEI | 置信度≥0.7 条数/准确率 |")
print("|---|---|---|---|---|")
for name, m in [("全部", np.ones(len(P), bool))] + [(s, src == s) for s in SRC]:
    prec = "/".join(f"{ok[m & (pl == k)].mean():.2f}" if (m & (pl == k)).any() else "—" for k in range(3))
    h = m & (conf >= 0.7)
    print(f"| {name} | {m.sum()} | {ok[m].mean():.3f} | {prec} | {h.sum()} / {ok[h].mean():.3f} |")

print("\n== 各信号区分伪标签对错的 AUROC（LogicScore 已修正）==")
print("| 来源 | 置信度 | \\|LogicScore\\| | 方向一致性 c | Discourse | 复合权重 |")
print("|---|---|---|---|---|---|")
for name, m in [("全部", np.ones(len(P), bool))] + [(s, src == s) for s in SRC]:
    print(f"| {name} | {auc(m, conf):.3f} | {auc(m, np.abs(ls)):.3f} | {auc(m, cons):.3f} | {auc(m, disc):.3f} | {auc(m, w):.3f} |")
print("\n== B+L 式过滤（置信度≥0.7 且 c>0.3）==")
for name, m in [("全部", np.ones(len(P), bool))] + [(s, src == s) for s in SRC]:
    b = m & (conf >= 0.7); bl = b & (cons > 0.3)
    print(f"  {name}: B {b.sum()} 条 / {ok[b].mean():.3f}  →  B+L {bl.sum()} 条 / {ok[bl].mean() if bl.any() else float('nan'):.3f}")

print("\n== detector 测试集 macro-F1 ==")
test = {r["id"]: r for r in ld(V2 / "processed/sci/labeled/test.jsonl")}
print("| 组 | 伪标签条数 | 参数更新 | 最佳 epoch | 全部 | ClimateCheck | Climate-FEVER | SciFact | AUC |")
print("|---|---|---|---|---|---|---|---|---|")
for m in "ABLO":
    rp = RUN / f"outputs/detector/{m}"
    if not (rp / "test_results.json").exists(): continue
    res = json.load(open(rp / "test_results.json")); pr = ld(rp / "test_predictions.jsonl")
    f = lambda s: f1_score([r["gold"] for r in pr if s is None or test[r["id"]]["source"] == s],
                           [r["pred"] for r in pr if s is None or test[r["id"]]["source"] == s], average="macro")
    print(f"| {m} | {res['pseudo_size']} | {res.get('total_optimizer_steps')} | {res['best_epoch']} | {f(None):.3f} | "
          + " | ".join(f"{f(s):.3f}" for s in SRC) + f" | {res['auc']:.3f} |")
rng = np.random.default_rng(0)
for a, b in (("O", "A"), ("B", "A"), ("L", "A"), ("L", "B"), ("O", "B")):
    pa, pb = RUN / f"outputs/detector/{a}/test_predictions.jsonl", RUN / f"outputs/detector/{b}/test_predictions.jsonl"
    if pa.exists() and pb.exists():
        r = paired_bootstrap(load_predictions(pa), load_predictions(pb), 1000, rng)
        print(f"  {a} − {b} = {r['delta_macro_f1']:+.4f}  95% CI ({r['ci_low']:+.4f}, {r['ci_high']:+.4f})  P(Δ≤0)={r['p_delta_le_0']:.3f}")

print("\n== 只看 claim 的基线（TF-IDF + LR）==")
tr_all = ld(V2 / "processed/sci/labeled/train.jsonl"); lab = ld(RUN / "data/labeled_train.jsonl"); te = list(test.values())
for name, tr in (("10% 有标签部分（与 A 同样的标签）", lab), ("全部 train 标签", tr_all)):
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v.fit_transform([r["claim"] for r in tr]), [r["label"] for r in tr])
    pred = clf.predict(v.transform([r["claim"] for r in te]))
    print(f"  {name}: macro-F1 {f1_score([r['label'] for r in te], pred, average='macro'):.3f}")
