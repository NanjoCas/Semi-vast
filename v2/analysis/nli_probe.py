# 四个数据集（PUBHEALTH、Climate-FEVER、ClimateCheck、SciFact）上的 NLI 零样本检验 + 修正后 LogicScore 作为伪标签信号的 AUROC。
# 在 Semi-vast/ 下运行：python v2/analysis/nli_probe.py。依赖 v2/runs/r0.10_s42（PUBHEALTH 版）的伪标签池和 v2/analysis/scifact_pairs.jsonl。
import json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score
NAME = "cross-encoder/nli-deberta-v3-large"
tok = AutoTokenizer.from_pretrained(NAME, cache_dir="model_cache")
model = AutoModelForSequenceClassification.from_pretrained(NAME, cache_dir="model_cache").cuda().eval()
i2l = {int(k): v for k, v in model.config.id2label.items()}
CON, ENT, NEU = [next(i for i, v in i2l.items() if v == n) for n in ("contradiction", "entailment", "neutral")]
print("label order from config:", i2l)

def nli(premises, hyps, bs=64):
    order = np.argsort([len(p) for p in premises]); out = np.zeros((len(premises), 3), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([premises[i] for i in idx], [hyps[i] for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return out

L = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
sets = {}
# PUBHEALTH / Climate-FEVER: v2 unlabeled pool (gold hidden labels), evidence joined as in extractor.generate_pseudolabels
gold = {json.loads(l)["id"]: json.loads(l)["label"] for l in open("v2/runs/r0.10_s42/data/unlabeled_gold.jsonl")}
pool = [json.loads(l) for l in open("v2/runs/r0.10_s42/pseudo/pseudo_pool.jsonl")]
for src, name in (("pubhealth", "PUBHEALTH"), ("climate_fever", "Climate-FEVER")):
    P = [p for p in pool if p["source"] == src]
    sets[name] = dict(prem=[" ".join(p["evidence"][:3]) for p in P], hyp=[p["claim"] for p in P], gold=[gold[p["id"]] for p in P], pool=P)
cc = pd.concat([pd.read_parquet(f"Data/ClimateCheck/{s}-00000-of-00001.parquet") for s in ("train", "test")])
ccmap = {"Supports": "SUPPORTS", "Refutes": "REFUTES", "Not Enough Information": "NOT_ENOUGH_INFO"}
sets["ClimateCheck"] = dict(prem=cc.abstract.tolist(), hyp=cc.claim.tolist(), gold=[ccmap[a] for a in cc.annotation])
corpus = {d["doc_id"]: d for d in map(json.loads, open("Data/SciFact/data/corpus.jsonl"))}
sf = pd.read_json("v2/analysis/scifact_pairs.jsonl", lines=True)
sfmap = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO"}
sets["SciFact"] = dict(prem=[" ".join(corpus[d]["abstract"]) for d in sf.doc], hyp=sf.claim.tolist(), gold=[sfmap[l] for l in sf.label])

res = {}
for name, d in sets.items():
    pr = nli(d["prem"], d["hyp"]); g = np.array([L.index(x) for x in d["gold"]])
    res[name] = (pr, g)
    if "pool" in d:   # validate the bug diagnosis against stored logic_score
        stored = np.array([p["logic_score"] for p in d["pool"]])
        buggy, fixed = pr[:, NEU] - pr[:, CON], pr[:, ENT] - pr[:, CON]
        print(f"[{name}] 与已保存 LogicScore 的相关系数：按旧代码（中立−矛盾）{np.corrcoef(stored, buggy)[0,1]:.3f}，按正确定义（蕴含−矛盾）{np.corrcoef(stored, fixed)[0,1]:.3f}")
np.savez("v2/analysis/nli_probs.npz", **{k: v[0] for k, v in res.items()}, **{k + "_gold": v[1] for k, v in res.items()})

print("\n== NLI 判断随金标签的分布（argmax：蕴含 / 矛盾 / 中立）与零样本效果 ==")
print("| 数据集 | 金标签 | n | 判蕴含 | 判矛盾 | 判中立 |")
print("|---|---|---|---|---|---|")
for name, (pr, g) in res.items():
    am = pr.argmax(1)
    for k, lab in enumerate(L):
        m = g == k
        print(f"| {name} | {lab} | {m.sum()} | {np.mean(am[m]==ENT):.0%} | {np.mean(am[m]==CON):.0%} | {np.mean(am[m]==NEU):.0%} |")
print("\n| 数据集 | 零样本准确率 | 零样本 macro-F1 | 多数类准确率 |")
print("|---|---|---|---|")
for name, (pr, g) in res.items():
    pred = np.select([pr.argmax(1) == ENT, pr.argmax(1) == CON], [0, 1], 2)
    print(f"| {name} | {np.mean(pred == g):.3f} | {f1_score(g, pred, average='macro'):.3f} | {np.bincount(g).max()/len(g):.3f} |")

# corrected LogicScore as a signal for pseudo-label correctness (the README 7.2.2 metric), v2 pool, no prior adjustment
st = json.load(open("v2/runs/r0.10_s42/pseudo/pseudo_stats.json"))["prior_adjustment"]
prior, tau = np.array(st["priors"]), st["tau"]
print("\n== 修正后的 LogicScore 作为伪标签对错信号的 AUROC（v2 池，关闭先验校正后的伪标签）==")
for name in ("PUBHEALTH", "Climate-FEVER"):
    pr, g = res[name]; P = sets[name]["pool"]
    adj = np.array([p["probs"] for p in P]); z = np.log(adj + 1e-12) + tau * np.log(prior)
    raw = np.exp(z - z.max(1, keepdims=True)); raw /= raw.sum(1, keepdims=True)
    pl, conf = raw.argmax(1), raw.max(1); ok = pl == g
    ls = pr[:, ENT] - pr[:, CON]
    cons = np.where(pl == 0, ls, np.where(pl == 1, -ls, 1 - np.abs(ls)))
    agree_nli = np.select([pr.argmax(1) == ENT, pr.argmax(1) == CON], [0, 1], 2) == pl
    print(f"  {name}: 置信度 {roc_auc_score(ok, conf):.3f} | 修正后 |LS| {roc_auc_score(ok, np.abs(ls)):.3f} | 修正后方向一致性 {roc_auc_score(ok, cons):.3f} | "
          f"置信度≥0.7 且 NLI 判断与伪标签一致: n={np.sum((conf>=0.7)&agree_nli)} 准确率={ok[(conf>=0.7)&agree_nli].mean():.3f}（仅置信度≥0.7: n={np.sum(conf>=0.7)} 准确率={ok[conf>=0.7].mean():.3f}）")
