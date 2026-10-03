# 泄漏探针：只用 claim / 只用证据 / claim+证据 的 TF-IDF + LR，比较 v1 与 v2 数据。在 Semi-vast/ 下运行：python v2/analysis/leakage_probe.py
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
ld=lambda p:[json.loads(l) for l in open(p)]
def uniq(r):
    s=set(); o=[]
    for x in r:
        if x["id"] not in s: s.add(x["id"]); o.append(x)
    return o
fields={"claim only":lambda x:x["claim"],"evidence only":lambda x:" ".join(x["evidence"]),
        "claim+evidence":lambda x:x["claim"]+" [SEP] "+" ".join(x["evidence"]),
        "evidence 条数(仅CF)":None}
print(f"{'输入':22}{'子集':10}{'v1 macroF1':>12}{'v2 macroF1':>12}")
for name,f in fields.items():
    for sub in ("all","climate_fever","pubhealth"):
        row=[]
        for root in ("processed","v2/processed"):
            tr=uniq(ld(f"{root}/labeled/train.jsonl")); te=uniq(ld(f"{root}/labeled/test.jsonl"))
            if sub!="all": tr=[x for x in tr if x["source"]==sub]; te=[x for x in te if x["source"]==sub]
            if f is None:
                if sub!="climate_fever": row=None; break
                Xtr=[[len(x["evidence"])==k for k in (1,2,3)] for x in tr]; Xte=[[len(x["evidence"])==k for k in (1,2,3)] for x in te]
            else:
                v=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=200000)
                Xtr=v.fit_transform([f(x) for x in tr]); Xte=v.transform([f(x) for x in te])
            clf=LogisticRegression(max_iter=2000,C=4,class_weight="balanced").fit(Xtr,[x["label"] for x in tr])
            row.append(f1_score([x["label"] for x in te],clf.predict(Xte),average="macro"))
        if row: print(f"{name:22}{sub:14}{row[0]:>10.3f}{row[1]:>12.3f}")
