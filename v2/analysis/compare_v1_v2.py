# v1 与 v2 有标签数据对比（样本集合、重复、Climate-FEVER 证据与标签一致率、证据差异）。在 Semi-vast/ 下运行：python v2/analysis/compare_v1_v2.py
import json, collections
from collections import Counter
ld=lambda p:[json.loads(l) for l in open(p)]
V1={s:ld(f"processed/labeled/{s}.jsonl") for s in ["train","dev","test"]}
V2={s:ld(f"v2/processed/labeled/{s}.jsonl") for s in ["train","dev","test"]}
print("== 1. 规模 / 重复 ==")
for s in V1:
    for n,D in (("v1",V1),("v2",V2)):
        ids=[x["id"] for x in D[s]]; c=Counter(x["source"] for x in D[s])
        print(f"{s:5} {n}: total={len(ids):5} unique={len(set(ids)):5} dup_rows={len(ids)-len(set(ids)):4} by_source={dict(c)}")
print("\n== 2. 唯一样本集合与 split 归属是否一致 ==")
for s in V1:
    a={x["id"] for x in V1[s]}; b={x["id"] for x in V2[s]}
    print(f"{s:5} v1_unique={len(a)} v2={len(b)} same_set={a==b}  only_v1={len(a-b)} only_v2={len(b-a)}")
lab1={x["id"]:x["label"] for s in V1 for x in V1[s]}; lab2={x["id"]:x["label"] for s in V2 for x in V2[s]}
print("label mismatches:",sum(lab1[k]!=lab2[k] for k in lab1 if k in lab2))
# 3. Climate-FEVER evidence: label agreement of selected evidence
raw={f"cf_{j['claim_id']}":j for j in map(json.loads,open("Data/Climate Fever Dataset/archive/climate-fever.jsonl"))}
print("\n== 3. Climate-FEVER 被选证据的 evidence_label 与 claim_label 一致率（衡量是否用金标签选证据）==")
for n,D in (("v1",V1),("v2",V2)):
    for s in D:
        seen=set(); agree=tot=0; same_ev=0; per=Counter(); pertot=Counter()
        for x in D[s]:
            if x["source"]!="climate_fever" or x["id"] in seen: continue
            seen.add(x["id"]); r=raw[x["id"]]; m={e["evidence"].strip():e["evidence_label"] for e in r["evidences"]}
            for e in x["evidence"]:
                el=next((v for k,v in m.items() if e.rstrip(" .")[:60] in k),None)
                tot+=1; ok= el==r["claim_label"]; agree+=ok; per[x["label"]]+=ok; pertot[x["label"]]+=1
        print(f"  {n} {s:5}: {agree/tot:.1%}  分标签: " + ", ".join(f"{k}={per[k]/pertot[k]:.0%}" for k in sorted(pertot)))
# evidence count distribution v1 CF (count of evidences reveals label?)
print("\n== 4. v1 Climate-FEVER 证据条数 vs 标签（条数本身是否泄露标签）==")
seen=set(); tab=collections.defaultdict(Counter)
for s in V1:
    for x in V1[s]:
        if x["source"]=="climate_fever" and x["id"] not in seen:
            seen.add(x["id"]); tab[x["label"]][len(x["evidence"])]+=1
for k,v in tab.items(): print(" ",k,dict(sorted(v.items())))
ov=sum(1 for s in V1 for x,y in [(a,None) for a in V1[s]] if False)
print("\n== 5. 证据文本差异 ==")
m2={x["id"]:x["evidence"] for s in V2 for x in V2[s]}
for src in ("climate_fever","pubhealth"):
    seen=set(); same=n=0; l1=l2=0
    for s in V1:
        for x in V1[s]:
            if x["source"]!=src or x["id"] in seen: continue
            seen.add(x["id"]); n+=1; same+= x["evidence"]==m2[x["id"]]
            l1+=len(" ".join(x["evidence"]).split()); l2+=len(" ".join(m2[x["id"]]).split())
    print(f"  {src}: 证据完全相同 {same}/{n}，平均证据词数 v1={l1/n:.0f} v2={l2/n:.0f}")
