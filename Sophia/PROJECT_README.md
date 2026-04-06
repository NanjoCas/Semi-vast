# PROJECT README — AI Agent Reference Document
# Semi-Supervised Fake News Detection via Reinforcement Learning and Logical Feature Integration

> **Purpose of this document**: This README is designed to be consumed by an AI coding agent. It provides complete architectural specifications, implementation tasks, file locations, formulas, and constraints required to implement this research project end-to-end.
>
> **Author**: LIU Bo, Sophia University (Environmental School)
> **Domain**: Environmental & Public Health Misinformation Detection

---

## 1. Project Overview

### 1.1 Research Goal

Build a **semi-supervised misinformation detection framework** that:
1. Reduces dependence on labeled data by leveraging large unlabeled corpora
2. Uses **Reinforcement Learning** to dynamically select high-quality pseudo-labeled samples
3. Integrates **logical and argumentative structure features** for interpretability
4. Targets the **environmental and public health** domain (NOT COVID-19)

### 1.2 Key Innovation

Most existing models rely on confidence scores alone for pseudo-label selection. This project adds:
- **LogicScore**: logical consistency between claim and evidence (entailment vs. contradiction)
- **DiscourseScore**: linguistic cues (negation, modality, causal expressions, sentiment polarity)
- **RL-based selector (PPO)**: treats pseudo-label quality as a reward signal

### 1.3 Evaluation Metrics

- Accuracy, F1 (macro), Precision, Recall, AUC
- Logic Consistency Score (custom metric)
- Interpretability (attention visualization, rationale extraction)

---

## 2. System Architecture

The pipeline consists of three sequential components:

```
[Labeled Data]  ──► Textual Feature Extractor ──► pseudo-labels + scores
                           │                              │
[Unlabeled Data] ──────────┘               Reinforced Selector (PPO)
                                                          │
                                               filtered pseudo-labeled set
                                                          │
                               Fake News Detector (Dual-Channel DeBERTa)
                               ├── Reasoning Channel (labeled claim+evidence)
                               └── Content Channel (pseudo-labeled claim-only)
```

---

## 3. Component Specifications

### 3.1 Component A: Textual Feature Extractor

**Purpose**: Pre-train on labeled data; generate pseudo-labels + confidence scores for unlabeled claims.

**Base Model**: `microsoft/deberta-v3-base`

**Task**: 3-way classification — `SUPPORTS (0)`, `REFUTES (1)`, `NOT_ENOUGH_INFO (2)`

**Loss Function**:
```
L_sup = -Σ_i  y_i * log(ŷ_i)
```
Where `y_i` is one-hot true label, `ŷ_i` is predicted probability.

**LogicScore Computation** (uses separate model `cross-encoder/nli-deberta-v3-large`):
```python
LogicScore = p_entail - p_contradict
# p_entail:    probability that claim is SUPPORTED by evidence
# p_contradict: probability that claim is CONTRADICTED by evidence
# Range: [-1, 1], higher = more logically consistent
```

**DiscourseScore** — extract from claim text using rule-based or fine-tuned probes:
- Negation cues: "not", "never", "no evidence", "contrary to"
- Modality: "may", "might", "could", "allegedly", "claimed"
- Causal: "because", "therefore", "leads to", "as a result"
- Sentiment polarity: VADER or TextBlob score on claim text

**Combined pseudo-label weight**:
```
w_i = β₁ * p_clf + β₂ * |LogicScore| + β₃ * discourse_score
```
Default: `β₁=0.5, β₂=0.3, β₃=0.2` (tunable hyperparameters)

**Output per unlabeled sample**:
```json
{
  "id": "tw_a1b2c3d4",
  "claim": "...",
  "pseudo_label": 0,
  "confidence": 0.87,
  "logic_score": 0.62,
  "discourse_score": 0.41,
  "weight": 0.73
}
```

**Input format** (from `processed/labeled/train.jsonl`):
```json
{
  "id": "cf_75",
  "claim": "Arctic sea ice has declined significantly since 1980.",
  "evidence": ["Arctic sea ice extent has decreased by 13% per decade..."],
  "label": "SUPPORTS",
  "source": "climate_fever"
}
```

---

### 3.2 Component B: Reinforced Selector

**Purpose**: Use PPO to learn which pseudo-labeled samples improve model performance.

**Algorithm**: Proximal Policy Optimization (PPO), library: `stable-baselines3` or custom `torch`

**State representation** (per pseudo-labeled sample):
```
s_t = [confidence, entropy, |LogicScore|, diversity]
```
- `confidence`: `p_clf` from extractor
- `entropy`: `-Σ p_k * log(p_k)` over class probabilities (lower = more certain)
- `|LogicScore|`: absolute logical consistency
- `diversity`: cosine distance from centroid of already-selected samples (promotes variety)

**Action space**: Binary — `{0: discard, 1: keep}`

**Reward function**:
```
R_t = α * ΔF1 + β * mean(|LogicScore|_selected)
```
- `ΔF1`: change in validation macro-F1 after adding selected samples to training
- Default: `α=0.7, β=0.3`

**PPO Clipped Objective**:
```
L^CLIP(θ) = E_t [ min(r_t(θ) * Â_t,  clip(r_t(θ), 1-ε, 1+ε) * Â_t) ]
```
Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` — policy ratio
- `Â_t` — advantage estimate (GAE recommended)
- `ε = 0.2` — clipping parameter

**Output**: Filtered pseudo-labeled set with per-sample weights:
```json
{
  "id": "tw_a1b2c3d4",
  "claim": "...",
  "pseudo_label": 0,
  "weight": 0.73
}
```

---

### 3.3 Component C: Fake News Detector

**Purpose**: Final classification model trained jointly on labeled + selected pseudo-labeled data.

**Base Model**: `microsoft/deberta-v3-base`

**Architecture**: Dual-Channel with shared encoder

```
                    ┌─────────────────────────────┐
                    │   Shared DeBERTa-v3-base     │
                    │   Encoder (frozen early,     │
                    │   fine-tuned later)          │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┴──────────────────┐
              ▼                                    ▼
   Reasoning Channel                    Content Channel
   (labeled: claim + evidence)          (pseudo: claim only)
   → factual reasoning                  → linguistic patterns
              └────────────────┬──────────────────┘
                               ▼
                    Shared Classification Head
                    (Linear → Softmax, 3 classes)
```

**Joint Loss**:
```
L = L_sup + λ * Σ_i (w_i * L_pseudo_i)
```
- `L_sup`: Cross-entropy on labeled claim-evidence pairs
- `L_pseudo`: Cross-entropy on pseudo-labeled claim-only samples
- `w_i`: confidence weight from selector output
- `λ < 1` (default `λ=0.3`) — ensures labeled data dominates early training

**Multi-level balancing mechanism**:
1. **Global level (λ)**: scale pseudo-labeled gradient contribution
2. **Sample level (w_i)**: down-weight low-confidence pseudo-labels
3. **Batch composition**: maintain labeled:pseudo ratio of ~1:3 per batch

---

## 4. Data Specification

### 4.1 Dataset Summary

| Dataset | Role | Size | Format | Location |
|---------|------|------|--------|----------|
| Climate-FEVER | Labeled (primary) | 1,535 claims / 7,675 evidence pairs | JSONL | `Data/Climate Fever Dataset/archive/climate-fever.jsonl` |
| PUBHEALTH | Labeled (supplement, COVID filtered) | ~8,000–9,000 claims | TSV | `Data/PUBHEALTH-DATASET/archive/` |
| SciFact | Labeled (scientific claims) | 1,409 claims / 5,183 abstracts | JSONL | `Data/SciFact/data/` |
| Guardian Environment News | Unlabeled pool | 30,059 articles | CSV | `Data/Environment News Dataset/guardian_environment_news.csv` |

> Note: SciFact raw data is available under `Data/SciFact/data/` as an additional labeled scientific claims source. It is not currently integrated into the default `data_processing/process_labeled.py` pipeline.

### 4.2 Processed Data (Ready to Use)

After running `data_processing/run_pipeline.py`, the following files are available:

```
processed/
├── labeled/
│   ├── train.jsonl          # Climate-FEVER (×2 oversampled) + PUBHEALTH
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── processing_stats.json
└── unlabeled/
    ├── unlabeled_pool.jsonl                # Main input for pseudo-label generation
    ├── unlabeled_environment_news.jsonl    # Source-specific environment news output
    └── processing_stats.json
```

**Labeled record schema**:
```json
{
  "id":       "cf_75",
  "claim":    "string",
  "evidence": ["string", "string"],
  "label":    "SUPPORTS | REFUTES | NOT_ENOUGH_INFO",
  "source":   "climate_fever | pubhealth"
}
```

**Unlabeled record schema**:
```json
{
  "id":     "env_abcdef12",
  "claim":  "string",
  "source": "guardian_environment_news"
}
```

### 4.3 Label Mapping Reference

| Original | Mapped | Notes |
|----------|--------|-------|
| Climate-FEVER: SUPPORTS | SUPPORTS | — |
| Climate-FEVER: REFUTES | REFUTES | — |
| Climate-FEVER: DISPUTED | REFUTES | Both supporting & refuting evidence exist |
| Climate-FEVER: NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | — |
| PUBHEALTH: true | SUPPORTS | — |
| PUBHEALTH: false | REFUTES | — |
| PUBHEALTH: unproven | NOT_ENOUGH_INFO | — |
| PUBHEALTH: mixture | NOT_ENOUGH_INFO | Partially true |

---

## 5. File Structure

```
project_root/
│
├── PROJECT_README.md               ← This file (agent reference)
│
├── data_processing/                ← Data pipeline (COMPLETE)
│   ├── process_labeled.py          ✓ Climate-FEVER + PUBHEALTH processing
│   ├── process_unlabeled.py        ✓ Environment News Dataset processing
│   └── run_pipeline.py             ✓ Main runner + PyTorch Dataset classes
│
├── Data/                          ← Raw downloaded datasets
│   ├── Climate Fever Dataset/
│   ├── PUBHEALTH-DATASET/
│   ├── SciFact/
│   │   └── data/
│   └── Environment News Dataset/
│
├── processed/                      ← Auto-generated by run_pipeline.py
│   ├── labeled/
│   └── unlabeled/
│
├── models/                         ← TO BE IMPLEMENTED
│   ├── extractor.py                ✗ Textual Feature Extractor (DeBERTa-v3-base)
│   ├── logic_scorer.py             ✗ LogicScore computation (DeBERTa-v3-large-MNLI)
│   ├── discourse_scorer.py         ✗ DiscourseScore feature extraction
│   ├── rl_selector.py              ✗ PPO-based Reinforced Selector
│   └── detector.py                 ✗ Dual-Channel Fake News Detector
│
├── training/                       ← TO BE IMPLEMENTED
│   ├── train_extractor.py          ✗ Phase 1: supervised pre-training
│   ├── generate_pseudolabels.py    ✗ Phase 2: pseudo-label generation
│   ├── train_rl_selector.py        ✗ Phase 3: PPO selector training
│   └── train_detector.py           ✗ Phase 4: joint training
│
├── evaluation/                     ← TO BE IMPLEMENTED
│   ├── metrics.py                  ✗ Accuracy, F1, AUC, LogicConsistency
│   └── evaluate.py                 ✗ Full evaluation script
│
├── configs/
│   └── config.yaml                 ← TO BE CREATED (hyperparameters)
│
└── requirements.txt                ← TO BE CREATED
```

---

## 6. Implementation Tasks (Ordered)

### Task 1 — Environment Setup
**Files to create**: `requirements.txt`, `configs/config.yaml`

`requirements.txt` must include:
```
torch>=2.0.0
transformers>=4.35.0
datasets
pandas
scikit-learn
stable-baselines3   # for PPO
gym                 # RL environment
numpy
tqdm
wandb               # experiment tracking (optional)
```

`configs/config.yaml` must specify:
- model names: `microsoft/deberta-v3-base`, `cross-encoder/nli-deberta-v3-large`
- data paths: `processed/labeled/`, `processed/unlabeled/`
- hyperparameters: `λ=0.3`, `β₁=0.5, β₂=0.3, β₃=0.2`, `α=0.7, β=0.3`, `ε=0.2`
- training settings: batch_size, learning_rate, max_epochs, warmup_steps

---

### Task 2 — Implement Logic Scorer
**File**: `models/logic_scorer.py`

```python
# Pseudocode specification:
# Input: claim (str), evidence (str)
# Model: cross-encoder/nli-deberta-v3-large (NLI model)
# Output: LogicScore = p_entail - p_contradict
# NLI labels order: ["contradiction", "neutral", "entailment"]
# p_entail = softmax_output[2]
# p_contradict = softmax_output[0]
# Batch processing required (GPU efficiency)
```

---

### Task 3 — Implement Discourse Scorer
**File**: `models/discourse_scorer.py`

Must extract three sub-scores from claim text:
1. `negation_score`: count of negation cues normalized by claim length
2. `modality_score`: count of epistemic modality markers normalized by length
3. `sentiment_score`: absolute sentiment polarity (use VADER: `pip install vaderSentiment`)

Final discourse_score = weighted average of sub-scores.

---

### Task 4 — Implement Textual Feature Extractor
**File**: `models/extractor.py`

Requirements:
- Fine-tune `microsoft/deberta-v3-base` for 3-way classification
- Input: `[CLS] claim [SEP] evidence_1 [SEP] evidence_2 [SEP]` (max 512 tokens)
- Output: class logits (3,) + confidence score (max softmax probability)
- Must expose `generate_pseudo_labels(unlabeled_pool)` method
- Must compute per-sample weights using formula: `w = β₁*p_clf + β₂*|LogicScore| + β₃*discourse_score`

Use the `ClaimEvidenceDataset` and `UnlabeledClaimDataset` classes from `data_processing/run_pipeline.py`.

---

### Task 5 — Implement PPO Reinforced Selector
**File**: `models/rl_selector.py`

Requirements:
- State dim: 4 (`[confidence, entropy, |LogicScore|, diversity]`)
- Action space: Discrete(2) — keep/discard
- Reward: `R = α * ΔF1 + β * mean(|LogicScore|_selected)`
- To compute `ΔF1`: maintain a held-out validation set; retrain a lightweight probe (e.g., logistic regression on DeBERTa embeddings) and compute macro-F1 delta
- Use `stable-baselines3` PPO or implement custom PPO in PyTorch
- Policy network: 2-layer MLP (4 → 64 → 32 → 2)

---

### Task 6 — Implement Dual-Channel Fake News Detector
**File**: `models/detector.py`

Requirements:
- Shared `microsoft/deberta-v3-base` encoder
- Reasoning Channel input: `[CLS] claim [SEP] evidence [SEP]`
- Content Channel input: `[CLS] claim [SEP]` (no evidence)
- Shared classification head (Linear → Dropout(0.1) → Linear → Softmax)
- Joint loss: `L = L_sup + λ * Σ_i(w_i * L_pseudo_i)`
- Lambda scheduler: start `λ=0.1`, anneal to `λ=0.3` over training
---

### Task 7 — Training Scripts
**Files**: `training/train_extractor.py`, `training/generate_pseudolabels.py`, `training/train_rl_selector.py`, `training/train_detector.py`

Training order is strictly sequential:
1. `train_extractor.py` → trains Feature Extractor on `processed/labeled/train.jsonl`
2. `generate_pseudolabels.py` → runs Extractor on `processed/unlabeled/unlabeled_pool.jsonl`
3. `train_rl_selector.py` → trains PPO Selector using pseudo-labeled pool + validation set
4. `train_detector.py` → joint training with labeled + RL-selected pseudo-labeled data

---

### Task 8 — Evaluation
**File**: `evaluation/evaluate.py`

Must report on `processed/labeled/test.jsonl`.
- Accuracy, Macro-F1, Precision, Recall per class
- AUC (one-vs-rest)
- Logic Consistency Score: average `|LogicScore|` of correctly classified samples
- Confusion matrix

---

## 7. Key Constraints and Notes for Agent

1. **No COVID data**: Any data pipeline modification must maintain COVID filtering. Regex: `r"\b(covid|coronavirus|sars[\-\s]?cov|pandemic|lockdown|quarantine|mRNA vaccine|pfizer|moderna|astrazeneca)\b"`

2. **Label imbalance handling**: The labeled dataset has class imbalance (Climate-FEVER: SUPPORTS 42.7%, NOT_ENOUGH_INFO 30.9%, REFUTES 16.5%, DISPUTED 10%). Use weighted cross-entropy: `torch.nn.CrossEntropyLoss(weight=class_weights)` where `class_weights` are inverse class frequencies.

3. **Evidence-only vs claim-only**: The Reasoning Channel receives `claim + evidence`; the Content Channel receives `claim only`. Do NOT pass evidence to the Content Channel even if it is available for pseudo-labeled samples.

4. **DISPUTED label**: Climate-FEVER has 4 labels; all processing scripts map `DISPUTED → REFUTES`. Do not re-introduce the 4-class problem.

5. **DeBERTa tokenizer**: `microsoft/deberta-v3-base` uses SentencePiece tokenizer. Token type IDs may not be used by all DeBERTa variants — check `tokenizer.model_type` before using `token_type_ids`.

6. **Memory constraints**: DeBERTa-v3-base with batch_size=16 and max_length=512 requires ~16GB GPU VRAM. If constrained, use gradient checkpointing (`model.gradient_checkpointing_enable()`) and reduce batch size with gradient accumulation.

7. **PUBHEALTH evidence style**: PUBHEALTH explanations are longer narrative texts (avg ~200 words) vs. Climate-FEVER Wikipedia sentences (avg ~30 words). Consider adding a source-specific embedding or domain token `[CLIMATE]` / `[HEALTH]` prepended to the input.

8. **Reproducibility**: Fix all random seeds at the start of every script: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `transformers.set_seed(42)`.

---

## 8. Expected Experimental Results (Baseline Reference)

Based on related work, reasonable target performance on the test set:

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| Accuracy | 0.72 | 0.80 |
| Macro F1 | 0.65 | 0.75 |
| NOT_ENOUGH_INFO F1 | 0.55 | 0.65 |
| Logic Consistency | 0.50 | 0.65 |

Compare against ablation baselines:
- Baseline A: Supervised only (no pseudo-labels)
- Baseline B: Semi-supervised with confidence threshold (no RL, no LogicScore)
- Baseline C: Full model (this project)

---

## 9. Quick Start for Agent

```bash
# Step 0: Clone project structure
pip install -r requirements.txt

# Step 1: Download datasets (run on local machine)
chmod +x download_datasets.sh && ./download_datasets.sh

# Step 2: Run data pipeline
cd data_processing
python run_pipeline.py
cd ..

# Step 3: Train (sequential)
python training/train_extractor.py --config configs/config.yaml
python training/generate_pseudolabels.py --config configs/config.yaml
python training/train_rl_selector.py --config configs/config.yaml
python training/train_detector.py --config configs/config.yaml

# Step 4: Evaluate
python evaluation/evaluate.py --config configs/config.yaml --split test
```

---

## 10. References

- Diggelmann et al. (2020). *CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims.* NeurIPS Workshop. [GitHub](https://github.com/tdiggelm/climate-fever-dataset)
- Kotonya & Toni (2020). *Explainable Automated Fact-Checking for Public Health Claims.* EMNLP 2020. [GitHub](https://github.com/neemakot/Health-Fact-Checking)
- Wadden et al. (2020). *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020. [GitHub](https://github.com/allenai/scifact)
- He et al. (2021). *DeBERTaV3.* arXiv:2111.09543. [HuggingFace](https://huggingface.co/microsoft/deberta-v3-base)
- Wang et al. (2020). *Weak Supervision for Fake News Detection via Reinforcement Learning.* AAAI 2020.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
