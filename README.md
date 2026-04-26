# KRR Fact Verification System

[![Test Project](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml/badge.svg)](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml)

> Attribution-aware fact verification using Knowledge Representation and Reasoning (KRR) — a master's group project.

---

## Overview

### The Problem

Large language models frequently hallucinate facts and provide no evidence for their claims. When a model says "SUPPORTS" or "REFUTES", there is no way to trace *why* — which sentences were used, which facts were matched, or where the reasoning went wrong. This makes LLM-based fact verification unauditable and untrustworthy in high-stakes settings.

### The Solution

This project implements a **transparent, symbolic KRR pipeline** for automated fact verification against the FEVER dataset. Instead of asking a model to classify a claim, the system:

1. Retrieves relevant evidence sentences using hybrid semantic + lexical search
2. Extracts structured (subject, relation, object) triples from both the claim and evidence
3. Compares triples symbolically using explicit reasoning rules
4. Returns a verdict — SUPPORTS, REFUTES, or NOT ENOUGH INFO — with **full attribution** to the specific triples that drove the decision

Every prediction is explainable. You can inspect exactly which evidence triple matched the claim and why.

---

## Architecture

```
                          Input Claim
                               │
                    ┌──────────▼──────────┐
                    │   Hybrid Retriever  │
                    │  TF-IDF + semantic  │
                    │  (all-MiniLM-L6-v2) │
                    └──────────┬──────────┘
                               │  top-K evidence sentences
                    ┌──────────▼──────────┐
                    │   Triple Extractor  │
                    │  spaCy dep. parsing │
                    │  - passive voice    │
                    │  - negation         │
                    │  - comparatives     │
                    │  - nested location  │
                    └──────────┬──────────┘
                               │  claim triples + evidence triples
                    ┌──────────▼──────────┐
                    │   Reasoning Module  │
                    │  - relation norms   │
                    │  - synonym matching │
                    │  - subject aliases  │
                    │  - numeric compare  │
                    │  - negation polarity│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Verdict + Attribution│
                    │  SUPPORTS / REFUTES  │
                    │  / NOT ENOUGH INFO   │
                    └─────────────────────┘
```

### Retrieval — Hybrid TF-IDF + Semantic

The retriever combines two signals:

- **TF-IDF cosine similarity** (sklearn) — fast lexical overlap
- **Sentence embeddings** (all-MiniLM-L6-v2 via sentence-transformers) — semantic similarity

Final score: `0.3 × tfidf + 0.7 × semantic`

This hybrid approach retrieves the correct evidence even when the claim and evidence use different words (e.g. "tallest" vs "highest", "wrote" vs "written by").

### Triple Extraction — spaCy

The `TripleExtractor` uses spaCy's dependency parser to produce `(subject, relation, object)` triples. It handles:

| Structure | Example | Triple |
|-----------|---------|--------|
| Active SVO | "Shakespeare wrote Hamlet" | `shakespeare\|write\|hamlet` |
| Passive voice | "Hamlet was written by Shakespeare" | `william shakespeare\|write\|hamlet` |
| Passive ACL | "a tragedy written by Shakespeare" | `william shakespeare\|write\|hamlet` |
| Negation | "not visible from space" | `great wall\|be\|not visible from space` |
| Comparative | "slower than the speed of sound" | `speed of light\|be\|slower than speed of sound` |
| Nested location | "tower on the Champ de Mars in Paris" | `eiffel tower\|locate\|in paris` |
| Full subject | "speed of light" (not truncated to "speed") | `speed of light\|be\|...` |

### Reasoning — Symbolic KRR

The `ReasoningModule` compares claim triples against evidence triples using:

1. **Relation normalization** — `become → be`, `locate → be`, `born → bear`
2. **Subject alias matching** — `"shakespeare"` matches `"william shakespeare"`
3. **Negation-aware object matching** — `"visible"` vs `"not visible"` → REFUTES
4. **Synonym canonicalization** — `"tallest"` ≡ `"highest"` via canonical form `"tall"`
5. **Weak-evidence filtering** — drops `"on march"`, `"during mission"`, `"at metres"` before they create spurious counts
6. **Comparative numerical reasoning** — extracts numeric values from evidence and compares them to resolve claims like "X is slower than Y"

---

## Dataset and Evaluation

### FEVER-Style Dataset

The dataset follows the FEVER format: each record contains a natural language claim, a label, and supporting evidence sentences.

Labels:
- `SUPPORTS` — the evidence confirms the claim
- `REFUTES` — the evidence contradicts the claim
- `NOT ENOUGH INFO` — the evidence is insufficient to decide

### Train / Test Split

| Split | File | Claims | SUPPORTS | REFUTES | NEI |
|-------|------|--------|----------|---------|-----|
| Train | `data/train.jsonl` | 14 | 7 | 5 | 2 |
| Test | `data/test.jsonl` | 6 | 3 | 2 | 1 |

**No data leakage.** The reasoning rules and synonym tables were developed using the training split only. Test claim labels are never seen during retrieval index construction or reasoning — they are used exclusively for final metric computation.

The retrieval corpus is built from evidence sentences in both splits (simulating a shared Wikipedia knowledge base, as in the original FEVER setup). Only evidence text is used — no labels.

---

## Results

> **Evaluation is performed on unseen test data only.**

| Pipeline | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI | Macro Precision | Macro Recall |
|----------|----------|-------------|------------|--------|-----------------|--------------|
| **KRR** | **0.833** | **0.800** | **1.000** | **0.667** | **0.833** | **0.889** |
| Baseline (Keyword) | 0.667 | 0.750 | 0.667 | 0.000 | 0.533 | 0.500 |

**KRR outperforms the keyword baseline by ~16.6 percentage points** on the test set.

Key observations:
- KRR achieves **perfect F1-REFUTES (1.000)** — it correctly identifies all refuted claims
- KRR is the only pipeline that predicts NOT ENOUGH INFO — the baseline scores **F1-NEI = 0.000**
- The baseline over-predicts SUPPORTS (it classifies every claim with keyword overlap as SUPPORTS)

### Confusion Matrix — KRR

```
              Predicted
              SUPPORTS  REFUTES  NEI
SUPPORTS         2         0      1
REFUTES          0         2      0
NEI              0         0      1
```

### Confusion Matrix — Baseline

```
              Predicted
              SUPPORTS  REFUTES  NEI
SUPPORTS         3         0      0
REFUTES          1         1      0
NEI              1         0      0
```

---

## Key Improvements

### 1. Hybrid Semantic Retrieval
Replaced pure TF-IDF with a hybrid retriever combining TF-IDF and `all-MiniLM-L6-v2` sentence embeddings (70% semantic weight). This retrieves the correct evidence even when the claim and evidence use different vocabulary — the primary cause of KRR failures in the original system.

### 2. Full Subject Extraction
Extended subject extraction to include prepositional chains (`of light`, `of sound`). Previously, "speed of light" was truncated to "speed", making it impossible to distinguish the speed of light from the speed of sound in comparative claims.

### 3. Numeric Comparative Reasoning
Added a dedicated resolver for comparative claims ("X is slower than Y"). It extracts numeric values from evidence triples (e.g. 299,792,458 m/s vs 343 m/s) and compares them to determine whether the comparative claim is supported or refuted.

### 4. Improved NEI Detection
The weak-evidence filter now exempts objects containing numeric values, ensuring measurement triples reach the comparative resolver. Combined with the tie-breaking rule (support = refute → NOT ENOUGH INFO), the system correctly abstains when evidence is ambiguous.

---

## Limitations

- **Ambiguous evidence → NEI**: When evidence supports and refutes in equal measure (e.g. "Great Barrier Reef is located in the Coral Sea off the coast of Queensland, Australia" produces both `locate|in coral sea` and `locate|in australia`), the system correctly abstains but cannot resolve the ambiguity without geographic inference.
- **Symbolic reasoning limits**: The system cannot perform multi-hop reasoning, coreference resolution, or inference over implicit knowledge. Claims requiring world knowledge beyond the retrieved sentences will fail.
- **Small dataset**: 6 test claims is insufficient for statistically robust evaluation. Results should be validated on the full FEVER dev set (19,998 claims).
- **Synonym coverage**: The synonym table is hand-curated. A thesaurus or word embeddings would generalise better to unseen vocabulary.

---

## Setup

### Prerequisites

- Python 3.11
- `pip`

### 1. Create virtual environment

```powershell
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate
```

```bash
# Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install spaCy model

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

### 4. Configure environment

```bash
cp .env.example .env
# Defaults work out of the box with the included dataset
```

---

## How to Run

### Run tests (104 tests, all passing)

```powershell
.\venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

### Run both pipelines — full evaluation on test set

```powershell
.\venv\Scripts\python.exe main.py --pipeline both
```

### Run KRR only

```powershell
.\venv\Scripts\python.exe main.py --pipeline krr
```

### Run Baseline only

```powershell
.\venv\Scripts\python.exe main.py --pipeline baseline
```

### Demo mode — verify a single claim interactively

```powershell
.\venv\Scripts\python.exe main.py --claim "The Eiffel Tower is located in Paris."
```

Example output:
```
======================================================================
  DEMO MODE — Claim: The Eiffel Tower is located in Paris.
======================================================================
--- BASELINE (Keyword) ---
  Verdict : SUPPORTS
  Evidence: The Eiffel Tower is a wrought-iron lattice tower ...

--- KRR VERDICT ---
  Verdict        : SUPPORTS
  Support count  : 1
  Refute count   : 0
  Irrelevant     : 4
  Attribution    :
    eiffel tower|locate|in paris
======================================================================
```

### Generate presentation graphs

```powershell
.\venv\Scripts\python.exe scripts/generate_graphs.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/eval_report.txt` | Human-readable evaluation report with confusion matrices |
| `output/metrics.json` | Machine-readable metrics — accuracy, F1, precision, recall, confusion matrix |
| `output/confusion_matrix.csv` | Confusion matrices for both pipelines in CSV format |
| `output/krr_results.jsonl` | Per-claim KRR output — triples, counts, verdict, attribution |
| `output/error_analysis.md` | Per-error breakdown with root-cause analysis |
| `output/figures/model_accuracy.png` | KRR vs Baseline accuracy bar chart |
| `output/figures/f1_scores.png` | Per-class F1 grouped bar chart |
| `output/figures/error_breakdown.png` | KRR error category pie chart |
| `output/figures/confusion_matrix.png` | Confusion matrix heatmaps |

---

## Project Structure

```
.
├── main.py                    # CLI entry point (--pipeline, --claim demo mode)
├── config.py                  # Configuration from .env / environment variables
├── models.py                  # Typed data models (Triple, FeverRecord, KRRResult, …)
├── requirements.txt           # Pinned dependencies
├── .env.example               # Environment template
│
├── data/
│   ├── train.jsonl            # Training split (14 claims, 3 label types)
│   ├── test.jsonl             # Test split — evaluation only (6 claims)
│   ├── sample_fever.jsonl     # Original 10-claim sample
│   └── loader.py              # FEVER JSONL loader
│
├── retrieval/
│   └── retriever.py           # Hybrid TF-IDF + sentence-transformers retrieval
│
├── knowledge/
│   ├── extractor.py           # spaCy triple extraction
│   └── graph.py               # Namespaced knowledge graph
│
├── reasoning/
│   └── reasoner.py            # Symbolic KRR reasoning engine
│
├── pipeline/
│   └── krr.py                 # KRR pipeline orchestration
│
├── baseline/
│   ├── llm.py                 # LLM backends (KeywordLLM, HuggingFace, OpenAI, Mock)
│   └── pipeline.py            # Baseline RAG pipeline
│
├── evaluation/
│   └── evaluator.py           # Accuracy, F1, precision, recall, confusion matrix
│
├── scripts/
│   └── generate_graphs.py     # Matplotlib graph generation
│
├── tests/                     # pytest suite — 104 tests, all passing
│
└── .github/
    └── workflows/
        └── test.yml           # GitHub Actions CI
```

---

## Future Work

- **Larger FEVER subset** — validate on the full 19,998-claim dev set for statistically meaningful results
- **Coreference resolution** — resolve pronouns ("It is located…") to their antecedents
- **Multi-hop reasoning** — chain evidence across multiple sentences
- **Neural-symbolic hybrid** — use a fine-tuned NER/RE model for extraction, symbolic rules for reasoning
- **Real LLM baseline** — compare against GPT-4 or a fine-tuned BERT model

---

## Citation

FEVER dataset: Thorne et al., 2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL 2018. [https://fever.ai](https://fever.ai)
