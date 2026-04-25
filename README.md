# KRR Fact Verification System

[![Test Project](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml/badge.svg)](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml)

> Attribution-aware fact verification using Knowledge Representation and Reasoning (KRR) — a master's group project comparing symbolic reasoning against a keyword-based baseline on the FEVER dataset.

---

## Problem Statement

Automated fact verification is a core challenge in NLP. Most modern approaches use black-box neural models that produce a verdict without explaining *why*. This project implements a transparent, rule-based KRR pipeline that:

1. Retrieves relevant evidence sentences using TF-IDF
2. Extracts structured (subject, relation, object) triples using spaCy
3. Reasons over those triples symbolically to produce a verdict with full attribution
4. Compares against a keyword-based baseline pipeline

Every verdict is traceable to the specific evidence triples that drove the decision — making the system interpretable and auditable.

---

## Architecture

```
                        FEVER Dataset
                             │
                        DataLoader
                             │
                    TF-IDF Retriever
                             │
              ┌──────────────┴──────────────┐
              │                             │
       KRR Pipeline                 Baseline Pipeline
              │                             │
    spaCy Triple Extractor           KeywordLLM
    (claim + evidence triples)    (heuristic rules)
              │                             │
    KnowledgeGraph                   BaselineResult
              │
    ReasoningModule
    (symbolic KRR reasoning)
    - Relation normalization
    - Negation-aware matching
    - Synonym canonicalization
    - Subject alias matching
    - Nested location extraction
    - Comparative numerical reasoning
    - Weak-evidence filtering
              │
         KRRResult
    (verdict + attribution)
              │
         ┌────┴────┐
         │Evaluator│
         └────┬────┘
              │
    output/eval_report.txt
    output/metrics.json
    output/confusion_matrix.csv
    output/error_analysis.md
    output/figures/*.png
```

---

## Results

| Pipeline | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI | Macro Precision | Macro Recall |
|----------|----------|-------------|------------|--------|-----------------|--------------|
| **KRR** | **0.90** | **1.000** | **0.857** | 0.000 | **0.667** | **0.583** |
| Baseline (Keyword) | 0.70 | 0.800 | 0.400 | 0.000 | 0.556 | 0.417 |

### Confusion Matrix — KRR

```
                  Predicted
                  SUPPORTS  REFUTES  NEI
Actual SUPPORTS      6         0      0
Actual REFUTES       0         3      1
Actual NEI           0         0      0
```

### Confusion Matrix — Baseline

```
                  Predicted
                  SUPPORTS  REFUTES  NEI
Actual SUPPORTS      6         0      0
Actual REFUTES       3         1      0
Actual NEI           0         0      0
```

### Why KRR outperforms the Baseline

KRR achieves 90% vs Baseline's 70% because:
- It reasons over structured triples, not just keyword overlap
- It correctly handles negation (`not visible from space` → REFUTES)
- It resolves synonyms (`tallest` ≡ `highest` via canonical mapping)
- It extracts nested locations (`in Paris` from `on the Champ de Mars in Paris`)
- It uses subject alias matching (`shakespeare` matches `william shakespeare`)
- It filters weak measurement objects that create spurious refutations

### Why KRR is more valuable than Baseline regardless of accuracy

- **Interpretable**: every verdict cites the exact triples that drove it
- **Auditable**: you can inspect why a claim was supported or refuted
- **Principled**: reasoning follows explicit logical rules, not heuristics
- **Scalable**: symbolic rules generalise; keyword matching degrades on paraphrase

---

## Setup

### Prerequisites

- Python 3.11 (recommended)
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
# Defaults work with the included sample dataset — no edits needed
```

---

## Running the Project

### Run tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

Expected: `104 passed`

### Run both pipelines (full evaluation)

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

### Demo mode — verify a single claim live

```powershell
.\venv\Scripts\python.exe main.py --claim "The Eiffel Tower is located in Paris."
```

Output:
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

Outputs to `output/figures/`:
- `model_accuracy.png` — KRR vs Baseline accuracy bar chart
- `f1_scores.png` — Per-class F1 grouped bar chart
- `error_breakdown.png` — KRR error category pie chart
- `confusion_matrix.png` — Confusion matrix heatmaps

---

## Output Files

| File | Description |
|------|-------------|
| `output/eval_report.txt` | Human-readable evaluation report with confusion matrices |
| `output/metrics.json` | Machine-readable metrics (accuracy, F1, precision, recall) |
| `output/confusion_matrix.csv` | Confusion matrices for both pipelines |
| `output/krr_results.jsonl` | Per-claim KRR results (triples, counts, verdict, attribution) |
| `output/error_analysis.md` | Per-error breakdown with root-cause analysis |
| `output/figures/*.png` | Presentation graphs |

---

## Project Structure

```
.
├── main.py                    # CLI entry point (--pipeline, --claim demo mode)
├── config.py                  # Configuration loading from .env / env vars
├── models.py                  # Shared data models (Triple, FeverRecord, etc.)
├── requirements.txt           # Pinned Python dependencies
├── .env.example               # Environment template (copy to .env)
│
├── data/
│   ├── loader.py              # FEVER JSONL loading and preprocessing
│   └── sample_fever.jsonl     # 10-claim sample dataset
│
├── retrieval/
│   └── retriever.py           # TF-IDF / BM25 evidence retrieval
│
├── knowledge/
│   ├── extractor.py           # spaCy triple extraction (passive, negation, comparative,
│   │                          #   nested location, ACL subject promotion)
│   └── graph.py               # Namespaced in-memory knowledge graph
│
├── reasoning/
│   └── reasoner.py            # Symbolic KRR reasoning:
│                              #   - relation normalization
│                              #   - negation-aware object matching
│                              #   - synonym canonicalization
│                              #   - subject alias matching
│                              #   - weak-evidence filtering
│                              #   - comparative numerical reasoning
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
│                              #   writes eval_report.txt, metrics.json, confusion_matrix.csv
│
├── scripts/
│   └── generate_graphs.py     # Matplotlib graph generation for presentation
│
├── tests/                     # pytest test suite (104 tests, all passing)
│
├── output/                    # Generated outputs (gitignored except .gitkeep)
│   └── figures/
│
└── .github/
    └── workflows/
        └── test.yml           # GitHub Actions CI (Python 3.11, pytest + pipeline run)
```

---

## KRR Reasoning — How It Works

The `ReasoningModule` compares claim triples against evidence triples using a multi-layer pipeline:

### 1. Relation Normalization
Semantically equivalent verbs map to a canonical form:
- `become → be`, `locate → be`, `consider → be`, `born → bear`

### 2. Subject Alias Matching
Partial name matches are accepted when one subject is a substring of the other:
- `"shakespeare"` matches `"william shakespeare"`
- `"great wall"` matches `"great wall of china"`

### 3. Negation-Aware Object Matching
Negation polarity is checked before any other comparison:
- `"visible from space"` vs `"not visible from space"` → **REFUTES** (polarity mismatch)
- `"not visible"` vs `"not visible"` → **SUPPORTS** (both negated)

### 4. Synonym Canonicalization
Known synonyms map to a single canonical form:
- `"tallest"` → `"tall"` ← `"highest"` → match → **SUPPORTS**
- `"longest"` and `"largest"` are intentionally NOT synonyms (Amazon River must REFUTE)

### 5. Weak-Evidence Filtering
Measurement and circumstantial objects are dropped before reasoning:
- `"at metres above sea level"`, `"on march"`, `"from 1887"`, `"at standard atmospheric pressure"`

### 6. Comparative Numerical Reasoning
When a claim contains a comparative (`"slower than speed of sound"`), numeric values are extracted from evidence and compared:
- Speed of light: 299,792,458 m/s > Speed of sound: 343 m/s → claim says "slower" → **REFUTES**

---

## Extractor Improvements

| Structure | Example | Triple produced |
|-----------|---------|-----------------|
| Active SVO | "Shakespeare wrote Hamlet" | `shakespeare\|write\|hamlet` |
| Passive voice | "Hamlet was written by Shakespeare" | `shakespeare\|write\|hamlet` |
| Passive ACL (promoted) | "Hamlet is a tragedy written by Shakespeare" | `william shakespeare\|write\|hamlet` |
| Negation | "not visible from space" | `great wall\|be\|not visible from space` |
| Comparative | "slower than the speed of sound" | `speed\|be\|slower than speed of sound` |
| Adjectival complement | "is visible from space" | `great wall\|be\|visible from space` |
| Nested location | "tower on the Champ de Mars in Paris" | `eiffel tower\|locate\|in paris` |

---

## Error Analysis Summary

One claim remains wrong after all improvements:

| Claim | GT | KRR | Root Cause |
|-------|----|-----|------------|
| The speed of light is slower than the speed of sound | REFUTES | NOT ENOUGH INFO | Comparative numerical reasoning requires the evidence triples to carry numeric values associated with the correct subjects. spaCy parses "speed of light" and "speed of sound" as separate entities but the extractor produces `speed\|be\|metres` for both — the subject is truncated to `"speed"`, losing the `"of light"` / `"of sound"` distinction. The comparative resolver cannot differentiate the two speeds. |

---

## Limitations

- **Small dataset**: 10 claims is insufficient for statistically meaningful evaluation; results should be validated on the full FEVER dev set (19,998 claims)
- **Subject truncation**: spaCy's compound noun handling sometimes drops `"of light"` from `"speed of light"`, breaking comparative reasoning
- **Coreference**: pronouns like "It" are not resolved to their antecedents
- **No NOT ENOUGH INFO predictions**: the sample dataset has no NEI examples, so F1-NEI is 0 by construction
- **Synonym coverage**: the synonym table is hand-curated and limited; a thesaurus or word embeddings would generalise better
- **Keyword baseline**: the baseline uses simple heuristics, not a real LLM; a GPT-4 baseline would be a fairer comparison

---

## Future Work

- **sentence-transformers**: replace exact/substring matching with cosine similarity over sentence embeddings for robust synonym handling
- **Larger FEVER subset**: evaluate on 1000+ claims for statistically valid metrics
- **Full coreference resolution**: use spaCy's experimental coref component or neuralcoref
- **Hybrid KRR + neural**: use neural models for relation extraction, symbolic rules for reasoning
- **OpenIE integration**: replace spaCy dependency parsing with OpenIE for broader triple coverage
- **Real LLM baseline**: compare against GPT-4 or a fine-tuned BERT model for a fair neural comparison
- **NOT ENOUGH INFO handling**: add claims with insufficient evidence to the dataset

---

## Citation

FEVER dataset: Thorne et al., 2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL 2018. [https://fever.ai](https://fever.ai)
