# KRR Fact Verification System

[![Test Project](https://github.com/harsha271199/KRR-new/actions/workflows/test.yml/badge.svg)](https://github.com/harsha271199/KRR-new/actions/workflows/test.yml)

---

## Overview

### The Problem

Large language models frequently hallucinate facts and provide unverifiable claims. Fact verification systems must determine whether a claim is supported or refuted by evidence, but traditional approaches lack transparency and interpretability.

### Why It Matters

In high-stakes domains like journalism, healthcare, and legal systems, we need verifiable, explainable fact-checking systems that can trace their reasoning and provide attribution for every decision.

### The Solution

This project implements a **Knowledge Representation and Reasoning (KRR)** approach to automated fact verification. Instead of relying on black-box neural models, the system uses structured reasoning over symbolic knowledge representations to produce transparent, auditable verdicts with full attribution.

---

## System Architecture

The KRR pipeline follows a clear four-stage process:

```text
Claim → Retrieval → Triple Extraction → Reasoning → Prediction
```

### 1. Retrieval: Hybrid TF-IDF + Sentence Transformers

- **TF-IDF**: Fast lexical matching for keyword overlap
- **Sentence Transformers**: Semantic similarity using `all-MiniLM-L6-v2` embeddings
- **Hybrid scoring**: `0.3 × TF-IDF + 0.7 × semantic` for optimal retrieval
- Retrieves top-K most relevant evidence sentences from the corpus

### 2. Triple Extraction: spaCy-Based Parsing

- Extracts structured `(subject, relation, object)` triples from claims and evidence
- Uses spaCy dependency parsing to handle:
  - Active and passive voice
  - Negation (`not visible`)
  - Comparative claims (`slower than`)
  - Full subject extraction (`speed of light`, not truncated to `speed`)
  - Nested location phrases (`tower in Paris`)

### 3. Reasoning: Support/Refute Counting

- Compares claim triples against evidence triples symbolically
- **Relation normalization**: Maps similar relations (`become → be`, `locate → be`)
- **Subject alias matching**: Handles name variations (`Shakespeare ↔ William Shakespeare`)
- **Negation-aware matching**: Detects contradictions via negation polarity
- **Synonym canonicalization**: Treats `tallest` and `highest` as equivalent
- **Numeric comparative reasoning**: Resolves claims like `X is slower than Y` by extracting and comparing numeric values
- Counts supporting and refuting evidence triples

### 4. Prediction

- **SUPPORTS**: More supporting evidence than refuting
- **REFUTES**: More refuting evidence than supporting
- **NOT ENOUGH INFO**: Tie, equal support/refute, or no matching triples found

---

## Dataset

### FEVER-Style Format

The dataset follows the FEVER, Fact Extraction and VERification, format with three label types:

- **SUPPORTS**: Evidence confirms the claim
- **REFUTES**: Evidence contradicts the claim
- **NOT ENOUGH INFO**: Insufficient evidence to decide

### Train/Test Split

| Split | File | Claims | SUPPORTS | REFUTES | NEI |
|---|---|---:|---:|---:|---:|
| Train | `data/train.jsonl` | 30 | 12 | 10 | 8 |
| Test | `data/test.jsonl` | 15 | 5 | 5 | 5 |

**Evaluation is performed on unseen test data with no data leakage.**

The test set is **perfectly balanced** across all three label classes, with five examples per class. This removes class-imbalance bias from the evaluation metrics. Reasoning rules and synonym tables were developed using the training split only. Test labels are used exclusively for final evaluation.

A larger, balanced dataset improves evaluation stability. Metrics computed over 15 balanced claims are more reliable than metrics computed over the earlier 6-claim version.

---

## Results

### Performance Metrics

| Pipeline | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI |
|---|---:|---:|---:|---:|
| **KRR** | **0.933** | **0.889** | **1.000** | **0.909** |
| Baseline | 0.400 | 0.526 | 0.333 | 0.000 |

### Key Findings

- **KRR outperforms the baseline by 53.3 percentage points** in accuracy, from 0.400 to 0.933.
- **Perfect refutation detection**: KRR achieves F1-REFUTES = 1.000.
- **Strong NEI handling**: KRR achieves F1-NEI = 0.909, while the baseline never predicts NEI.
- **Balanced evaluation**: Results are computed on a balanced test set with 5 SUPPORTS, 5 REFUTES, and 5 NEI claims.
- **Better precision**: KRR avoids the baseline's main failure mode, which is over-predicting SUPPORTS.

### Why KRR Performs Better

- **Semantic retrieval** retrieves correct evidence even when claim and evidence use different vocabulary.
- **Structured reasoning** compares symbolic triples instead of relying only on keyword overlap.
- **Negation awareness** explicitly detects contradictions.
- **Abstention capability** allows the system to predict NOT ENOUGH INFO when evidence is ambiguous.

---

## Key Improvements

### 1. Hybrid Semantic Retrieval

- Combined TF-IDF lexical matching with sentence-transformer semantic retrieval
- Uses 70% weight on semantic similarity for better vocabulary-independent matching
- Improves evidence retrieval when the claim and evidence use different wording

### 2. Improved Subject Extraction

- Extended subject extraction to include prepositional chains
- Example: `speed of light` is preserved instead of being truncated to `speed`
- Critical for comparative claims involving similar entities

### 3. Numeric Reasoning for Comparative Claims

- Adds a dedicated resolver for comparative claims such as `X is slower than Y`
- Extracts numeric evidence values, such as `299,792,458 m/s` and `343 m/s`
- Compares values directly to determine support or refutation

### 4. Improved NOT ENOUGH INFO Handling

- Uses weak-evidence filtering
- Allows ties to map to NOT ENOUGH INFO
- Correctly abstains when evidence is ambiguous or insufficient

---

## Limitations

### 1. Ambiguous Evidence

When evidence contains conflicting location information, such as `located in Coral Sea` and `located in Australia`, the system may abstain because it does not have geographic world knowledge.

### 2. Symbolic Reasoning Limitations

- No multi-hop reasoning across multiple sentences
- No full coreference resolution for pronouns such as `it`
- Cannot infer implicit world knowledge without explicit evidence

### 3. Dependence on Retrieval Quality

If the retriever fails to find relevant evidence, the reasoning module has no reliable signal. Retrieval quality remains the main bottleneck for system performance.

### 4. Small Dataset

Although the test set was expanded to 15 balanced claims, it is still a small FEVER-style subset. The results are suitable for demonstrating the KRR pipeline, but future work should validate the system on larger FEVER dev/test subsets.

---

## How to Run

### Run Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

### Run Full Evaluation

```powershell
.\venv\Scripts\python.exe main.py --pipeline both
```

### Run KRR Only

```powershell
.\venv\Scripts\python.exe main.py --pipeline krr
```

### Run Baseline Only

```powershell
.\venv\Scripts\python.exe main.py --pipeline baseline
```

### Generate Visualization Graphs

```powershell
.\venv\Scripts\python.exe scripts\generate_graphs.py
```

---

## Outputs

| File | Description |
|---|---|
| `output/metrics.json` | Machine-readable metrics including accuracy, F1, precision, recall, and confusion matrix |
| `output/eval_report.txt` | Human-readable evaluation report with confusion matrices |
| `output/confusion_matrix.csv` | Confusion matrices for both pipelines in CSV format |
| `output/krr_results.jsonl` | Per-claim KRR output with triples, counts, verdict, and attribution |
| `output/error_analysis.md` | Per-error breakdown with root-cause analysis |
| `output/figures/` | Visualization graphs for accuracy, F1 scores, confusion matrices, and error breakdown |

---

## Conclusion

This project demonstrates that **Knowledge Representation and Reasoning improves both interpretability and performance** for fact verification tasks.

- **KRR outperforms the baseline** by 53.3 percentage points in test accuracy.
- **Structured reasoning provides transparency** because every prediction includes attribution to evidence triples.
- **Hybrid semantic retrieval is critical** for finding relevant evidence.
- **Balanced evaluation** using 15 test claims provides a more stable and representative evaluation than the earlier 6-claim version, while remaining suitable for demonstration-scale analysis.

The symbolic approach offers a viable alternative to black-box neural models, especially in domains where explainability and auditability are required.

---

## Citation

FEVER dataset: Thorne et al., 2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL 2018. [https://fever.ai](https://fever.ai)
