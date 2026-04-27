# KRR Fact Verification System

[![Test Project](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml/badge.svg)](https://github.com/harsha271199/dard-e-Disco/actions/workflows/test.yml)

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

```
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
  - Negation ("not visible")
  - Comparative claims ("slower than")
  - Full subject extraction ("speed of light" not truncated to "speed")
  - Nested location phrases ("tower in Paris")

### 3. Reasoning: Support/Refute Counting
- Compares claim triples against evidence triples symbolically
- **Relation normalization**: Maps similar relations (`become → be`, `locate → be`)
- **Subject alias matching**: Handles name variations ("Shakespeare" ↔ "William Shakespeare")
- **Negation-aware matching**: Detects contradictions via negation polarity
- **Synonym canonicalization**: Treats "tallest" and "highest" as equivalent
- **Numeric comparative reasoning**: Resolves claims like "X is slower than Y" by extracting and comparing numeric values
- Counts supporting and refuting evidence triples

### 4. Prediction
- **SUPPORTS**: More supporting evidence than refuting
- **REFUTES**: More refuting evidence than supporting
- **NOT ENOUGH INFO**: Tie (equal support/refute) or no matching triples found

---

## Dataset

### FEVER-Style Format
The dataset follows the FEVER (Fact Extraction and VERification) format with three label types:
- **SUPPORTS**: Evidence confirms the claim
- **REFUTES**: Evidence contradicts the claim
- **NOT ENOUGH INFO**: Insufficient evidence to decide

### Train/Test Split

| Split | File | Claims | SUPPORTS | REFUTES | NEI |
|-------|------|--------|----------|---------|-----|
| Train | `data/train.jsonl` | 30 | 12 | 10 | 8 |
| Test | `data/test.jsonl` | 15 | 5 | 5 | 5 |

**Evaluation is performed on unseen test data (no data leakage).**

The test set is **perfectly balanced** across all three label classes (5 each), which eliminates class-imbalance bias from the evaluation metrics. Reasoning rules and synonym tables were developed using the training split only. Test labels are never seen during development and are used exclusively for final evaluation.

A larger, balanced dataset improves evaluation stability — metrics computed over 15 balanced claims are more reliable than those over 6 imbalanced claims.

---

## Results

### Performance Metrics

| Pipeline | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI |
|----------|----------|-------------|------------|--------|
| **KRR** | **0.933** | **0.889** | **1.000** | **0.909** |
| Baseline | 0.400 | 0.526 | 0.333 | 0.000 |

### Key Findings
- **KRR outperforms baseline by ~53.3%** (0.933 vs 0.400 accuracy)
- **Perfect refutation detection**: KRR achieves F1-REFUTES = 1.000 — all 5 REFUTES claims correctly identified
- **Strong NEI handling**: KRR achieves F1-NEI = 0.909, while baseline never predicts NEI (F1 = 0.000)
- **Balanced evaluation**: Results computed over a perfectly balanced test set (5 SUPPORTS / 5 REFUTES / 5 NEI), eliminating class-imbalance bias
- **Better precision**: KRR avoids over-predicting SUPPORTS, which is the baseline's primary failure mode

### Why KRR Performs Better
- **Semantic retrieval**: Hybrid TF-IDF + sentence embeddings retrieve correct evidence even with vocabulary mismatch
- **Structured reasoning**: Symbolic triple comparison provides more reliable signal than keyword matching
- **Negation awareness**: Explicit negation handling correctly identifies contradictions
- **Abstention capability**: Tie-breaking rule allows the system to predict NOT ENOUGH INFO when evidence is ambiguous

---

## Key Improvements

The following enhancements were implemented to achieve the reported performance:

### 1. Hybrid Semantic Retrieval
- Combined TF-IDF (lexical) with sentence-transformers (semantic)
- 70% weight on semantic similarity for better vocabulary-independent matching
- Retrieves correct evidence even when claim and evidence use different words

### 2. Improved Subject Extraction
- Extended subject extraction to include prepositional chains
- Example: "speed of light" (not truncated to "speed")
- Critical for distinguishing similar entities in comparative claims

### 3. Numeric Reasoning for Comparative Claims
- Dedicated resolver for comparative claims ("X is slower than Y")
- Extracts numeric values from evidence (e.g., 299,792,458 m/s vs 343 m/s)
- Compares values to determine support/refute verdict

### 4. Improved NOT ENOUGH INFO Handling
- Weak-evidence filter exempts numeric values
- Tie-breaking rule: equal support and refute counts → NOT ENOUGH INFO
- System correctly abstains when evidence is ambiguous

---

## Limitations

### 1. Ambiguous Evidence → NOT ENOUGH INFO
When evidence contains conflicting information (e.g., "located in Coral Sea" vs "located in Australia"), the system abstains but cannot resolve the ambiguity without geographic inference.

### 2. Symbolic Reasoning Limitations
- No multi-hop reasoning across multiple sentences
- No coreference resolution (pronouns like "it" are not resolved)
- Cannot perform inference over implicit world knowledge

### 3. Dependence on Retrieval Quality
If the retriever fails to find relevant evidence, the reasoning module has no signal to work with. Retrieval is the primary bottleneck for system performance.

### 4. Small Dataset
Although the test set was expanded to 15 balanced claims, it is still a small FEVER-style subset. The results are suitable for demonstrating the KRR pipeline, but future work should validate the system on larger FEVER dev/test subsets.

---

## How to Run

### Run Tests
```powershell
.\venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

### Run Full Evaluation (Both Pipelines)
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

The system generates the following output files:

| File | Description |
|------|-------------|
| `output/metrics.json` | Machine-readable metrics (accuracy, F1, precision, recall, confusion matrix) |
| `output/eval_report.txt` | Human-readable evaluation report with confusion matrices |
| `output/confusion_matrix.csv` | Confusion matrices for both pipelines in CSV format |
| `output/krr_results.jsonl` | Per-claim KRR output with triples, counts, verdict, and attribution |
| `output/error_analysis.md` | Per-error breakdown with root-cause analysis |
| `output/figures/` | Visualization graphs (accuracy, F1 scores, confusion matrices, error breakdown) |

---

## Conclusion

This project demonstrates that **Knowledge Representation and Reasoning (KRR) improves both interpretability and performance** for fact verification tasks:

- **KRR outperforms the baseline** by 53.3 percentage points on test accuracy (0.933 vs 0.400)
- **Structured reasoning provides transparency**: Every prediction includes attribution to specific evidence triples
- **Retrieval quality is the key factor**: Hybrid semantic retrieval is critical for finding relevant evidence
- **Balanced evaluation**: A 15-claim balanced test set provides a more stable and representative evaluation than the earlier 6-claim version, while remaining suitable for demonstration-scale analysis.

The symbolic approach offers a viable alternative to black-box neural models, especially in domains where explainability and auditability are required.

---

## Citation

FEVER dataset: Thorne et al., 2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL 2018. [https://fever.ai](https://fever.ai)
