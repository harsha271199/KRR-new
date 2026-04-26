# Attribution-Aware KRR Fact Verification System

A Python-based system for verifying natural language claims using **Knowledge Representation and Reasoning (KRR)**.

This project focuses on **interpretable fact verification**, where decisions are made using structured reasoning instead of black-box models.

---

## 🔍 Overview

Modern AI systems (like LLMs) often:
- generate fluent but incorrect answers
- lack explainability
- do not show evidence reasoning

This project addresses these issues by:

✔ Converting text into structured triples  
✔ Performing rule-based reasoning  
✔ Providing attribution for every decision  

---

## 🧠 System Architecture

Claim → Retrieval → Triple Extraction → Reasoning → Prediction

### Components

### 🔹 Retriever
- Hybrid approach:
  - TF-IDF
  - Sentence Transformers (MiniLM)

### 🔹 Triple Extractor
- Uses spaCy
- Converts sentences into (subject, relation, object)

### 🔹 Reasoning Module
- Uses support_count, refute_count, irrelevant_count
- Produces final verdict:
  - SUPPORTS
  - REFUTES
  - NOT ENOUGH INFO

---

## 📊 Dataset

- FEVER-style dataset
- Labels:
  - SUPPORTS
  - REFUTES
  - NOT ENOUGH INFO

✔ Train/Test split used  
✔ No data leakage  

---

## 📈 Results (Test Set)

| Pipeline | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI |
|----------|----------|-------------|-------------|--------|
| **KRR (Proposed)** | **0.833** | 0.800 | 1.000 | 0.667 |
| Baseline (Keyword RAG) | 0.667 | 0.750 | 0.667 | 0.000 |

---

## 🔥 Key Improvements

- Hybrid retrieval (TF-IDF + semantic embeddings)
- Better subject extraction
- Numeric reasoning
- Improved NEI detection

---

## ⚠️ Limitations

- Ambiguous evidence → NOT ENOUGH INFO
- Symbolic reasoning lacks deep world knowledge

---

## ▶️ How to Run

Run tests:
python -m pytest tests/

Run pipeline:
python main.py --pipeline both

Generate graphs:
python scripts/generate_graphs.py

---

## 📂 Outputs

- metrics.json
- eval_report.txt
- confusion_matrix.csv
- figures/

---

## 🎯 Conclusion

The KRR system achieves **83.3% accuracy**, outperforming the baseline by ~16%.

✔ Structured reasoning improves reliability  
✔ KRR provides explainability  

