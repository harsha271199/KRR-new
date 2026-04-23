# Attribution-Aware KRR Fact Verification System

A Python 3.10+ pipeline that verifies natural language claims against the [FEVER dataset](https://fever.ai/) using explicit, traceable Knowledge Representation and Reasoning (KRR). Unlike black-box LLM approaches, the KRR pipeline converts evidence into structured knowledge triples, applies deterministic reasoning rules, and produces a verdict with full attribution to the specific triples that drove the decision.

The system runs two parallel pipelines and evaluates them side-by-side:

- **KRR Pipeline** — Retrieve → Extract Triples → Build Knowledge Graph → Reason → Verdict + Attribution
- **Baseline RAG Pipeline** — Retrieve → LLM Classify → Verdict

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Configuration (.env)](#configuration-env)
5. [Running the Pipelines](#running-the-pipelines)
6. [Running the Evaluator](#running-the-evaluator)
7. [Running Tests](#running-tests)
8. [Project Structure](#project-structure)

---

## Architecture Overview

```
FEVER Dataset ──► DataLoader ──► Retriever (TF-IDF / BM25)
                                      │
                    ┌─────────────────┴──────────────────┐
                    │ KRR Pipeline                        │ Baseline RAG Pipeline
                    │                                     │
                    ▼                                     ▼
             TripleExtractor                        PromptBuilder
             (claim + evidence)                          │
                    │                                     ▼
                    ▼                               LLMBackend
             KnowledgeGraph                    (HuggingFace / OpenAI)
                    │                                     │
                    ▼                                     ▼
             ReasoningModule                       BaselineResult
             (support/refute/irrelevant)
                    │
                    ▼
               KRRResult
          (verdict + attribution)
                    │
                    └──────────────┬──────────────────────┘
                                   ▼
                               Evaluator
                    (accuracy + per-class F1 comparison)
```

**Key design goals:**

- **Transparency** — every verdict is traceable to specific evidence triples.
- **Modularity** — each component has a clean, typed interface and is independently testable.
- **Configurability** — all paths, model choices, and hyperparameters are externalized via `.env` / environment variables.
- **Comparability** — both pipelines share the same retrieval layer so differences are attributable to the reasoning approach alone.

---

## Prerequisites

- Python 3.10 or later
- `pip` (comes with Python)
- A FEVER dataset JSONL file (e.g., `train.jsonl` or `paper_dev.jsonl`) — download from [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html)
- An evidence corpus file (plain text, one sentence per line, or the FEVER wiki-pages dump)
- *(Optional)* An OpenAI API key if you want to use the OpenAI LLM backend

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

> If you configure a different `SPACY_MODEL` in your `.env`, download that model instead:
> ```bash
> python -m spacy download <model-name>
> ```

### 4. Configure the environment

Copy the example below into a file named `.env` in the project root and fill in the values for your environment:

```dotenv
# --- Required ---
FEVER_DATASET_PATH=/path/to/fever/train.jsonl
EVIDENCE_CORPUS_PATH=/path/to/evidence_corpus.txt
EVAL_OUTPUT_PATH=./output/eval_report.txt

# --- Optional (defaults shown) ---
RETRIEVAL_TOP_K=5
NLP_ENGINE=spacy
LLM_BACKEND=huggingface
HF_MODEL_NAME=google/flan-t5-base
OPENAI_MODEL=gpt-3.5-turbo
# LLM_API_KEY=sk-...          # Required only when LLM_BACKEND=openai
MAX_RECORDS=                   # Leave blank to load all records
INDEX_PATH=                    # Leave blank to rebuild index on every run
SPACY_MODEL=en_core_web_sm
JSON_OUTPUT_PATH=              # Leave blank to skip per-claim JSON output
```

---

## Configuration (.env)

All configuration is loaded from the `.env` file and/or OS environment variables. **Environment variables take precedence over `.env` file values.**

### Required keys

| Key | Description | Example |
|-----|-------------|---------|
| `FEVER_DATASET_PATH` | Path to the FEVER JSONL dataset file | `/data/fever/train.jsonl` |
| `EVIDENCE_CORPUS_PATH` | Path to the evidence corpus file | `/data/fever/corpus.txt` |
| `EVAL_OUTPUT_PATH` | Path where the evaluation report will be written | `./output/eval_report.txt` |

If any required key is missing, the system raises a `ConfigError` identifying the missing key and exits before initializing any pipeline component.

### Optional keys

| Key | Default | Description |
|-----|---------|-------------|
| `RETRIEVAL_TOP_K` | `5` | Number of evidence sentences to retrieve per claim |
| `NLP_ENGINE` | `spacy` | NLP engine for triple extraction: `spacy` or `openie` |
| `LLM_BACKEND` | `huggingface` | LLM backend for the baseline pipeline: `huggingface` or `openai` |
| `LLM_API_KEY` | *(none)* | OpenAI API key — **required when `LLM_BACKEND=openai`** |
| `HF_MODEL_NAME` | `google/flan-t5-base` | HuggingFace model identifier |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `MAX_RECORDS` | *(all)* | Maximum number of FEVER records to load (useful for quick runs) |
| `INDEX_PATH` | *(none)* | Path to persist/load the retrieval index (pickle). If set and the file exists, the index is loaded instead of rebuilt |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model name to load for triple extraction |
| `JSON_OUTPUT_PATH` | *(none)* | Path for per-claim KRR results in newline-delimited JSON format |

> **Security note:** `LLM_API_KEY` is never logged or printed by the system.

---

## Running the Pipelines

All pipelines are launched through `main.py`. The `--pipeline` flag selects which pipeline(s) to run.

### Run the KRR pipeline only

```bash
python main.py --pipeline krr
```

### Run the Baseline RAG pipeline only

```bash
python main.py --pipeline baseline
```

### Run both pipelines (default)

```bash
python main.py --pipeline both
```

or simply:

```bash
python main.py
```

### Additional options

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline {krr,baseline,both}` | `both` | Which pipeline(s) to run |
| `--env-file PATH` | `.env` | Path to the `.env` configuration file |
| `--output-json PATH` | *(from config)* | Override `JSON_OUTPUT_PATH` — write per-claim KRR results to this file |

**Examples:**

```bash
# Use a custom .env file
python main.py --pipeline krr --env-file .env.production

# Write per-claim JSON output to a specific file
python main.py --pipeline both --output-json ./output/results.jsonl

# Quick test run: load only 100 records, KRR pipeline, custom env
python main.py --pipeline krr --env-file .env.dev
# (set MAX_RECORDS=100 in .env.dev)
```

### Per-claim output format (stdout)

For each claim the KRR pipeline prints a block like:

```
============================================================
CLAIM      : Albert Einstein was born in Germany.

EVIDENCE   :
  [1] Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire.
  ...

CLAIM TRIPLES:
  albert einstein|bear|germany

EVIDENCE TRIPLES:
  albert einstein|bear|ulm

SUPPORT COUNT  : 0
REFUTE COUNT   : 1
IRRELEVANT COUNT: 0
VERDICT        : REFUTES

ATTRIBUTION:
  albert einstein|bear|ulm
============================================================
```

---

## Running the Evaluator

The evaluator runs automatically at the end of every `main.py` invocation. It computes accuracy and per-class F1 for both pipelines and writes a comparison report to `EVAL_OUTPUT_PATH`.

**Sample report output:**

```
=== Evaluation Report ===
Total claims: 1000 | Skipped: 3
Label distribution: SUPPORTS=334, REFUTES=333, NOT ENOUGH INFO=333

Pipeline          | Accuracy | F1-SUPPORTS | F1-REFUTES | F1-NEI
------------------|----------|-------------|------------|-------
KRR               |   0.712  |    0.731    |   0.698    | 0.706
Baseline RAG      |   0.681  |    0.703    |   0.665    | 0.674
```

The report is also saved to the file specified by `EVAL_OUTPUT_PATH`.

---

## Running Tests

The test suite uses `pytest` and covers all core modules without requiring network access or external API calls.

### Run all tests

```bash
pytest
```

### Run tests with verbose output

```bash
pytest -v
```

### Run a specific test file

```bash
pytest tests/test_reasoning.py -v
pytest tests/test_knowledge_graph.py -v
pytest tests/test_retriever.py -v
```

### Run tests matching a keyword

```bash
pytest -k "triple" -v
```

> **Note:** Tests for `TripleExtractor` require the spaCy model to be installed (`python -m spacy download en_core_web_sm`). All other tests run without any external dependencies.

---

## Project Structure

```
.
├── main.py                    # CLI entry point (argparse)
├── config.py                  # Configuration loading from .env / env vars
├── models.py                  # Shared data models (Triple, FeverRecord, etc.)
├── requirements.txt           # Pinned Python dependencies
├── .env                       # Local configuration (not committed to VCS)
│
├── data/
│   └── loader.py              # FEVER dataset loading and preprocessing
│
├── retrieval/
│   └── retriever.py           # TF-IDF / BM25 evidence retrieval
│
├── knowledge/
│   ├── extractor.py           # spaCy-based triple extraction
│   └── graph.py               # Namespaced in-memory knowledge graph
│
├── reasoning/
│   └── reasoner.py            # Triple comparison and verdict assignment
│
├── pipeline/
│   └── krr.py                 # KRR pipeline orchestration
│
├── baseline/
│   ├── llm.py                 # LLM backends (HuggingFace, OpenAI, Mock)
│   └── pipeline.py            # Baseline RAG pipeline
│
├── evaluation/
│   └── evaluator.py           # Accuracy + F1 evaluation and report generation
│
└── tests/
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_extractor.py
    ├── test_knowledge_graph.py
    ├── test_models.py
    ├── test_reasoning.py
    └── test_retriever.py
```
