# Implementation Plan: KRR Fact Verification System

## Overview

Implement the Attribution-Aware KRR Fact Verification System in Python 3.10+. Tasks follow the data flow: project scaffold → data models → config → data loading → retrieval → triple extraction → knowledge graph → reasoning → KRR pipeline → baseline pipeline → evaluation → CLI entry point → documentation. Each task builds on the previous and ends with all components wired together.

## Tasks

- [x] 1. Scaffold project structure and install dependencies
  - Create the directory layout: `data/`, `retrieval/`, `knowledge/`, `reasoning/`, `pipeline/`, `baseline/`, `evaluation/`, `tests/`
  - Add `__init__.py` files to each package directory
  - Create `requirements.txt` with pinned versions for: `spacy==3.7.4`, `scikit-learn==1.4.2`, `rank-bm25==0.2.2`, `python-dotenv==1.0.1`, `openai==1.30.1`, `transformers==4.41.2`, `torch==2.3.1`, `hypothesis==6.103.1`, `pytest==8.2.2`
  - _Requirements: 10.5, 11.1_

- [x] 2. Implement data models (`models.py`)
  - [x] 2.1 Implement `Triple`, `FeverRecord`, `KnowledgeRecord`, `ReasoningResult`, `KRRResult`, `BaselineResult`, and `EvaluationReport` dataclasses with typed fields as specified in design sections 3.1–3.7
  - Implement `Triple.to_canonical_string()` and `Triple.from_canonical_string()` using pipe-delimited format with `split("|", 2)`
  - Implement `Triple.to_dict()` and `Triple.from_dict()`
  - Enforce normalization invariant: all `Triple` fields must be lowercase and stripped
  - _Requirements: 3.1, 3.4, 4.1, 4.5, 5.1–5.7, 6.1, 7.1, 8.1_

  - [ ] 2.2 Write property test for Triple round-trip (Property 1)
    - **Property 1: Triple round-trip consistency**
    - For any valid `Triple`, `Triple.from_canonical_string(t.to_canonical_string()) == t`
    - Use `hypothesis` to generate arbitrary normalized (lowercase, stripped) subject/relation/object strings
    - **Validates: Requirements 3.6**

  - [x] 2.3 Write unit tests for `Triple` and data models
    - Test `to_canonical_string` / `from_canonical_string` with pipe characters in the object field
    - Test `from_canonical_string` raises `ValueError` on malformed input
    - Test `Triple` equality and hashability (frozen dataclass)
    - Test `to_dict` / `from_dict` round-trip for `Triple`
    - _Requirements: 3.1, 3.4, 11.2_

- [x] 3. Implement `Config` (`config.py`)
  - [x] 3.1 Implement `Config.from_env()` loading values from `.env` file and OS environment variables using `python-dotenv`, with env vars taking precedence
  - Validate all required keys (`FEVER_DATASET_PATH`, `EVIDENCE_CORPUS_PATH`, `EVAL_OUTPUT_PATH`) and raise a descriptive `ConfigError` identifying the missing key if any are absent
  - Apply defaults for optional keys per design section 4.8
  - Never log or print `LLM_API_KEY` value
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 3.2 Write unit tests for `Config`
    - Test that missing required keys raise `ConfigError` with the key name in the message
    - Test that env vars override `.env` file values
    - Test that `LLM_API_KEY` is not present in any log output
    - Test defaults are applied for optional keys
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 11.2_

- [x] 4. Implement `DataLoader` (`data/loader.py`)
  - [x] 4.1 Implement `DataLoader.load()` to read FEVER JSONL line-by-line, parse each line as JSON, flatten nested evidence into a list of strings, normalize labels to uppercase, and stop after `max_records` if set
  - Raise `FileNotFoundError` with the missing path when the dataset file is absent
  - Skip and log a warning (with record index) for any line that raises `json.JSONDecodeError` or is missing required fields
  - Implement `DataLoader.load_corpus()` returning deduplicated evidence sentences across all loaded records
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 4.2 Write unit tests for `DataLoader`
    - Test loading a small in-memory JSONL fixture with valid records
    - Test `FileNotFoundError` is raised for a missing path
    - Test malformed records are skipped and a warning is logged
    - Test `max_records` limits the number of loaded records
    - Test `load_corpus()` deduplicates evidence sentences
    - _Requirements: 1.1–1.5, 11.2_

- [x] 5. Implement `Retriever` (`retrieval/retriever.py`)
  - [x] 5.1 Implement `Retriever.build_index()` using `TfidfVectorizer` + cosine similarity (default) and `BM25Okapi` (when configured), with optional pickle persistence to `config.index_path`
  - Implement `Retriever.load_index()` to restore a pre-built index from `config.index_path`
  - Implement `Retriever.retrieve()` returning top-K evidence sentences; return `[]` and log a warning when all scores are zero
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 5.2 Write unit tests for `Retriever`
    - Test `build_index` + `retrieve` with a small in-memory corpus (no FEVER dataset required)
    - Test that an empty result set returns `[]` and logs a warning
    - Test index persistence: build, save, reload, and verify retrieval results match
    - Test `retrieve` completes within 2 seconds for a 1000-sentence corpus
    - _Requirements: 2.1–2.5, 11.3_

- [x] 6. Implement `TripleExtractor` (`knowledge/extractor.py`)
  - [x] 6.1 Implement `TripleExtractor.extract()` using spaCy dependency parsing: identify `nsubj`/`nsubjpass` → subject, root verb → relation, `dobj`/`attr`/`prep+pobj` → object; include compound nouns and adjectival modifiers in span text
  - Apply `normalize()` (lowercase + strip) to all triple fields before constructing `Triple` objects
  - Return `[]` and log a debug message for sentences that yield no triples
  - Implement `TripleExtractor.extract_batch()` as a loop over `extract()`
  - Raise `TypeError` if input is not a `str`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 11.5_

  - [x] 6.2 Write property test for Triple round-trip via extractor (Property 1 — extractor path)
    - **Property 1: Round-trip consistency through extractor**
    - For any sentence that yields at least one triple, re-parsing each triple's canonical string produces an equivalent `Triple`
    - Implemented in `tests/test_extractor.py` as `TestTripleRoundTripViaExtractor` using fixed sentences (spaCy mocked)
    - **Validates: Requirements 3.6**

  - [x] 6.3 Write unit tests for `TripleExtractor`
    - Test extraction on a known sentence with a clear SVO structure
    - Test that normalization is applied (uppercase input → lowercase triple fields)
    - Test that sentences with no extractable triples return `[]`
    - Test `extract_batch` returns one list per input sentence
    - Test `TypeError` is raised for non-string input
    - _Requirements: 3.1–3.5, 11.2, 11.5_

- [x] 7. Implement `KnowledgeGraph` (`knowledge/graph.py`)
  - [x] 7.1 Implement `KnowledgeGraph.add_triple()` storing `KnowledgeRecord` objects in a `dict[str, list[KnowledgeRecord]]` keyed by namespace, with deduplication via a per-namespace set of `(subject, relation, object)` tuples
  - Implement `KnowledgeGraph.get_triples()` supporting filtering by `namespace`, `subject`, and `relation` (combinable)
  - Implement `KnowledgeGraph.to_dict()` and `KnowledgeGraph.from_dict()` using the JSON format from design section 4.4
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 7.2 Write property test for KnowledgeGraph round-trip (Property 2)
    - **Property 2: KnowledgeGraph JSON round-trip**
    - For any valid `KnowledgeGraph`, `KnowledgeGraph.from_dict(kg.to_dict())` produces an equivalent graph (same triples in same namespaces)
    - Use `hypothesis` to generate arbitrary sets of normalized triples across `"claim"` and `"evidence"` namespaces
    - **Validates: Requirements 4.6**

  - [x] 7.3 Write unit tests for `KnowledgeGraph`
    - Test that duplicate triples in the same namespace are deduplicated
    - Test that identical triples in different namespaces are stored separately
    - Test `get_triples()` with no filters returns all triples
    - Test `get_triples(namespace=...)`, `get_triples(subject=...)`, and combined filters
    - Test `to_dict()` / `from_dict()` preserves all fields and namespace assignments
    - _Requirements: 4.1–4.5, 11.2_

- [x] 8. Implement `ReasoningModule` (`reasoning/reasoner.py`)
  - [x] 8.1 Implement `ReasoningModule.reason()` following the algorithm in design section 4.5: iterate evidence triples, match against claim triples by subject+relation, count support/refute/irrelevant, assign verdict by strict inequality, return `ReasoningResult` with attribution list
  - Handle empty claim or evidence triple sets by returning `NOT ENOUGH INFO` with zero counts and empty attribution
  - Apply tie-breaking: `support_count == refute_count > 0` → `NOT ENOUGH INFO`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 8.2 Write unit tests for `ReasoningModule`
    - Test `SUPPORTS` verdict when support_count > refute_count > 0
    - Test `REFUTES` verdict when refute_count > support_count > 0
    - Test `NOT ENOUGH INFO` when both counts are zero
    - Test `NOT ENOUGH INFO` on tie (support_count == refute_count > 0)
    - Test `NOT ENOUGH INFO` when claim triples are empty
    - Test `NOT ENOUGH INFO` when evidence triples are empty
    - Test attribution list contains exactly the triples that drove the verdict
    - _Requirements: 5.1–5.9, 11.2_

- [ ] 9. Checkpoint — core modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement `KRRPipeline` (`pipeline/krr.py`)
  - [x] 10.1 Implement `KRRPipeline.run()` invoking `Retriever → TripleExtractor (claim + evidence) → KnowledgeGraph → ReasoningModule` in order, assembling a `KRRResult` with all intermediate outputs
  - Wrap the entire pipeline body in a try/except; on any unhandled exception log the claim and stage name and return a `KRRResult` with `verdict="NOT ENOUGH INFO"`, empty attribution, and the error message in `error`
  - Ensure no shared mutable state between calls (fresh `KnowledgeGraph` per claim)
  - Implement JSON output to `config.json_output_path` (append per-claim dict) and human-readable stdout output
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 10.1, 10.2, 10.3_

  - [ ] 10.2 Write unit tests for `KRRPipeline`
    - Test end-to-end run with mock `Retriever`, `TripleExtractor`, and `ReasoningModule`
    - Test that a stage exception returns `NOT ENOUGH INFO` with `error` field set
    - Test that two consecutive `run()` calls do not share state
    - Test JSON output is written to the configured path
    - Test stdout output contains claim, evidence, triples, counts, verdict, and attribution
    - _Requirements: 6.1–6.4, 10.1–10.3, 11.1_

- [x] 11. Implement `LLMBackend` and `BaselinePipeline` (`baseline/llm.py`, `baseline/pipeline.py`)
  - [x] 11.1 Implement `LLMBackend` Protocol, `HuggingFaceLLM.classify()` using `transformers.pipeline("text-generation")` with `max_new_tokens=20`, and `OpenAILLM.classify()` using `openai.ChatCompletion.create` with `temperature=0, max_tokens=20`
  - Implement `MockLLM` with a configurable response string for testing
  - _Requirements: 7.4, 11.4_

  - [x] 11.2 Implement `BaselinePipeline.run()`: retrieve evidence using the shared `Retriever`, build the prompt from design section 4.6, call `llm_backend.classify()`, parse the verdict by searching for `SUPPORTS`/`REFUTES`/`NOT ENOUGH INFO` (case-insensitive), fall back to `NOT ENOUGH INFO` with a warning on unrecognized response or API failure
  - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6_

  - [ ] 11.3 Write unit tests for `BaselinePipeline`
    - Test with `MockLLM` returning each valid verdict label
    - Test that an unrecognized LLM response maps to `NOT ENOUGH INFO` and logs a warning
    - Test that an LLM exception returns `NOT ENOUGH INFO` with `error` field set
    - Test that the prompt contains the claim text and numbered evidence sentences
    - _Requirements: 7.1–7.6, 11.4_

- [x] 12. Implement `Evaluator` (`evaluation/evaluator.py`)
  - [x] 12.1 Implement `Evaluator.evaluate()` computing overall accuracy and per-class F1 for both pipelines using `sklearn.metrics.accuracy_score` and `sklearn.metrics.f1_score(average=None, labels=VERDICT_LABELS)`
  - Map any verdict not in `{"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}` to a sentinel value, treat as incorrect, and log a warning with the claim and invalid verdict
  - Return an `EvaluationReport` including `total_claims`, `skipped_claims`, `label_distribution`, and per-pipeline accuracy + F1 dicts
  - Implement `Evaluator.save_report()` writing the formatted comparison table to the configured file path and printing to stdout
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 12.2 Write unit tests for `Evaluator`
    - Test accuracy and F1 computation against a known ground-truth/prediction fixture
    - Test that an invalid verdict is treated as incorrect and a warning is logged
    - Test `save_report` writes the file and prints to stdout
    - Test `label_distribution` counts match the ground-truth labels
    - Test `skipped_claims` is reported correctly when pipeline results contain errors
    - _Requirements: 8.1–8.5, 11.1_

- [x] 13. Implement CLI entry point (`main.py`)
  - Implement argument parsing with `argparse` supporting `--pipeline {krr,baseline,both}`, `--env-file`, and `--output-json`
  - Load `Config.from_env()`, initialize `DataLoader`, `Retriever` (build or load index), `TripleExtractor`, `ReasoningModule`, and `LLMBackend` based on config
  - Run the selected pipeline(s) over all loaded records, collect results, run `Evaluator`, and write the report
  - _Requirements: 6.1, 7.1, 8.1, 9.1, 10.1–10.3_

- [x] 14. Checkpoint — full pipeline wired
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Write `README.md` and finalize `requirements.txt`
  - Document setup instructions: creating a virtual environment, installing dependencies (`pip install -r requirements.txt`), downloading the spaCy model (`python -m spacy download en_core_web_sm`), and configuring `.env`
  - Document the exact commands to run the KRR pipeline, the baseline pipeline, and the evaluator
  - List all required `.env` keys with descriptions and example values
  - Finalize `requirements.txt` with all transitive dependencies pinned
  - _Requirements: 10.4, 10.5_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests (2.2, 7.2) use `hypothesis` and validate universal correctness properties; task 6.2 validates the round-trip property via fixed-sentence unit tests (spaCy mocked)
- All core modules (DataLoader, TripleExtractor, KnowledgeGraph, ReasoningModule) must be testable without network access or external API calls
- The `MockLLM` backend enables offline testing of `BaselinePipeline`
- Each `KRRPipeline.run()` call creates a fresh `KnowledgeGraph` to prevent cross-claim state leakage
