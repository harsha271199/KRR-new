# Requirements Document

## Introduction

The Attribution-Aware Knowledge Representation and Reasoning (KRR) Fact Verification System takes a natural language claim, retrieves relevant evidence from the FEVER dataset, converts that evidence into structured knowledge triples, and applies an explicit reasoning module to classify the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO — with full attribution to the evidence triples that drove the decision.

The system is built alongside a baseline RAG pipeline (retrieve → LLM classify) to enable direct comparison of transparency, accuracy, and interpretability between implicit and explicit reasoning approaches.

---

## Glossary

- **Claim**: A natural language statement submitted to the system for verification.
- **Evidence**: One or more sentences retrieved from the FEVER Wikipedia corpus that are relevant to a given Claim.
- **Triple**: A structured knowledge unit of the form `(subject, relation, object)` extracted from a natural language sentence.
- **Knowledge Graph (KG)**: An in-memory collection of Triples representing the structured content of Claims and Evidence.
- **Reasoning Module**: The component that compares Claim Triples against Evidence Triples and computes support/conflict features.
- **Verdict**: The final classification label assigned to a Claim — one of `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.
- **Attribution**: The set of Evidence Triples cited as justification for a Verdict.
- **Baseline RAG Pipeline**: A comparison pipeline that retrieves Evidence and passes it directly to an LLM for implicit classification.
- **KRR Pipeline**: The proposed pipeline that extracts Triples, applies explicit reasoning, and produces a Verdict with Attribution.
- **FEVER Dataset**: The Fact Extraction and VERification dataset containing Claims, Evidence sentences, and ground-truth labels.
- **Triple Extractor**: The NLP component (spaCy or Stanford OpenIE) responsible for converting sentences into Triples.
- **Retriever**: The component that uses TF-IDF or BM25 to retrieve relevant Evidence sentences for a given Claim.
- **Evaluator**: The component that computes accuracy and per-class F1 scores for both pipelines.
- **Config**: A configuration file or environment variable store that supplies dataset paths, model settings, and API keys.

---

## Requirements

### Requirement 1: FEVER Dataset Loading and Preprocessing

**User Story:** As a researcher, I want the system to load and preprocess the FEVER dataset, so that Claims, Evidence sentences, and ground-truth labels are available to both pipelines.

#### Acceptance Criteria

1. THE Data_Loader SHALL load Claims, Evidence sentences, and ground-truth labels from the FEVER dataset files whose paths are supplied via Config.
2. WHEN a FEVER dataset file is missing or unreadable, THE Data_Loader SHALL raise a descriptive error identifying the missing file path.
3. WHEN a FEVER record is malformed or missing required fields, THE Data_Loader SHALL skip that record and log a warning identifying the record index.
4. THE Data_Loader SHALL expose the loaded dataset as a collection of records, each containing a Claim string, a list of Evidence strings, and a ground-truth label string.
5. THE Data_Loader SHALL support loading a configurable subset of records (e.g., first N records) to enable fast development and testing runs.

---

### Requirement 2: Evidence Retrieval

**User Story:** As a researcher, I want the system to retrieve the most relevant Evidence sentences for a given Claim, so that downstream components have focused, pertinent context.

#### Acceptance Criteria

1. THE Retriever SHALL index the FEVER Evidence corpus using TF-IDF or BM25 at startup.
2. WHEN a Claim is submitted, THE Retriever SHALL return the top-K most relevant Evidence sentences, where K is a configurable parameter supplied via Config.
3. WHEN a Claim has no retrievable Evidence (empty result set), THE Retriever SHALL return an empty list and log a warning identifying the Claim.
4. THE Retriever SHALL complete retrieval for a single Claim within 2 seconds on a standard development machine.
5. THE Retriever SHALL be independently initializable from a pre-built index file when one is provided via Config, to avoid re-indexing on every run.

---

### Requirement 3: Triple Extraction

**User Story:** As a researcher, I want both Claims and Evidence sentences converted into structured Triples, so that explicit triple-to-triple comparison is possible.

#### Acceptance Criteria

1. THE Triple_Extractor SHALL accept a natural language sentence and return zero or more Triples, each in the form `(subject, relation, object)`.
2. THE Triple_Extractor SHALL extract Triples from both Claim sentences and Evidence sentences using the same extraction logic.
3. WHEN a sentence yields no extractable Triples, THE Triple_Extractor SHALL return an empty list for that sentence and log a debug message identifying the sentence.
4. THE Triple_Extractor SHALL normalize subject, relation, and object strings to lowercase and strip leading/trailing whitespace before storing them.
5. THE Triple_Extractor SHALL use spaCy or Stanford OpenIE as the underlying NLP engine, configurable via Config.
6. FOR ALL valid input sentences, parsing a sentence into Triples and formatting those Triples back into a canonical string representation, then re-parsing that string, SHALL produce a Triple set equivalent to the original (round-trip property).

---

### Requirement 4: Structured Knowledge Representation

**User Story:** As a researcher, I want extracted Triples stored in a structured Knowledge Graph, so that Claim Triples and Evidence Triples can be queried and compared efficiently.

#### Acceptance Criteria

1. THE Knowledge_Graph SHALL store Triples as structured records with distinct subject, relation, and object fields.
2. THE Knowledge_Graph SHALL support separate namespaces for Claim Triples and Evidence Triples within the same graph instance.
3. WHEN Triples are added to the Knowledge_Graph, THE Knowledge_Graph SHALL deduplicate exact-match Triples within the same namespace.
4. THE Knowledge_Graph SHALL support retrieval of all Triples by namespace, by subject, or by relation.
5. THE Knowledge_Graph SHALL be serializable to and deserializable from a JSON representation, preserving all Triple fields and namespace assignments.
6. FOR ALL valid Knowledge_Graph instances, serializing to JSON and then deserializing SHALL produce a Knowledge_Graph equivalent to the original (round-trip property).

---

### Requirement 5: Reasoning Module

**User Story:** As a researcher, I want the system to explicitly compare Claim Triples against Evidence Triples and compute reasoning features, so that the Verdict is explainable and traceable.

#### Acceptance Criteria

1. THE Reasoning_Module SHALL compute a `support_count` equal to the number of Evidence Triples where subject, relation, and object all match a Claim Triple.
2. THE Reasoning_Module SHALL compute a `refute_count` equal to the number of Evidence Triples where subject and relation match a Claim Triple but the object differs.
3. THE Reasoning_Module SHALL compute an `irrelevant_count` equal to the number of Evidence Triples with no meaningful overlap with any Claim Triple.
4. WHEN `support_count` exceeds `refute_count` and `support_count` is greater than zero, THE Reasoning_Module SHALL assign the Verdict `SUPPORTS`.
5. WHEN `refute_count` exceeds `support_count` and `refute_count` is greater than zero, THE Reasoning_Module SHALL assign the Verdict `REFUTES`.
6. WHEN neither `support_count` nor `refute_count` is greater than zero, THE Reasoning_Module SHALL assign the Verdict `NOT ENOUGH INFO`.
7. THE Reasoning_Module SHALL produce an Attribution list containing the specific Evidence Triples that contributed to the assigned Verdict.
8. WHEN the Claim Triple set is empty, THE Reasoning_Module SHALL assign the Verdict `NOT ENOUGH INFO` and return an empty Attribution list.
9. WHEN the Evidence Triple set is empty, THE Reasoning_Module SHALL assign the Verdict `NOT ENOUGH INFO` and return an empty Attribution list.

---

### Requirement 6: KRR Pipeline (End-to-End)

**User Story:** As a researcher, I want to run the full KRR pipeline on a Claim and receive a structured result, so that I can inspect every step of the verification process.

#### Acceptance Criteria

1. THE KRR_Pipeline SHALL accept a Claim string as input and return a structured result containing: retrieved Evidence sentences, Claim Triples, Evidence Triples, reasoning feature counts (`support_count`, `refute_count`, `irrelevant_count`), the Verdict, and the Attribution list.
2. WHEN the KRR_Pipeline processes a Claim, THE KRR_Pipeline SHALL invoke the Retriever, Triple_Extractor, Knowledge_Graph, and Reasoning_Module in that order.
3. WHEN any pipeline stage raises an unhandled exception, THE KRR_Pipeline SHALL catch the exception, log an error with the Claim and stage name, and return a result with Verdict `NOT ENOUGH INFO` and an empty Attribution list.
4. THE KRR_Pipeline SHALL process each Claim independently, with no shared mutable state between Claim executions.

---

### Requirement 7: Baseline RAG Pipeline

**User Story:** As a researcher, I want a baseline RAG pipeline that classifies Claims using an LLM without explicit reasoning, so that I can compare it against the KRR Pipeline.

#### Acceptance Criteria

1. THE Baseline_Pipeline SHALL accept a Claim string as input and return a structured result containing: retrieved Evidence sentences, the LLM prompt sent, and the predicted Verdict.
2. WHEN the Baseline_Pipeline processes a Claim, THE Baseline_Pipeline SHALL retrieve Evidence using the same Retriever used by the KRR_Pipeline.
3. THE Baseline_Pipeline SHALL construct a prompt containing the Claim and retrieved Evidence sentences, and submit it to the configured LLM to obtain a Verdict classification.
4. THE Baseline_Pipeline SHALL support both a HuggingFace-hosted open model and the OpenAI API as LLM backends, selectable via Config.
5. WHEN the LLM returns a response that does not contain a recognizable Verdict label, THE Baseline_Pipeline SHALL assign the Verdict `NOT ENOUGH INFO` and log a warning.
6. WHEN the LLM API call fails or times out, THE Baseline_Pipeline SHALL log an error identifying the Claim and return a result with Verdict `NOT ENOUGH INFO`.

---

### Requirement 8: Evaluation and Comparison

**User Story:** As a researcher, I want the system to evaluate both pipelines on the FEVER dataset and produce a comparison report, so that I can quantify the accuracy and interpretability gains of the KRR approach.

#### Acceptance Criteria

1. THE Evaluator SHALL compute overall accuracy and per-class F1 score for each of the three Verdict labels (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) for both the KRR_Pipeline and the Baseline_Pipeline.
2. THE Evaluator SHALL produce a comparison table showing accuracy and per-class F1 for both pipelines side by side.
3. THE Evaluator SHALL save the comparison report to a file path supplied via Config, in addition to printing it to standard output.
4. WHEN a pipeline produces a Verdict not in the set {`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`}, THE Evaluator SHALL treat that prediction as incorrect and log a warning identifying the Claim and the invalid Verdict.
5. THE Evaluator SHALL report the total number of Claims evaluated, the number skipped due to errors, and the per-label distribution of ground-truth labels.

---

### Requirement 9: Configuration Management

**User Story:** As a developer, I want all dataset paths, model settings, and API keys managed through a Config file or environment variables, so that the system runs without hardcoded values and is portable across environments.

#### Acceptance Criteria

1. THE Config SHALL supply the FEVER dataset file path, the Evidence corpus path, the retrieval top-K value, the NLP engine selection, the LLM backend selection, the LLM API key (if applicable), the evaluation output file path, and the maximum number of records to load.
2. WHEN a required Config value is missing, THE Config SHALL raise a descriptive error identifying the missing key before any pipeline component is initialized.
3. THE Config SHALL support loading values from a `.env` file and from OS environment variables, with environment variables taking precedence over `.env` file values.
4. THE Config SHALL never log or print API key values.

---

### Requirement 10: Output and Reporting

**User Story:** As a researcher, I want per-Claim output showing every step of the KRR Pipeline, so that I can audit the reasoning and attribution for any individual Claim.

#### Acceptance Criteria

1. THE KRR_Pipeline SHALL produce per-Claim output containing: the original Claim text, the list of retrieved Evidence sentences, the list of Claim Triples, the list of Evidence Triples, the `support_count`, `refute_count`, and `irrelevant_count` values, the Verdict, and the Attribution list with the specific Evidence Triples cited.
2. THE KRR_Pipeline SHALL support writing per-Claim output to a structured JSON file at a path supplied via Config.
3. THE KRR_Pipeline SHALL support writing per-Claim output to standard output in a human-readable format.
4. THE README SHALL document setup instructions, all required dependencies, and the exact commands to run both the KRR_Pipeline and the Baseline_Pipeline.
5. THE Project SHALL include a `requirements.txt` file listing all Python dependencies with pinned version numbers.

---

### Requirement 11: Modularity and Testability

**User Story:** As a developer, I want each module independently testable with clear interfaces, so that I can verify correctness in isolation and maintain the codebase over time.

#### Acceptance Criteria

1. THE Data_Loader, Retriever, Triple_Extractor, Knowledge_Graph, Reasoning_Module, Baseline_Pipeline, KRR_Pipeline, and Evaluator SHALL each expose a documented public interface with typed function signatures.
2. THE Data_Loader, Triple_Extractor, Knowledge_Graph, and Reasoning_Module SHALL be testable without network access or external API calls.
3. THE Retriever SHALL be testable using an in-memory corpus without requiring the full FEVER dataset.
4. THE Baseline_Pipeline SHALL be testable using a mock LLM backend that returns configurable responses without making real API calls.
5. WHEN a module receives an input of an unexpected type, THE module SHALL raise a `TypeError` with a message identifying the parameter name and expected type.
