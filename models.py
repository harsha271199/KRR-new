from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Triple:
    """Atomic unit of structured knowledge: (subject, relation, object).

    All fields are normalized to lowercase and stripped of leading/trailing
    whitespace at construction time.
    """

    subject: str
    relation: str
    object: str

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "subject", self.subject.strip().lower())
        object.__setattr__(self, "relation", self.relation.strip().lower())
        object.__setattr__(self, "object", self.object.strip().lower())

    def to_canonical_string(self) -> str:
        """Return 'subject|relation|object' for round-trip parsing."""
        return f"{self.subject}|{self.relation}|{self.object}"

    @classmethod
    def from_canonical_string(cls, s: str) -> Triple:
        """Construct a Triple from a pipe-delimited canonical string.

        Splits on the first two pipes only, so objects containing pipes are
        preserved.

        Raises:
            ValueError: If the string does not contain exactly 3 parts.
        """
        parts = s.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid canonical triple string: {s!r}")
        return cls(subject=parts[0], relation=parts[1], object=parts[2])

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Triple:
        """Construct a Triple from a dict with 'subject', 'relation', 'object' keys."""
        return cls(subject=d["subject"], relation=d["relation"], object=d["object"])


@dataclass
class FeverRecord:
    """One record from the FEVER dataset."""

    claim: str
    evidence: list[str]   # list of evidence sentences
    label: str            # "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
    record_index: int     # original index in dataset file


@dataclass
class KnowledgeRecord:
    """A triple annotated with its source namespace and provenance."""

    triple: Triple
    namespace: str        # "claim" | "evidence"
    source_sentence: str  # the sentence the triple was extracted from


@dataclass
class ReasoningResult:
    """Output of the ReasoningModule for a single claim."""

    support_count: int
    refute_count: int
    irrelevant_count: int
    verdict: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    attribution: list[Triple]  # evidence triples that drove the verdict


@dataclass
class KRRResult:
    """Full structured output of the KRR pipeline for one claim."""

    claim: str
    retrieved_evidence: list[str]
    claim_triples: list[Triple]
    evidence_triples: list[Triple]
    support_count: int
    refute_count: int
    irrelevant_count: int
    verdict: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    attribution: list[Triple]
    error: str | None  # set if a pipeline stage raised an exception


@dataclass
class BaselineResult:
    """Output of the Baseline RAG pipeline for one claim."""

    claim: str
    retrieved_evidence: list[str]
    prompt: str
    verdict: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    raw_llm_response: str
    error: str | None


@dataclass
class EvaluationReport:
    """Aggregated metrics for both pipelines.

    Fields are ``None`` when the corresponding pipeline was not run.
    """

    total_claims: int
    skipped_claims: int
    label_distribution: dict[str, int]

    # KRR metrics
    krr_accuracy: float | None = None
    krr_f1: dict[str, float] | None = None
    krr_precision: dict[str, float] | None = None
    krr_recall: dict[str, float] | None = None
    krr_confusion: list[list[int]] | None = None
    krr_predictions: list[str] | None = None

    # Baseline metrics
    baseline_accuracy: float | None = None
    baseline_f1: dict[str, float] | None = None
    baseline_precision: dict[str, float] | None = None
    baseline_recall: dict[str, float] | None = None
    baseline_confusion: list[list[int]] | None = None
    baseline_predictions: list[str] | None = None

    # Ground truth (for error analysis)
    ground_truth: list[str] | None = None
