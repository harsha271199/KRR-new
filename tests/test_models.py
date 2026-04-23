"""Tests for data models (models.py)."""

import pytest
from models import (
    Triple,
    FeverRecord,
    KnowledgeRecord,
    ReasoningResult,
    KRRResult,
    BaselineResult,
    EvaluationReport,
)


# ---------------------------------------------------------------------------
# Triple — normalization
# ---------------------------------------------------------------------------

class TestTripleNormalization:
    def test_fields_lowercased(self):
        t = Triple(subject="Albert Einstein", relation="Born In", object="ULM")
        assert t.subject == "albert einstein"
        assert t.relation == "born in"
        assert t.object == "ulm"

    def test_fields_stripped(self):
        t = Triple(subject="  paris  ", relation=" is in ", object=" france ")
        assert t.subject == "paris"
        assert t.relation == "is in"
        assert t.object == "france"

    def test_already_normalized_unchanged(self):
        t = Triple(subject="paris", relation="is in", object="france")
        assert t.subject == "paris"
        assert t.relation == "is in"
        assert t.object == "france"


# ---------------------------------------------------------------------------
# Triple — canonical string round-trip
# ---------------------------------------------------------------------------

class TestTripleCanonicalString:
    def test_to_canonical_string(self):
        t = Triple(subject="albert einstein", relation="born in", object="ulm")
        assert t.to_canonical_string() == "albert einstein|born in|ulm"

    def test_round_trip(self):
        t = Triple(subject="albert einstein", relation="born in", object="ulm")
        assert Triple.from_canonical_string(t.to_canonical_string()) == t

    def test_pipe_in_object_preserved(self):
        # Object contains a pipe — split("|", 2) must preserve it
        t = Triple(subject="a", relation="b", object="c|d|e")
        s = t.to_canonical_string()
        assert s == "a|b|c|d|e"
        t2 = Triple.from_canonical_string(s)
        assert t2.object == "c|d|e"
        assert t2 == t

    def test_from_canonical_string_raises_on_missing_pipes(self):
        with pytest.raises(ValueError):
            Triple.from_canonical_string("no-pipes-here")

    def test_from_canonical_string_raises_on_one_pipe(self):
        with pytest.raises(ValueError):
            Triple.from_canonical_string("only|one")

    def test_normalization_applied_on_parse(self):
        # from_canonical_string feeds into __init__, so normalization runs
        t = Triple.from_canonical_string("  A  |  B  |  C  ")
        assert t.subject == "a"
        assert t.relation == "b"
        assert t.object == "c"


# ---------------------------------------------------------------------------
# Triple — dict round-trip
# ---------------------------------------------------------------------------

class TestTripleDictRoundTrip:
    def test_to_dict_keys(self):
        t = Triple(subject="paris", relation="is in", object="france")
        d = t.to_dict()
        assert set(d.keys()) == {"subject", "relation", "object"}

    def test_to_dict_values(self):
        t = Triple(subject="paris", relation="is in", object="france")
        d = t.to_dict()
        assert d == {"subject": "paris", "relation": "is in", "object": "france"}

    def test_from_dict_round_trip(self):
        t = Triple(subject="paris", relation="is in", object="france")
        assert Triple.from_dict(t.to_dict()) == t


# ---------------------------------------------------------------------------
# Triple — equality and hashability (frozen dataclass)
# ---------------------------------------------------------------------------

class TestTripleEqualityAndHash:
    def test_equal_triples(self):
        t1 = Triple(subject="paris", relation="is in", object="france")
        t2 = Triple(subject="paris", relation="is in", object="france")
        assert t1 == t2

    def test_unequal_triples(self):
        t1 = Triple(subject="paris", relation="is in", object="france")
        t2 = Triple(subject="london", relation="is in", object="england")
        assert t1 != t2

    def test_hashable_and_usable_in_set(self):
        t1 = Triple(subject="paris", relation="is in", object="france")
        t2 = Triple(subject="paris", relation="is in", object="france")
        t3 = Triple(subject="london", relation="is in", object="england")
        s = {t1, t2, t3}
        assert len(s) == 2

    def test_usable_as_dict_key(self):
        t = Triple(subject="paris", relation="is in", object="france")
        d = {t: "value"}
        assert d[t] == "value"


# ---------------------------------------------------------------------------
# Other dataclasses — basic construction
# ---------------------------------------------------------------------------

class TestOtherDataclasses:
    def test_fever_record(self):
        r = FeverRecord(
            claim="Paris is in France.",
            evidence=["Paris is the capital of France."],
            label="SUPPORTS",
            record_index=0,
        )
        assert r.claim == "Paris is in France."
        assert r.label == "SUPPORTS"
        assert r.record_index == 0

    def test_knowledge_record(self):
        t = Triple(subject="paris", relation="is in", object="france")
        kr = KnowledgeRecord(triple=t, namespace="evidence", source_sentence="Paris is in France.")
        assert kr.namespace == "evidence"
        assert kr.triple == t

    def test_reasoning_result(self):
        t = Triple(subject="paris", relation="is in", object="france")
        rr = ReasoningResult(
            support_count=1,
            refute_count=0,
            irrelevant_count=0,
            verdict="SUPPORTS",
            attribution=[t],
        )
        assert rr.verdict == "SUPPORTS"
        assert len(rr.attribution) == 1

    def test_krr_result_no_error(self):
        result = KRRResult(
            claim="Paris is in France.",
            retrieved_evidence=[],
            claim_triples=[],
            evidence_triples=[],
            support_count=0,
            refute_count=0,
            irrelevant_count=0,
            verdict="NOT ENOUGH INFO",
            attribution=[],
            error=None,
        )
        assert result.error is None

    def test_baseline_result(self):
        result = BaselineResult(
            claim="Paris is in France.",
            retrieved_evidence=[],
            prompt="...",
            verdict="SUPPORTS",
            raw_llm_response="SUPPORTS",
            error=None,
        )
        assert result.verdict == "SUPPORTS"

    def test_evaluation_report(self):
        report = EvaluationReport(
            total_claims=100,
            skipped_claims=2,
            label_distribution={"SUPPORTS": 40, "REFUTES": 30, "NOT ENOUGH INFO": 30},
            krr_accuracy=0.75,
            krr_f1={"SUPPORTS": 0.8, "REFUTES": 0.7, "NOT ENOUGH INFO": 0.75},
            baseline_accuracy=0.70,
            baseline_f1={"SUPPORTS": 0.75, "REFUTES": 0.65, "NOT ENOUGH INFO": 0.70},
        )
        assert report.total_claims == 100
        assert report.krr_accuracy == 0.75
