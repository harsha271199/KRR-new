"""Tests for ReasoningModule (reasoning/reasoner.py)."""

import pytest
from models import Triple, ReasoningResult
from reasoning.reasoner import ReasoningModule


def t(s, r, o):
    return Triple(subject=s, relation=r, object=o)


# Shared claim triple: (einstein, born in, ulm)
CLAIM_EINSTEIN_ULM = t("einstein", "born in", "ulm")

# Evidence triples
EV_SUPPORTS = t("einstein", "born in", "ulm")       # full match → support
EV_REFUTES  = t("einstein", "born in", "munich")    # subject+relation match, object differs → refute
EV_IRRELEVANT = t("curie", "discovered", "radium")  # no subject+relation overlap → irrelevant


class TestReasoningModuleSupports:
    def test_supports_verdict(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_SUPPORTS])
        assert result.verdict == "SUPPORTS"
        assert result.support_count == 1
        assert result.refute_count == 0

    def test_supports_when_support_exceeds_refute(self):
        rm = ReasoningModule()
        result = rm.reason(
            [CLAIM_EINSTEIN_ULM],
            [EV_SUPPORTS, EV_SUPPORTS, EV_REFUTES],
        )
        assert result.verdict == "SUPPORTS"
        assert result.support_count == 2
        assert result.refute_count == 1

    def test_attribution_contains_supporting_triples(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_SUPPORTS])
        assert EV_SUPPORTS in result.attribution


class TestReasoningModuleRefutes:
    def test_refutes_verdict(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_REFUTES])
        assert result.verdict == "REFUTES"
        assert result.refute_count == 1
        assert result.support_count == 0

    def test_refutes_when_refute_exceeds_support(self):
        rm = ReasoningModule()
        result = rm.reason(
            [CLAIM_EINSTEIN_ULM],
            [EV_REFUTES, EV_REFUTES, EV_SUPPORTS],
        )
        assert result.verdict == "REFUTES"
        assert result.refute_count == 2
        assert result.support_count == 1

    def test_attribution_contains_refuting_triples(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_REFUTES])
        assert EV_REFUTES in result.attribution


class TestReasoningModuleNotEnoughInfo:
    def test_not_enough_info_when_both_zero(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_IRRELEVANT])
        assert result.verdict == "NOT ENOUGH INFO"
        assert result.support_count == 0
        assert result.refute_count == 0

    def test_not_enough_info_on_tie(self):
        rm = ReasoningModule()
        result = rm.reason(
            [CLAIM_EINSTEIN_ULM],
            [EV_SUPPORTS, EV_REFUTES],
        )
        assert result.verdict == "NOT ENOUGH INFO"
        assert result.support_count == 1
        assert result.refute_count == 1

    def test_not_enough_info_empty_claim_triples(self):
        rm = ReasoningModule()
        result = rm.reason([], [EV_SUPPORTS])
        assert result.verdict == "NOT ENOUGH INFO"
        assert result.support_count == 0
        assert result.refute_count == 0
        assert result.attribution == []

    def test_not_enough_info_empty_evidence_triples(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [])
        assert result.verdict == "NOT ENOUGH INFO"
        assert result.support_count == 0
        assert result.refute_count == 0
        assert result.attribution == []

    def test_not_enough_info_both_empty(self):
        rm = ReasoningModule()
        result = rm.reason([], [])
        assert result.verdict == "NOT ENOUGH INFO"

    def test_attribution_empty_on_not_enough_info(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_IRRELEVANT])
        assert result.attribution == []

    def test_attribution_empty_on_tie(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_SUPPORTS, EV_REFUTES])
        assert result.attribution == []


class TestReasoningModuleIrrelevantCount:
    def test_irrelevant_count(self):
        rm = ReasoningModule()
        result = rm.reason([CLAIM_EINSTEIN_ULM], [EV_IRRELEVANT, EV_IRRELEVANT])
        assert result.irrelevant_count == 2

    def test_mixed_counts(self):
        rm = ReasoningModule()
        result = rm.reason(
            [CLAIM_EINSTEIN_ULM],
            [EV_SUPPORTS, EV_REFUTES, EV_IRRELEVANT],
        )
        assert result.support_count == 1
        assert result.refute_count == 1
        assert result.irrelevant_count == 1


class TestReasoningModuleTypeErrors:
    def test_claim_triples_wrong_type_raises(self):
        rm = ReasoningModule()
        with pytest.raises(TypeError):
            rm.reason("not a list", [])

    def test_evidence_triples_wrong_type_raises(self):
        rm = ReasoningModule()
        with pytest.raises(TypeError):
            rm.reason([], "not a list")
