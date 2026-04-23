"""Tests for KnowledgeGraph (knowledge/graph.py)."""

import pytest
from models import Triple
from knowledge.graph import KnowledgeGraph


def t(s, r, o):
    return Triple(subject=s, relation=r, object=o)


PARIS = t("paris", "is in", "france")
LONDON = t("london", "is in", "england")
EINSTEIN = t("albert einstein", "born in", "ulm")


class TestKnowledgeGraphAddAndGet:
    def test_add_and_retrieve_single_triple(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim")
        result = kg.get_triples()
        assert PARIS in result

    def test_deduplication_same_namespace(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim")
        kg.add_triple(PARIS, namespace="claim")
        assert kg.get_triples(namespace="claim").count(PARIS) == 1

    def test_same_triple_different_namespaces_stored_separately(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim")
        kg.add_triple(PARIS, namespace="evidence")
        claim_triples = kg.get_triples(namespace="claim")
        evidence_triples = kg.get_triples(namespace="evidence")
        assert PARIS in claim_triples
        assert PARIS in evidence_triples
        assert len(claim_triples) == 1
        assert len(evidence_triples) == 1

    def test_get_triples_no_filter_returns_all(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim")
        kg.add_triple(LONDON, namespace="evidence")
        all_triples = kg.get_triples()
        assert PARIS in all_triples
        assert LONDON in all_triples
        assert len(all_triples) == 2


class TestKnowledgeGraphFiltering:
    def setup_method(self):
        self.kg = KnowledgeGraph()
        self.kg.add_triple(PARIS, namespace="claim")
        self.kg.add_triple(LONDON, namespace="evidence")
        self.kg.add_triple(EINSTEIN, namespace="evidence")

    def test_filter_by_namespace(self):
        result = self.kg.get_triples(namespace="claim")
        assert result == [PARIS]

    def test_filter_by_namespace_evidence(self):
        result = self.kg.get_triples(namespace="evidence")
        assert LONDON in result
        assert EINSTEIN in result
        assert PARIS not in result

    def test_filter_by_subject(self):
        result = self.kg.get_triples(subject="paris")
        assert result == [PARIS]

    def test_filter_by_relation(self):
        result = self.kg.get_triples(relation="is in")
        assert PARIS in result
        assert LONDON in result
        assert EINSTEIN not in result

    def test_filter_by_namespace_and_relation(self):
        result = self.kg.get_triples(namespace="evidence", relation="is in")
        assert LONDON in result
        assert PARIS not in result
        assert EINSTEIN not in result

    def test_nonexistent_namespace_returns_empty(self):
        result = self.kg.get_triples(namespace="nonexistent")
        assert result == []

    def test_nonexistent_subject_returns_empty(self):
        result = self.kg.get_triples(subject="nobody")
        assert result == []


class TestKnowledgeGraphSerialization:
    def test_to_dict_structure(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim", source_sentence="Paris is in France.")
        d = kg.to_dict()
        assert "namespaces" in d
        assert "claim" in d["namespaces"]
        entry = d["namespaces"]["claim"][0]
        assert entry["subject"] == "paris"
        assert entry["relation"] == "is in"
        assert entry["object"] == "france"
        assert entry["source_sentence"] == "Paris is in France."

    def test_round_trip_preserves_triples(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim", source_sentence="Paris is in France.")
        kg.add_triple(LONDON, namespace="evidence", source_sentence="London is in England.")
        kg2 = KnowledgeGraph.from_dict(kg.to_dict())
        assert PARIS in kg2.get_triples(namespace="claim")
        assert LONDON in kg2.get_triples(namespace="evidence")

    def test_round_trip_preserves_namespace_separation(self):
        kg = KnowledgeGraph()
        kg.add_triple(PARIS, namespace="claim")
        kg.add_triple(PARIS, namespace="evidence")
        kg2 = KnowledgeGraph.from_dict(kg.to_dict())
        assert PARIS in kg2.get_triples(namespace="claim")
        assert PARIS in kg2.get_triples(namespace="evidence")

    def test_round_trip_empty_graph(self):
        kg = KnowledgeGraph()
        kg2 = KnowledgeGraph.from_dict(kg.to_dict())
        assert kg2.get_triples() == []

    def test_from_dict_deduplicates(self):
        # Manually craft a dict with duplicate entries
        data = {
            "namespaces": {
                "claim": [
                    {"subject": "paris", "relation": "is in", "object": "france", "source_sentence": ""},
                    {"subject": "paris", "relation": "is in", "object": "france", "source_sentence": ""},
                ]
            }
        }
        kg = KnowledgeGraph.from_dict(data)
        assert len(kg.get_triples(namespace="claim")) == 1
