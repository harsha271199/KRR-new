"""Tests for TripleExtractor (knowledge/extractor.py).

spaCy is mocked so these tests run without downloading a language model.
The mock NLP pipeline returns a predictable dependency parse for a small
set of fixture sentences, letting us verify extraction logic in isolation.
"""

from __future__ import annotations

import os
import types
from unittest.mock import MagicMock, patch

import pytest

from models import Triple


# ---------------------------------------------------------------------------
# Helpers to build a minimal spaCy-like doc/token structure
#
# We use plain Python objects (not MagicMock) so that identity comparisons
# like `token.head == root` work correctly — the extractor uses `is`-style
# equality via Python's default object identity.
# ---------------------------------------------------------------------------

class _FakeToken:
    """Minimal stand-in for a spaCy Token."""

    def __init__(self, text, dep_, i=0, lemma_=None):
        self.text = text
        self.dep_ = dep_
        self.i = i
        self.lemma_ = lemma_ or text.lower()
        self.children: list["_FakeToken"] = []
        self.head: "_FakeToken" = self  # default: self (root)


class _FakeSent:
    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class _FakeDoc:
    def __init__(self, sents):
        self.sents = sents


def _make_nlp(doc):
    """Return a callable mock NLP that always returns *doc*."""
    nlp = MagicMock()
    nlp.return_value = doc
    return nlp


# ---------------------------------------------------------------------------
# Fixture: "Albert Einstein was born in Ulm."
# Dependency structure:
#   Einstein -nsubj-> born (ROOT)
#   born     -ROOT
#   Ulm      -pobj-> in (prep)
#   in       -prep-> born
# ---------------------------------------------------------------------------

def _build_einstein_doc():
    root = _FakeToken("born", "ROOT", i=2, lemma_="bear")
    root.head = root  # root points to itself

    # prep child "in" with pobj child "Ulm"
    ulm = _FakeToken("Ulm", "pobj", i=4)
    prep_in = _FakeToken("in", "prep", i=3)
    prep_in.head = root
    ulm.head = prep_in
    prep_in.children = [ulm]

    # subject — head must be the same object as root
    einstein = _FakeToken("Einstein", "nsubj", i=1)
    einstein.head = root

    root.children = [einstein, prep_in]

    return _FakeDoc([_FakeSent([einstein, root, prep_in, ulm])])


# ---------------------------------------------------------------------------
# Fixture: "The sky is blue."
# Dependency structure:
#   sky  -nsubj-> is (ROOT)
#   blue -attr  -> is
# ---------------------------------------------------------------------------

def _build_sky_doc():
    root = _FakeToken("is", "ROOT", i=1, lemma_="be")
    root.head = root

    sky = _FakeToken("sky", "nsubj", i=0)
    sky.head = root

    blue = _FakeToken("blue", "attr", i=2)
    blue.head = root

    root.children = [sky, blue]
    return _FakeDoc([_FakeSent([sky, root, blue])])


# ---------------------------------------------------------------------------
# Fixture: sentence with no extractable triples (no subject)
# ---------------------------------------------------------------------------

def _build_no_subject_doc():
    root = _FakeToken("raining", "ROOT", i=0, lemma_="rain")
    root.head = root
    root.children = []
    return _FakeDoc([_FakeSent([root])])


# ---------------------------------------------------------------------------
# Patch helper
# ---------------------------------------------------------------------------

def _make_extractor_with_nlp(nlp_mock):
    """Instantiate TripleExtractor with a patched spaCy load."""
    import config as cfg_module
    from config import Config

    with patch.dict(os.environ, {
        "FEVER_DATASET_PATH": "/x",
        "EVIDENCE_CORPUS_PATH": "/x",
        "EVAL_OUTPUT_PATH": "/x",
    }, clear=True):
        cfg = Config.from_env()

    with patch("knowledge.extractor.TripleExtractor.__init__", lambda self, c: None):
        from knowledge.extractor import TripleExtractor
        extractor = TripleExtractor.__new__(TripleExtractor)
        extractor._nlp = nlp_mock
        return extractor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTripleExtractorBasic:
    def test_extract_returns_list(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_einstein_doc()))
        result = extractor.extract("Albert Einstein was born in Ulm.")
        assert isinstance(result, list)

    def test_extract_svo_sentence(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_sky_doc()))
        result = extractor.extract("The sky is blue.")
        assert len(result) >= 1
        triple = result[0]
        assert triple.subject == "sky"
        assert triple.relation == "be"
        assert triple.object == "blue"

    def test_extract_prep_object(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_einstein_doc()))
        result = extractor.extract("Albert Einstein was born in Ulm.")
        assert len(result) >= 1
        triple = result[0]
        assert triple.subject == "einstein"
        assert "ulm" in triple.object

    def test_no_triples_returns_empty_list(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_no_subject_doc()))
        result = extractor.extract("It is raining.")
        assert result == []

    def test_normalization_applied(self):
        """All triple fields must be lowercase and stripped."""
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_sky_doc()))
        result = extractor.extract("THE SKY IS BLUE.")
        for triple in result:
            assert triple.subject == triple.subject.lower()
            assert triple.relation == triple.relation.lower()
            assert triple.object == triple.object.lower()

    def test_type_error_on_non_string(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(MagicMock())
        with pytest.raises(TypeError):
            extractor.extract(123)

    def test_type_error_message_mentions_str(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(MagicMock())
        with pytest.raises(TypeError) as exc_info:
            extractor.extract(None)
        assert "str" in str(exc_info.value).lower()


class TestTripleExtractorBatch:
    def test_extract_batch_returns_one_list_per_sentence(self):
        from knowledge.extractor import TripleExtractor
        nlp = MagicMock(side_effect=[
            _build_sky_doc(),
            _build_einstein_doc(),
        ])
        extractor = _make_extractor_with_nlp(nlp)
        results = extractor.extract_batch(["The sky is blue.", "Einstein was born in Ulm."])
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_extract_batch_empty_input(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(MagicMock())
        results = extractor.extract_batch([])
        assert results == []


class TestTripleRoundTripViaExtractor:
    """Property 1 (extractor path): canonical string round-trip for extracted triples."""

    def test_round_trip_sky_sentence(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_sky_doc()))
        triples = extractor.extract("The sky is blue.")
        for triple in triples:
            restored = Triple.from_canonical_string(triple.to_canonical_string())
            assert restored == triple

    def test_round_trip_einstein_sentence(self):
        from knowledge.extractor import TripleExtractor
        extractor = _make_extractor_with_nlp(_make_nlp(_build_einstein_doc()))
        triples = extractor.extract("Albert Einstein was born in Ulm.")
        for triple in triples:
            restored = Triple.from_canonical_string(triple.to_canonical_string())
            assert restored == triple
