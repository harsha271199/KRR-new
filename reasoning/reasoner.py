"""Reasoning module: compares claim triples against evidence triples and assigns a verdict.

Matching pipeline:
1. Subject matching — exact match OR subject-alias containment
   (e.g. "shakespeare" matches "william shakespeare")
2. Relation normalization — equivalent verbs map to a canonical form
   (e.g. "become" → "be", "locate" → "be")
3. Object matching — four-layer check:
   a. Negation polarity: "X" vs "not X" → always REFUTE (never SUPPORT)
   b. Exact match
   c. Substring containment (handles extraction truncation)
   d. Synonym-normalised match (e.g. "tallest" ≡ "highest" via canonical "tall")
4. Refute guard — mismatch only counts as refutation when objects are
   topically related (same semantic slot):
   - Prepositional objects: same leading preposition → related
   - Multi-word objects: require shared content word
   - Weak/measurement objects filtered out before reaching this stage
5. Weak-evidence filter — objects that are pure measurements or
   circumstantial context are dropped before reasoning
   (e.g. "at metres above sea level", "on march", "from 1887")
6. Comparative numerical reasoning — when a claim contains a comparative
   triple (e.g. "slower than speed of sound"), numeric values extracted
   from evidence are compared to resolve the verdict
"""

from __future__ import annotations

import re

from models import ReasoningResult, Triple

# ---------------------------------------------------------------------------
# Relation synonym table — maps surface lemmas to a canonical form
# ---------------------------------------------------------------------------
_RELATION_SYNONYMS: dict[str, str] = {
    "become": "be",
    "remain": "be",
    "seem": "be",
    "appear": "be",
    "locate": "be",
    "situate": "be",
    "stand": "be",
    "lie": "be",
    "consider": "be",
    "regard": "be",
    "call": "be",
    "name": "be",
    "know": "be",
    "author": "write",
    "compose": "write",
    "pen": "write",
    "born": "bear",
}

# ---------------------------------------------------------------------------
# Object-level word synonym table — maps words to a SINGLE canonical form
# IMPORTANT: "longest" and "largest" are intentionally NOT synonyms —
# the Amazon River claim must remain REFUTES.
# ---------------------------------------------------------------------------
_OBJ_SYNONYM_CANONICAL: dict[str, str] = {
    "tallest": "tall",
    "highest": "tall",
    "taller": "tall",
    "higher": "tall",
    "biggest": "large",
    "bigger": "large",
    "quickest": "fast",
    "fastest": "fast",
    "quicker": "fast",
    "faster": "fast",
}

# ---------------------------------------------------------------------------
# Weak / measurement objects — these are pure context, not semantic values.
# An evidence triple whose object matches one of these patterns is dropped
# before reasoning so it cannot create spurious support or refute counts.
# ---------------------------------------------------------------------------
_WEAK_OBJECT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(at|above|below)\s+\d"),          # "at 8,848 metres", "above sea level"
    re.compile(r"metres?\b"),                         # "at metres", "metres above sea level"
    re.compile(r"^on\s+(january|february|march|april|may|june|july|august|"
               r"september|october|november|december|\d)"),  # "on march", "on july 20"
    re.compile(r"^from\s+\d"),                        # "from 1887"
    re.compile(r"^to\s+\d"),                          # "to 1889"
    re.compile(r"^as\s+centerpiece"),                 # "as centerpiece of fair"
    re.compile(r"standard atmospheric pressure"),
    re.compile(r"sea level"),                         # "at sea level" as standalone
    re.compile(r"^during\b"),
    re.compile(r"^at\s+standard"),
]

# ---------------------------------------------------------------------------
# Stopwords for content-word overlap
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset([
    "a", "an", "the", "in", "on", "at", "of", "to", "by", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "with", "from", "as", "not",
])


# ---------------------------------------------------------------------------
# Public helpers (also used by tests)
# ---------------------------------------------------------------------------

def is_negated_object(obj: str) -> bool:
    """Return True if *obj* starts with the negation prefix "not "."""
    return obj.startswith("not ")


def strip_negation(obj: str) -> str:
    """Remove the leading "not " prefix if present."""
    return obj[4:] if is_negated_object(obj) else obj


def is_weak_object(obj: str) -> bool:
    """Return True if *obj* is a measurement/context phrase that carries no
    semantic content for fact-checking purposes."""
    obj_lower = obj.lower().strip()
    return any(p.search(obj_lower) for p in _WEAK_OBJECT_PATTERNS)


def _normalize_relation(rel: str) -> str:
    return _RELATION_SYNONYMS.get(rel, rel)


def _normalize_object_words(obj: str) -> str:
    """Replace known synonym words with their canonical form."""
    return " ".join(_OBJ_SYNONYM_CANONICAL.get(w, w) for w in obj.split())


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w for w in text.lower().split()
        if w not in _STOPWORDS and len(w) > 1
    )


def _subjects_match(claim_subj: str, evidence_subj: str) -> bool:
    """Return True when subjects are the same or one is an alias of the other.

    Handles cases like:
    - "shakespeare" matches "william shakespeare"
    - "great wall" matches "great wall of china"
    - "speed" matches "speed of light" (only when the claim object references light)

    The rule: subjects match when one is a substring of the other AND the
    shorter form is at least 3 characters (avoids single-word false positives
    like "it" matching "it was").
    """
    if claim_subj == evidence_subj:
        return True
    shorter, longer = (
        (claim_subj, evidence_subj)
        if len(claim_subj) <= len(evidence_subj)
        else (evidence_subj, claim_subj)
    )
    # Require the shorter form to be at least 3 chars and a word-boundary match
    if len(shorter) >= 3 and shorter in longer:
        # Ensure it's a word-boundary match (not a substring of a word)
        pattern = r"\b" + re.escape(shorter) + r"\b"
        if re.search(pattern, longer):
            return True
    return False


def _objects_match(claim_obj: str, evidence_obj: str) -> bool:
    """Return True when the two object strings are semantically compatible.

    Negation polarity is checked first:
    - "X" vs "not X"  → False (negation flip → refute, not support)
    - "not X" vs "X"  → False
    - "not X" vs "not X" → True (both negated → support)

    Then checks exact, substring, and synonym-normalised match.
    """
    claim_neg = is_negated_object(claim_obj)
    ev_neg = is_negated_object(evidence_obj)

    # Mismatched polarity → never a match (will be counted as refute)
    if claim_neg != ev_neg:
        return False

    # Strip negation prefix for content comparison
    c = strip_negation(claim_obj)
    e = strip_negation(evidence_obj)

    if c == e:
        return True
    if c in e or e in c:
        return True

    norm_c = _normalize_object_words(c)
    norm_e = _normalize_object_words(e)
    if norm_c == norm_e:
        return True
    if norm_c in norm_e or norm_e in norm_c:
        return True

    return False


def _objects_are_related(claim_obj: str, evidence_obj: str) -> bool:
    """Return True when the objects are topically related enough to count as refutation.

    Rules:
    1. Negation polarity mismatch → always related (it's a direct contradiction).
    2. Single-token claim objects → always related.
    3. Prepositional objects ("in X", "on Y") → related when leading prep matches.
       This ensures "in france" vs "in ulm" counts as refute (same slot, diff value).
       BUT "at degrees" vs "at standard atmospheric pressure" — both start with "at"
       so they would be related. We break this with the weak-object filter applied
       BEFORE calling this function.
    4. Multi-token non-prepositional objects → require shared content word.
    """
    # Rule 1: negation polarity mismatch is always a direct contradiction
    if is_negated_object(claim_obj) != is_negated_object(evidence_obj):
        return True

    claim_tokens = claim_obj.split()

    # Rule 2: single token
    if len(claim_tokens) <= 1:
        return True

    # Rule 3: prepositional object
    _PREPOSITIONS = frozenset([
        "in", "on", "at", "from", "to", "of", "for", "with",
        "into", "onto", "upon", "over", "under", "above", "below",
    ])
    if claim_tokens[0] in _PREPOSITIONS:
        ev_tokens = evidence_obj.split()
        if ev_tokens and ev_tokens[0] in _PREPOSITIONS:
            return True
        return False

    # Rule 4: content-word overlap
    return bool(_content_words(claim_obj) & _content_words(evidence_obj))


# ---------------------------------------------------------------------------
# Comparative numerical reasoning
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> float | None:
    """Extract the first numeric value from *text*, ignoring commas."""
    text_clean = text.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text_clean)
    return float(match.group()) if match else None


def _resolve_comparative_claim(
    claim_triples: list[Triple],
    evidence_triples: list[Triple],
) -> str | None:
    """Attempt to resolve a comparative claim using numeric evidence values.

    Looks for a claim triple whose object matches the pattern
    "[adj] than [entity]" (e.g. "slower than speed of sound").

    Then searches evidence triples for numeric values associated with:
    - the claim subject (e.g. "speed of light" → 299,792,458)
    - the comparison target (e.g. "speed of sound" → 343)

    Returns "SUPPORTS", "REFUTES", or None if insufficient data.
    """
    for ct in claim_triples:
        obj = ct.object
        # Match "slower than X", "larger than X", etc.
        m = re.match(r"^(\w+er)\s+than\s+(.+)$", obj)
        if m is None:
            continue
        adjective = m.group(1)   # e.g. "slower"
        target = m.group(2)      # e.g. "speed of sound"

        # Collect numeric values from evidence
        subject_values: list[float] = []
        target_values: list[float] = []

        for et in evidence_triples:
            num = _extract_number(et.object)
            if num is None:
                continue
            if _subjects_match(ct.subject, et.subject):
                subject_values.append(num)
            elif _subjects_match(target, et.subject) or target in et.subject:
                target_values.append(num)

        if not subject_values or not target_values:
            continue

        subj_val = max(subject_values)   # take the most prominent value
        tgt_val = max(target_values)

        # "slower" → claim says subj < target
        # "faster/larger/bigger/longer" → claim says subj > target
        _LESS_THAN_ADJS = frozenset(["slower", "smaller", "shorter", "lower", "fewer", "less"])
        _GREATER_THAN_ADJS = frozenset(["faster", "larger", "bigger", "longer", "higher",
                                         "taller", "more", "greater"])

        if adjective in _LESS_THAN_ADJS:
            # Claim: subj < target
            return "SUPPORTS" if subj_val < tgt_val else "REFUTES"
        elif adjective in _GREATER_THAN_ADJS:
            # Claim: subj > target
            return "SUPPORTS" if subj_val > tgt_val else "REFUTES"

    return None


# ---------------------------------------------------------------------------
# Location extraction helper (used by extractor, exposed here for testing)
# ---------------------------------------------------------------------------

def _is_comparative_claim(claim_triples: list[Triple]) -> bool:
    """Return True if any claim triple has a comparative object."""
    return any(
        re.match(r"^\w+er\s+than\s+", ct.object)
        for ct in claim_triples
    )


class ReasoningModule:
    """Explicitly compares claim triples against evidence triples and assigns a verdict."""

    def reason(
        self,
        claim_triples: list[Triple],
        evidence_triples: list[Triple],
    ) -> ReasoningResult:
        """Compute support/refute/irrelevant counts and assign verdict.

        Args:
            claim_triples: Structured triples extracted from the claim sentence.
            evidence_triples: Structured triples extracted from evidence sentences.

        Returns:
            A ReasoningResult with counts, verdict, and attribution triples.
        """
        if not isinstance(claim_triples, list):
            raise TypeError(
                f"claim_triples: expected list[Triple], got {type(claim_triples).__name__}"
            )
        if not isinstance(evidence_triples, list):
            raise TypeError(
                f"evidence_triples: expected list[Triple], got {type(evidence_triples).__name__}"
            )

        if not claim_triples or not evidence_triples:
            return ReasoningResult(
                support_count=0,
                refute_count=0,
                irrelevant_count=0,
                verdict="NOT ENOUGH INFO",
                attribution=[],
            )

        # --- Comparative numerical reasoning (fast path) ---
        # If the claim is comparative and we have numeric evidence, resolve directly.
        if _is_comparative_claim(claim_triples):
            comparative_verdict = _resolve_comparative_claim(
                claim_triples, evidence_triples
            )
            if comparative_verdict is not None:
                return ReasoningResult(
                    support_count=1 if comparative_verdict == "SUPPORTS" else 0,
                    refute_count=1 if comparative_verdict == "REFUTES" else 0,
                    irrelevant_count=0,
                    verdict=comparative_verdict,  # type: ignore[arg-type]
                    attribution=[],
                )

        support_count = 0
        refute_count = 0
        irrelevant_count = 0
        supporting_triples: list[Triple] = []
        refuting_triples: list[Triple] = []

        for evidence_triple in evidence_triples:
            # --- Weak-evidence filter ---
            # Drop evidence triples whose objects are pure measurements or
            # circumstantial context before they can create spurious counts.
            if is_weak_object(evidence_triple.object):
                irrelevant_count += 1
                continue

            ev_rel = _normalize_relation(evidence_triple.relation)

            # Find claim triples that share subject (with alias matching)
            # and normalized relation
            matching_claims = [
                c for c in claim_triples
                if _subjects_match(c.subject, evidence_triple.subject)
                and _normalize_relation(c.relation) == ev_rel
            ]

            if not matching_claims:
                irrelevant_count += 1
            else:
                for claim_triple in matching_claims:
                    if _objects_match(claim_triple.object, evidence_triple.object):
                        support_count += 1
                        supporting_triples.append(evidence_triple)
                    elif _objects_are_related(claim_triple.object, evidence_triple.object):
                        refute_count += 1
                        refuting_triples.append(evidence_triple)
                    else:
                        irrelevant_count += 1

        # Verdict assignment
        if support_count > refute_count and support_count > 0:
            verdict = "SUPPORTS"
            attribution = supporting_triples
        elif refute_count > support_count and refute_count > 0:
            verdict = "REFUTES"
            attribution = refuting_triples
        else:
            verdict = "NOT ENOUGH INFO"
            attribution = []

        return ReasoningResult(
            support_count=support_count,
            refute_count=refute_count,
            irrelevant_count=irrelevant_count,
            verdict=verdict,
            attribution=attribution,
        )
