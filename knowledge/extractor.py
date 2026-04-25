"""Triple extraction from natural language sentences using spaCy dependency parsing.

Improvements:
- Passive voice inversion: "Hamlet was written by Shakespeare" →
  (shakespeare, write, hamlet)
- Passive ACL with sentence-subject promotion:
  "Hamlet is a tragedy written by William Shakespeare" →
  (william shakespeare, write, hamlet)   ← sentence subject, not ACL head
- Negation detection: "not visible from space" preserved in object
- Comparative structures: "X is slower than Y" → (x, be, slower than y)
- acomp objects: "X is visible from space" → (x, be, visible from space)
- Location extraction: nested prep chains like "on the Champ de Mars in Paris"
  produce an additional (subject, locate, in paris) triple
- Excluded noisy preps: "during", "while", "after", "before", "since"
- Full prepositional noun phrases: "speed of sound" captured as single object
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import Config
from models import Triple

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

_EXCLUDED_PREPS = frozenset(["by", "during", "while", "after", "before", "since"])


def normalize(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower()


def _get_span_text(token: "spacy.tokens.Token") -> str:
    """Return token text with compound/amod modifiers, in sentence order."""
    tokens = [token]
    for child in token.children:
        if child.dep_ in ("compound", "amod"):
            tokens.append(child)
    tokens.sort(key=lambda t: t.i)
    return " ".join(t.text for t in tokens)


def _get_full_noun_phrase(token: "spacy.tokens.Token") -> str:
    """Return token text with compound/amod modifiers AND one level of prep+pobj."""
    parts = [_get_span_text(token)]
    for child in token.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    parts.append(child.text.lower())
                    parts.append(_get_span_text(grandchild).lower())
    return " ".join(parts)


def _find_agent(root: "spacy.tokens.Token") -> "spacy.tokens.Token | None":
    """Return the by-agent pobj token for a passive construction, or None."""
    for child in root.children:
        if child.dep_ == "agent":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild
        if child.dep_ == "prep" and child.text.lower() == "by":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild
    return None


def _has_negation(root: "spacy.tokens.Token") -> bool:
    """Return True if the root verb has a negation (neg) child."""
    return any(child.dep_ == "neg" for child in root.children)


def _find_comparative_object(root: "spacy.tokens.Token") -> str | None:
    """Detect "X is [adj/adv]-er than Y" and return a descriptive object string."""
    for child in root.children:
        if child.dep_ in ("acomp", "advmod"):
            for grandchild in child.children:
                if grandchild.dep_ == "prep" and grandchild.text.lower() == "than":
                    for ggchild in grandchild.children:
                        if ggchild.dep_ == "pobj":
                            adj_text = normalize(_get_span_text(child))
                            than_obj = normalize(_get_full_noun_phrase(ggchild))
                            return f"{adj_text} than {than_obj}"
    return None


def _collect_objects(root: "spacy.tokens.Token", negated: bool) -> list[str]:
    """Collect dobj/attr/acomp/prep+pobj objects from root."""
    objects: list[str] = []
    for child in root.children:
        if child.dep_ in ("dobj", "attr"):
            obj_text = normalize(_get_span_text(child))
            objects.append(("not " + obj_text) if negated else obj_text)

        elif child.dep_ == "acomp":
            acomp_parts = [normalize(_get_span_text(child))]
            for grandchild in child.children:
                if grandchild.dep_ == "prep":
                    for ggchild in grandchild.children:
                        if ggchild.dep_ == "pobj":
                            acomp_parts.append(normalize(grandchild.text))
                            acomp_parts.append(normalize(_get_full_noun_phrase(ggchild)))
            obj_text = " ".join(acomp_parts)
            objects.append(("not " + obj_text) if negated else obj_text)

        elif child.dep_ == "prep":
            if child.text.lower() in _EXCLUDED_PREPS:
                continue
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    prep_text = normalize(child.text)
                    pobj_text = normalize(_get_full_noun_phrase(grandchild))
                    obj_text = f"{prep_text} {pobj_text}"
                    objects.append(("not " + obj_text) if negated else obj_text)
    return objects


def _extract_nested_location(
    subject_text: str,
    root: "spacy.tokens.Token",
) -> list[Triple]:
    """Extract location triples buried in nested prepositional chains.

    Handles patterns like:
      "The Eiffel Tower is a tower on the Champ de Mars in Paris, France."
    where "in Paris" is a child of "Mars" (pobj of "on"), not a direct child
    of the root verb.

    Strategy: walk all prep→pobj chains reachable from the root's attr/dobj
    children and collect any "in/at/near <place>" phrases at any depth.
    """
    triples: list[Triple] = []
    _LOCATION_PREPS = frozenset(["in", "at", "near"])

    # Non-discriminating location objects — too generic to be useful
    _GENERIC_LOCATIONS = frozenset(["world", "the world", "earth", "universe"])

    def _walk(token: "spacy.tokens.Token", depth: int = 0) -> None:
        if depth > 5:
            return
        for child in token.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        # Emit a location triple only for location prepositions
                        if child.text.lower() in _LOCATION_PREPS:
                            loc_text = normalize(_get_span_text(grandchild))
                            # Skip generic/non-discriminating locations
                            if loc_text not in _GENERIC_LOCATIONS:
                                prep_text = normalize(child.text)
                                obj_text = f"{prep_text} {loc_text}"
                                triples.append(Triple(
                                    subject=subject_text,
                                    relation="locate",
                                    object=obj_text,
                                ))
                        # Always recurse into pobj regardless of prep type
                        _walk(grandchild, depth + 1)
            elif child.dep_ in ("pobj", "attr", "dobj"):
                _walk(child, depth + 1)

    # Walk from attr/dobj children of root
    for child in root.children:
        if child.dep_ in ("attr", "dobj"):
            _walk(child)

    return triples


def _extract_acl_passives(sent: "spacy.tokens.Span") -> list[Triple]:
    """Extract triples from passive ACL verbs.

    Handles two patterns:

    Pattern A — ACL head is the logical object:
      "a tragedy written by Shakespeare"
      → (shakespeare, write, tragedy)

    Pattern B — Sentence subject is the logical object (promoted):
      "Hamlet is a tragedy written by Shakespeare"
      → (shakespeare, write, hamlet)   ← sentence subject promoted

    Pattern B fires when the ACL head is an attr/dobj of the sentence root
    AND the sentence root has an nsubj.  The sentence subject is the true
    logical object (the work), not the intermediate noun (tragedy).
    """
    triples: list[Triple] = []

    # Find sentence root for Pattern B
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    root_nsubj: "spacy.tokens.Token | None" = None
    if root is not None:
        for t in sent:
            if t.dep_ == "nsubj" and t.head == root:
                root_nsubj = t
                break

    for token in sent:
        if token.dep_ != "acl" or token.tag_ not in ("VBN", "VBD"):
            continue
        agent = _find_agent(token)
        if agent is None:
            continue

        logical_subject = normalize(_get_span_text(agent))
        relation = normalize(token.lemma_)
        head_noun = token.head  # e.g. "tragedy"

        # Pattern B: promote sentence subject when ACL head is attr/dobj of root
        if (
            root_nsubj is not None
            and head_noun.head == root
            and head_noun.dep_ in ("attr", "dobj")
        ):
            # Sentence subject (e.g. "Hamlet") is the logical object
            logical_object = normalize(_get_span_text(root_nsubj))
            triples.append(Triple(
                subject=logical_subject,
                relation=relation,
                object=logical_object,
            ))
            # Also emit the ACL-head triple as a fallback
            triples.append(Triple(
                subject=logical_subject,
                relation=relation,
                object=normalize(_get_span_text(head_noun)),
            ))
        else:
            # Pattern A: ACL head is the logical object
            triples.append(Triple(
                subject=logical_subject,
                relation=relation,
                object=normalize(_get_span_text(head_noun)),
            ))

    return triples


class TripleExtractor:
    """Extract (subject, relation, object) triples from natural language sentences."""

    def __init__(self, config: Config) -> None:
        import spacy
        self._nlp = spacy.load(config.spacy_model)

    def extract(self, sentence: str) -> list[Triple]:
        """Extract triples from a single sentence.

        Args:
            sentence: A natural language sentence.

        Returns:
            List of Triple objects; [] if nothing extractable.

        Raises:
            TypeError: If sentence is not a str.
        """
        if not isinstance(sentence, str):
            raise TypeError(f"sentence must be a str, got {type(sentence).__name__}")

        doc = self._nlp(sentence)
        triples: list[Triple] = []

        for sent in doc.sents:
            # --- ACL passive triples ---
            triples.extend(_extract_acl_passives(sent))

            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root is None:
                continue

            relation_text = normalize(root.lemma_)
            negated = _has_negation(root)

            gram_subjects = [
                t for t in sent
                if t.dep_ in ("nsubj", "nsubjpass") and t.head == root
            ]
            if not gram_subjects:
                continue

            is_passive = any(t.dep_ == "nsubjpass" for t in gram_subjects)

            # --- Passive voice (ROOT is passive) ---
            if is_passive:
                agent = _find_agent(root)
                if agent is not None:
                    logical_subject = normalize(_get_span_text(agent))
                    for gram_subj in gram_subjects:
                        logical_object = normalize(_get_span_text(gram_subj))
                        if negated:
                            logical_object = "not " + logical_object
                        triples.append(Triple(
                            subject=logical_subject,
                            relation=relation_text,
                            object=logical_object,
                        ))
                    for gram_subj in gram_subjects:
                        subj_text = normalize(_get_span_text(gram_subj))
                        for obj_text in _collect_objects(root, negated):
                            triples.append(Triple(
                                subject=subj_text,
                                relation=relation_text,
                                object=obj_text,
                            ))
                    continue

            # --- Comparative structure ---
            comparative_obj = _find_comparative_object(root)
            if comparative_obj is not None:
                for subj_token in gram_subjects:
                    subj_text = normalize(_get_span_text(subj_token))
                    triples.append(Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=comparative_obj,
                    ))
                continue

            # --- Standard active extraction ---
            objects = _collect_objects(root, negated)
            if not objects:
                continue

            for subj_token in gram_subjects:
                subj_text = normalize(_get_span_text(subj_token))
                for obj_text in objects:
                    triples.append(Triple(
                        subject=subj_text,
                        relation=relation_text,
                        object=obj_text,
                    ))

                # --- Nested location extraction ---
                # For sentences like "X is a tower on the Champ de Mars in Paris"
                # extract additional (subject, locate, in paris) triples
                location_triples = _extract_nested_location(subj_text, root)
                triples.extend(location_triples)

        if not triples:
            logger.debug("No triples extracted from sentence: %r", sentence)

        return triples

    def extract_batch(self, sentences: list[str]) -> list[list[Triple]]:
        """Extract triples from multiple sentences."""
        return [self.extract(sentence) for sentence in sentences]
