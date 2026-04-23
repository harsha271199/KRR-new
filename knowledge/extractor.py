"""Triple extraction from natural language sentences using spaCy dependency parsing.

Implements the TripleExtractor class that converts sentences into normalized
(subject, relation, object) triples.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import Config
from models import Triple

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Normalize text to lowercase and strip leading/trailing whitespace."""
    return text.strip().lower()


def _get_span_text(token: "spacy.tokens.Token") -> str:
    """Build span text for a token, including compound and adjectival modifiers.

    Collects tokens that are compound or amod dependents of the head token,
    then combines with the head token text, preserving original token order.
    """
    # Gather the head token and its compound/amod children
    tokens = [token]
    for child in token.children:
        if child.dep_ in ("compound", "amod"):
            tokens.append(child)

    # Sort by position in the sentence to preserve original order
    tokens.sort(key=lambda t: t.i)
    return " ".join(t.text for t in tokens)


class TripleExtractor:
    """Extract (subject, relation, object) triples from natural language sentences.

    Uses spaCy dependency parsing to identify grammatical relations and build
    normalized triples.

    Args:
        config: Runtime configuration supplying the spaCy model name.
    """

    def __init__(self, config: Config) -> None:
        import spacy

        self._nlp = spacy.load(config.spacy_model)

    def extract(self, sentence: str) -> list[Triple]:
        """Extract triples from a single sentence.

        Uses spaCy dependency parsing to identify:
        - Subject: tokens with nsubj or nsubjpass dependency
        - Relation: the root verb of the sentence
        - Object: tokens with dobj, attr, or prep+pobj dependency

        Compound nouns and adjectival modifiers are included in span text.
        All fields are normalized (lowercase + strip) before constructing Triples.

        Args:
            sentence: A natural language sentence to extract triples from.

        Returns:
            A list of Triple objects. Returns [] if no triples can be extracted.

        Raises:
            TypeError: If sentence is not a str.
        """
        if not isinstance(sentence, str):
            raise TypeError(
                f"sentence must be a str, got {type(sentence).__name__}"
            )

        doc = self._nlp(sentence)
        triples: list[Triple] = []

        for sent in doc.sents:
            # Find the root verb of this sentence
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break

            if root is None:
                continue

            relation_text = normalize(root.lemma_)

            # Find all subjects (nsubj or nsubjpass)
            subjects = [
                token for token in sent
                if token.dep_ in ("nsubj", "nsubjpass") and token.head == root
            ]

            if not subjects:
                continue

            # Find all objects attached to the root verb
            objects: list[str] = []

            for child in root.children:
                if child.dep_ in ("dobj", "attr"):
                    objects.append(normalize(_get_span_text(child)))
                elif child.dep_ == "prep":
                    # Look for pobj children of the prep
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            prep_text = normalize(child.text)
                            pobj_text = normalize(_get_span_text(grandchild))
                            objects.append(f"{prep_text} {pobj_text}")

            if not objects:
                continue

            # Create one triple per (subject, object) combination
            for subject_token in subjects:
                subject_text = normalize(_get_span_text(subject_token))
                for object_text in objects:
                    triples.append(
                        Triple(
                            subject=subject_text,
                            relation=relation_text,
                            object=object_text,
                        )
                    )

        if not triples:
            logger.debug(
                "No triples extracted from sentence: %r", sentence
            )

        return triples

    def extract_batch(self, sentences: list[str]) -> list[list[Triple]]:
        """Extract triples from multiple sentences.

        Args:
            sentences: A list of natural language sentences.

        Returns:
            A list of triple lists, one per input sentence.
        """
        return [self.extract(sentence) for sentence in sentences]
