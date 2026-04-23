"""Data loading and preprocessing for the FEVER dataset.

Reads FEVER JSONL files line-by-line, parses each record, flattens nested
evidence into a list of strings, normalizes labels to uppercase, and exposes
the loaded records and deduplicated evidence corpus for downstream components.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import Config
from models import FeverRecord

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess FEVER dataset files into typed records.

    Args:
        config: Runtime configuration supplying ``fever_dataset_path`` and
            ``max_records``.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> list[FeverRecord]:
        """Load up to ``config.max_records`` records from the FEVER dataset.

        Reads the JSONL file at ``config.fever_dataset_path`` line by line.
        Each line is parsed as JSON; malformed lines or records missing
        required fields are skipped with a warning.  The nested evidence
        structure is flattened into a list of sentence strings and the label
        is normalised to uppercase.

        Returns:
            A list of :class:`~models.FeverRecord` objects.

        Raises:
            FileNotFoundError: If the dataset file does not exist at the
                configured path.
        """
        dataset_path = Path(self._config.fever_dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"FEVER dataset file not found: {self._config.fever_dataset_path}"
            )

        records: list[FeverRecord] = []
        max_records = self._config.max_records

        with dataset_path.open("r", encoding="utf-8") as fh:
            for line_index, raw_line in enumerate(fh):
                # Stop early if we have reached the requested limit
                if max_records is not None and len(records) >= max_records:
                    break

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                # --- Parse JSON ---
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping record at index %d: JSON decode error — %s",
                        line_index,
                        exc,
                    )
                    continue

                # --- Validate required fields ---
                try:
                    claim: str = obj["claim"]
                    raw_label: str = obj["label"]
                    raw_evidence = obj["evidence"]
                except KeyError as exc:
                    logger.warning(
                        "Skipping record at index %d: missing required field %s",
                        line_index,
                        exc,
                    )
                    continue

                # --- Flatten nested evidence ---
                evidence_sentences = self._flatten_evidence(raw_evidence)

                # --- Normalise label ---
                label = raw_label.upper()

                records.append(
                    FeverRecord(
                        claim=claim,
                        evidence=evidence_sentences,
                        label=label,
                        record_index=line_index,
                    )
                )

        return records

    def load_corpus(self) -> list[str]:
        """Return the flat, deduplicated list of all evidence sentences.

        Calls :meth:`load` internally and collects every evidence sentence
        across all records, removing duplicates while preserving insertion
        order.

        Returns:
            A list of unique evidence sentence strings suitable for indexing.
        """
        records = self.load()
        seen: set[str] = set()
        corpus: list[str] = []
        for record in records:
            for sentence in record.evidence:
                if sentence not in seen:
                    seen.add(sentence)
                    corpus.append(sentence)
        return corpus

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_evidence(raw_evidence: object) -> list[str]:
        """Flatten the FEVER nested evidence structure into sentence strings.

        The FEVER evidence field is a list of annotation sets, where each
        annotation set is a list of evidence items.  Each evidence item is
        itself a list whose fifth element (index 2 in the inner list after
        ``[annotation_id, evidence_id, page_title, sentence_id]``) may be a
        sentence string.  In practice the raw format varies across FEVER
        versions, so we defensively extract any string leaf values.

        Args:
            raw_evidence: The value of the ``"evidence"`` key from a FEVER
                JSON record.  Expected to be a list of lists of lists.

        Returns:
            A flat list of non-empty sentence strings.
        """
        sentences: list[str] = []

        if not isinstance(raw_evidence, list):
            return sentences

        for annotation_set in raw_evidence:
            if not isinstance(annotation_set, list):
                continue
            for evidence_item in annotation_set:
                if isinstance(evidence_item, str):
                    # Some FEVER variants store sentences directly as strings
                    if evidence_item.strip():
                        sentences.append(evidence_item.strip())
                elif isinstance(evidence_item, list):
                    # Standard FEVER format: [ann_id, ev_id, page, sent_id, ...]
                    # The sentence text (if present) is at index 4 in some
                    # versions, or the item may only contain IDs.  We extract
                    # any string element that looks like a sentence.
                    for element in evidence_item:
                        if isinstance(element, str) and element.strip():
                            sentences.append(element.strip())

        return sentences
