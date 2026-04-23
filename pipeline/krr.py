"""KRR Pipeline: orchestrates Retriever → TripleExtractor → KnowledgeGraph → ReasoningModule."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from config import Config
from knowledge.extractor import TripleExtractor
from knowledge.graph import KnowledgeGraph
from models import KRRResult, Triple
from reasoning.reasoner import ReasoningModule
from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class KRRPipeline:
    """Orchestrate the full KRR fact-verification pipeline for a single claim.

    Each call to :meth:`run` is independent: a fresh :class:`KnowledgeGraph`
    is created per claim so no mutable state leaks between executions.

    Args:
        retriever: A pre-built :class:`Retriever` instance.
        extractor: A :class:`TripleExtractor` instance.
        reasoner: A :class:`ReasoningModule` instance.
        config: Runtime configuration (used for output paths).
    """

    def __init__(
        self,
        retriever: Retriever,
        extractor: TripleExtractor,
        reasoner: ReasoningModule,
        config: Config,
    ) -> None:
        self._retriever = retriever
        self._extractor = extractor
        self._reasoner = reasoner
        self._config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, claim: str) -> KRRResult:
        """Run the full pipeline for *claim*.

        Stages (in order):
        1. Retriever.retrieve(claim) → top-K evidence sentences
        2. TripleExtractor.extract(claim) → claim triples
        3. TripleExtractor.extract_batch(evidence) → evidence triples (flattened)
        4. Build a fresh KnowledgeGraph; add claim triples to "claim" namespace
           and evidence triples to "evidence" namespace
        5. ReasoningModule.reason(claim_triples, evidence_triples) → ReasoningResult
        6. Assemble and return a KRRResult

        On any unhandled exception the pipeline logs the claim and stage name,
        then returns a ``KRRResult`` with ``verdict="NOT ENOUGH INFO"``, empty
        attribution, and the error message in ``error``.

        Args:
            claim: The natural language claim to verify.

        Returns:
            A fully populated :class:`KRRResult`.
        """
        stage = "initialization"
        retrieved_evidence: list[str] = []
        claim_triples: list[Triple] = []
        evidence_triples: list[Triple] = []

        try:
            # Stage 1 — retrieval
            stage = "retrieval"
            retrieved_evidence = self._retriever.retrieve(claim)

            # Stage 2 — claim triple extraction
            stage = "claim triple extraction"
            claim_triples = self._extractor.extract(claim)

            # Stage 3 — evidence triple extraction (flatten list-of-lists)
            stage = "evidence triple extraction"
            evidence_triples_nested = self._extractor.extract_batch(retrieved_evidence)
            evidence_triples = [t for sublist in evidence_triples_nested for t in sublist]

            # Stage 4 — knowledge graph construction (fresh per call)
            stage = "knowledge graph construction"
            kg = KnowledgeGraph()
            for triple in claim_triples:
                kg.add_triple(triple, namespace="claim")
            for triple in evidence_triples:
                kg.add_triple(triple, namespace="evidence")

            # Stage 5 — reasoning
            stage = "reasoning"
            reasoning_result = self._reasoner.reason(claim_triples, evidence_triples)

            # Stage 6 — assemble result
            stage = "result assembly"
            result = KRRResult(
                claim=claim,
                retrieved_evidence=retrieved_evidence,
                claim_triples=claim_triples,
                evidence_triples=evidence_triples,
                support_count=reasoning_result.support_count,
                refute_count=reasoning_result.refute_count,
                irrelevant_count=reasoning_result.irrelevant_count,
                verdict=reasoning_result.verdict,
                attribution=reasoning_result.attribution,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "KRRPipeline error at stage %r for claim %r: %s",
                stage,
                claim,
                exc,
            )
            result = KRRResult(
                claim=claim,
                retrieved_evidence=retrieved_evidence,
                claim_triples=claim_triples,
                evidence_triples=evidence_triples,
                support_count=0,
                refute_count=0,
                irrelevant_count=0,
                verdict="NOT ENOUGH INFO",
                attribution=[],
                error=str(exc),
            )

        # Output side-effects (never raise — log and continue)
        self._write_json_output(result)
        self._write_stdout_output(result)

        return result

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _result_to_dict(self, result: KRRResult) -> dict:
        """Convert a KRRResult to a JSON-serializable dict."""
        return {
            "claim": result.claim,
            "retrieved_evidence": result.retrieved_evidence,
            "claim_triples": [t.to_dict() for t in result.claim_triples],
            "evidence_triples": [t.to_dict() for t in result.evidence_triples],
            "support_count": result.support_count,
            "refute_count": result.refute_count,
            "irrelevant_count": result.irrelevant_count,
            "verdict": result.verdict,
            "attribution": [t.to_dict() for t in result.attribution],
            "error": result.error,
        }

    def _write_json_output(self, result: KRRResult) -> None:
        """Append the per-claim dict to the JSON output file (newline-delimited JSON).

        Each call appends one JSON object per line to ``config.json_output_path``.
        If the path is not configured this is a no-op.
        """
        path = self._config.json_output_path
        if not path:
            return

        try:
            record = json.dumps(self._result_to_dict(result), ensure_ascii=False)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(record + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write JSON output to %r: %s", path, exc)

    def _write_stdout_output(self, result: KRRResult) -> None:
        """Print a human-readable summary of the per-claim result to stdout."""
        lines: list[str] = [
            "=" * 60,
            f"CLAIM      : {result.claim}",
            "",
            "EVIDENCE   :",
        ]
        if result.retrieved_evidence:
            for i, ev in enumerate(result.retrieved_evidence, 1):
                lines.append(f"  [{i}] {ev}")
        else:
            lines.append("  (none retrieved)")

        lines.append("")
        lines.append("CLAIM TRIPLES:")
        if result.claim_triples:
            for t in result.claim_triples:
                lines.append(f"  {t.to_canonical_string()}")
        else:
            lines.append("  (none extracted)")

        lines.append("")
        lines.append("EVIDENCE TRIPLES:")
        if result.evidence_triples:
            for t in result.evidence_triples:
                lines.append(f"  {t.to_canonical_string()}")
        else:
            lines.append("  (none extracted)")

        lines.append("")
        lines.append(f"SUPPORT COUNT  : {result.support_count}")
        lines.append(f"REFUTE COUNT   : {result.refute_count}")
        lines.append(f"IRRELEVANT COUNT: {result.irrelevant_count}")
        lines.append(f"VERDICT        : {result.verdict}")

        lines.append("")
        lines.append("ATTRIBUTION:")
        if result.attribution:
            for t in result.attribution:
                lines.append(f"  {t.to_canonical_string()}")
        else:
            lines.append("  (none)")

        if result.error:
            lines.append("")
            lines.append(f"ERROR: {result.error}")

        lines.append("=" * 60)

        print("\n".join(lines), file=sys.stdout)
