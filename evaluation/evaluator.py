"""Evaluator for the KRR Fact Verification System.

Computes accuracy and per-class F1 for both the KRR and Baseline RAG pipelines,
produces a formatted comparison report, and writes it to a file and stdout.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from models import BaselineResult, EvaluationReport, KRRResult

logger = logging.getLogger(__name__)

# The three valid verdict labels, in a fixed order for consistent F1 indexing.
VERDICT_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Sentinel used for invalid verdicts — never matches any ground-truth label.
_INVALID_SENTINEL = "__INVALID__"


class Evaluator:
    """Compute evaluation metrics and produce comparison reports."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        krr_results: list[KRRResult],
        baseline_results: list[BaselineResult],
        ground_truth: list[str],
    ) -> EvaluationReport:
        """Compute overall accuracy and per-class F1 for both pipelines.

        Args:
            krr_results: Results from the KRR pipeline, one per claim.
            baseline_results: Results from the Baseline RAG pipeline, one per claim.
            ground_truth: Ground-truth verdict labels, one per claim.

        Returns:
            An :class:`EvaluationReport` with all computed metrics.
        """
        # Align lengths — use the minimum to avoid index errors.
        n = min(len(krr_results), len(baseline_results), len(ground_truth))

        krr_results = krr_results[:n]
        baseline_results = baseline_results[:n]
        ground_truth = ground_truth[:n]

        total_claims = n

        # Count skipped claims (either pipeline returned an error).
        skipped_claims = sum(
            1
            for krr, base in zip(krr_results, baseline_results)
            if krr.error is not None or base.error is not None
        )

        # Label distribution from ground truth.
        label_distribution: dict[str, int] = {label: 0 for label in VERDICT_LABELS}
        for gt_label in ground_truth:
            if gt_label in label_distribution:
                label_distribution[gt_label] += 1

        # Build sanitised prediction lists.
        krr_preds = [
            self._sanitise_verdict(krr.verdict, krr.claim, "KRR")
            for krr in krr_results
        ]
        baseline_preds = [
            self._sanitise_verdict(base.verdict, base.claim, "Baseline")
            for base in baseline_results
        ]

        # Compute metrics.
        krr_accuracy, krr_f1 = self._compute_metrics(krr_preds, ground_truth)
        baseline_accuracy, baseline_f1 = self._compute_metrics(
            baseline_preds, ground_truth
        )

        return EvaluationReport(
            total_claims=total_claims,
            skipped_claims=skipped_claims,
            label_distribution=label_distribution,
            krr_accuracy=krr_accuracy,
            krr_f1=krr_f1,
            baseline_accuracy=baseline_accuracy,
            baseline_f1=baseline_f1,
        )

    def save_report(self, report: EvaluationReport, path: str) -> None:
        """Write the formatted comparison table to *path* and print to stdout.

        Args:
            report: The :class:`EvaluationReport` to format and save.
            path: File path where the report should be written.
        """
        formatted = self._format_report(report)
        print(formatted)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted, encoding="utf-8")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_verdict(verdict: str, claim: str, pipeline_name: str) -> str:
        """Return the verdict unchanged if valid, otherwise log a warning and
        return the sentinel value."""
        if verdict in VERDICT_LABELS:
            return verdict
        logger.warning(
            "[%s] Invalid verdict %r for claim: %r — treating as incorrect.",
            pipeline_name,
            verdict,
            claim,
        )
        return _INVALID_SENTINEL

    @staticmethod
    def _compute_metrics(
        predictions: list[str],
        ground_truth: list[str],
    ) -> tuple[float, dict[str, float]]:
        """Compute accuracy and per-label F1.

        Returns:
            A tuple of (accuracy, {label: f1_score}).
        """
        if not predictions:
            return 0.0, {label: 0.0 for label in VERDICT_LABELS}

        accuracy: float = float(accuracy_score(ground_truth, predictions))

        # f1_score with average=None returns one score per label in the order
        # given by the `labels` parameter.
        f1_scores = f1_score(
            ground_truth,
            predictions,
            average=None,
            labels=VERDICT_LABELS,
            zero_division=0,
        )

        f1_dict: dict[str, float] = {
            label: float(score)
            for label, score in zip(VERDICT_LABELS, f1_scores)
        }
        return accuracy, f1_dict

    @staticmethod
    def _format_report(report: EvaluationReport) -> str:
        """Render the report as a human-readable comparison table."""
        dist = report.label_distribution
        dist_str = ", ".join(
            f"{label}={dist.get(label, 0)}" for label in VERDICT_LABELS
        )

        # Shorten "NOT ENOUGH INFO" to "NEI" in the column header.
        header = (
            f"{'Pipeline':<17} | {'Accuracy':^8} | {'F1-SUPPORTS':^11} | "
            f"{'F1-REFUTES':^10} | {'F1-NEI':^7}"
        )
        separator = "-" * len(header)

        def row(name: str, accuracy: float, f1: dict[str, float]) -> str:
            return (
                f"{name:<17} | {accuracy:^8.3f} | "
                f"{f1.get('SUPPORTS', 0.0):^11.3f} | "
                f"{f1.get('REFUTES', 0.0):^10.3f} | "
                f"{f1.get('NOT ENOUGH INFO', 0.0):^7.3f}"
            )

        lines = [
            "=== Evaluation Report ===",
            f"Total claims: {report.total_claims} | Skipped: {report.skipped_claims}",
            f"Label distribution: {dist_str}",
            "",
            header,
            separator,
            row("KRR", report.krr_accuracy, report.krr_f1),
            row("Baseline RAG", report.baseline_accuracy, report.baseline_f1),
        ]
        return "\n".join(lines)
