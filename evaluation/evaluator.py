"""Evaluator for the KRR Fact Verification System.

Computes accuracy, precision, recall, and per-class F1 for both pipelines,
produces a formatted comparison report, writes it to a file and stdout,
and saves machine-readable metrics.json and confusion_matrix.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models import BaselineResult, EvaluationReport, KRRResult

logger = logging.getLogger(__name__)

VERDICT_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
_INVALID_SENTINEL = "__INVALID__"


class Evaluator:
    """Compute evaluation metrics and produce comparison reports."""

    def evaluate(
        self,
        krr_results: list[KRRResult] | None,
        baseline_results: list[BaselineResult] | None,
        ground_truth: list[str],
    ) -> EvaluationReport:
        """Compute accuracy, precision, recall, and per-class F1.

        Args:
            krr_results: Results from the KRR pipeline, or None if not run.
            baseline_results: Results from the Baseline RAG pipeline, or None.
            ground_truth: Ground-truth verdict labels, one per claim.

        Returns:
            An EvaluationReport with all computed metrics.
        """
        result_lengths = [
            len(r) for r in (krr_results, baseline_results) if r is not None
        ]
        n = min(*result_lengths, len(ground_truth)) if result_lengths else 0
        ground_truth = ground_truth[:n]

        skipped_claims = 0
        if krr_results is not None:
            skipped_claims = max(
                skipped_claims,
                sum(1 for r in krr_results[:n] if r.error is not None),
            )
        if baseline_results is not None:
            skipped_claims = max(
                skipped_claims,
                sum(1 for r in baseline_results[:n] if r.error is not None),
            )

        label_distribution: dict[str, int] = {label: 0 for label in VERDICT_LABELS}
        for gt_label in ground_truth:
            if gt_label in label_distribution:
                label_distribution[gt_label] += 1

        krr_accuracy = krr_f1 = None
        krr_precision = krr_recall = None
        krr_confusion: list[list[int]] | None = None
        krr_preds: list[str] = []
        if krr_results is not None:
            krr_preds = [
                self._sanitise_verdict(r.verdict, r.claim, "KRR")
                for r in krr_results[:n]
            ]
            krr_accuracy, krr_f1, krr_precision, krr_recall, krr_confusion = (
                self._compute_metrics(krr_preds, ground_truth)
            )

        baseline_accuracy = baseline_f1 = None
        baseline_precision = baseline_recall = None
        baseline_confusion: list[list[int]] | None = None
        baseline_preds: list[str] = []
        if baseline_results is not None:
            baseline_preds = [
                self._sanitise_verdict(r.verdict, r.claim, "Baseline")
                for r in baseline_results[:n]
            ]
            (
                baseline_accuracy,
                baseline_f1,
                baseline_precision,
                baseline_recall,
                baseline_confusion,
            ) = self._compute_metrics(baseline_preds, ground_truth)

        return EvaluationReport(
            total_claims=n,
            skipped_claims=skipped_claims,
            label_distribution=label_distribution,
            krr_accuracy=krr_accuracy,
            krr_f1=krr_f1,
            krr_precision=krr_precision,
            krr_recall=krr_recall,
            krr_confusion=krr_confusion,
            krr_predictions=krr_preds,
            baseline_accuracy=baseline_accuracy,
            baseline_f1=baseline_f1,
            baseline_precision=baseline_precision,
            baseline_recall=baseline_recall,
            baseline_confusion=baseline_confusion,
            baseline_predictions=baseline_preds,
            ground_truth=list(ground_truth),
        )

    def save_report(self, report: EvaluationReport, path: str) -> None:
        """Write the formatted comparison table to *path* and print to stdout.

        Also writes:
        - output/metrics.json  — machine-readable metrics
        - output/confusion_matrix.csv — confusion matrices for both pipelines
        """
        formatted = self._format_report(report)
        print(formatted)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted, encoding="utf-8")

        # Write metrics.json
        metrics_path = output_path.parent / "metrics.json"
        self._save_metrics_json(report, metrics_path)

        # Write confusion_matrix.csv
        cm_path = output_path.parent / "confusion_matrix.csv"
        self._save_confusion_matrix_csv(report, cm_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_verdict(verdict: str, claim: str, pipeline_name: str) -> str:
        if verdict in VERDICT_LABELS:
            return verdict
        logger.warning(
            "[%s] Invalid verdict %r for claim: %r — treating as incorrect.",
            pipeline_name, verdict, claim,
        )
        return _INVALID_SENTINEL

    @staticmethod
    def _compute_metrics(
        predictions: list[str],
        ground_truth: list[str],
    ) -> tuple[
        float,
        dict[str, float],
        dict[str, float],
        dict[str, float],
        list[list[int]],
    ]:
        """Compute accuracy, per-label F1, precision, recall, and confusion matrix."""
        if not predictions:
            empty = {label: 0.0 for label in VERDICT_LABELS}
            empty_cm = [[0] * len(VERDICT_LABELS)] * len(VERDICT_LABELS)
            return 0.0, empty, empty, empty, empty_cm

        accuracy = float(accuracy_score(ground_truth, predictions))

        f1_scores = f1_score(
            ground_truth, predictions,
            average=None, labels=VERDICT_LABELS, zero_division=0,
        )
        precision_scores = precision_score(
            ground_truth, predictions,
            average=None, labels=VERDICT_LABELS, zero_division=0,
        )
        recall_scores = recall_score(
            ground_truth, predictions,
            average=None, labels=VERDICT_LABELS, zero_division=0,
        )
        cm = confusion_matrix(ground_truth, predictions, labels=VERDICT_LABELS)

        f1_dict = {label: float(s) for label, s in zip(VERDICT_LABELS, f1_scores)}
        prec_dict = {label: float(s) for label, s in zip(VERDICT_LABELS, precision_scores)}
        rec_dict = {label: float(s) for label, s in zip(VERDICT_LABELS, recall_scores)}
        cm_list = cm.tolist()

        return accuracy, f1_dict, prec_dict, rec_dict, cm_list

    @staticmethod
    def _format_report(report: EvaluationReport) -> str:
        dist = report.label_distribution
        dist_str = ", ".join(f"{label}={dist.get(label, 0)}" for label in VERDICT_LABELS)

        header = (
            f"{'Pipeline':<17} | {'Accuracy':^8} | {'F1-SUPPORTS':^11} | "
            f"{'F1-REFUTES':^10} | {'F1-NEI':^7} | {'Precision':^9} | {'Recall':^6}"
        )
        separator = "-" * len(header)

        def row(
            name: str,
            accuracy: float | None,
            f1: dict[str, float] | None,
            prec: dict[str, float] | None,
            rec: dict[str, float] | None,
        ) -> str:
            if accuracy is None or f1 is None:
                return f"{name:<17} | {'N/A (not run)':^63}"
            macro_prec = sum((prec or {}).values()) / len(VERDICT_LABELS) if prec else 0.0
            macro_rec = sum((rec or {}).values()) / len(VERDICT_LABELS) if rec else 0.0
            return (
                f"{name:<17} | {accuracy:^8.3f} | "
                f"{f1.get('SUPPORTS', 0.0):^11.3f} | "
                f"{f1.get('REFUTES', 0.0):^10.3f} | "
                f"{f1.get('NOT ENOUGH INFO', 0.0):^7.3f} | "
                f"{macro_prec:^9.3f} | {macro_rec:^6.3f}"
            )

        lines = [
            "=== Evaluation Report ===",
            f"Total claims: {report.total_claims} | Skipped: {report.skipped_claims}",
            f"Label distribution: {dist_str}",
            "",
            header,
            separator,
            row("KRR", report.krr_accuracy, report.krr_f1,
                report.krr_precision, report.krr_recall),
            row("Baseline RAG", report.baseline_accuracy, report.baseline_f1,
                report.baseline_precision, report.baseline_recall),
        ]

        # Confusion matrices
        for name, cm, preds in [
            ("KRR", report.krr_confusion, report.krr_predictions),
            ("Baseline RAG", report.baseline_confusion, report.baseline_predictions),
        ]:
            if cm is None:
                continue
            lines.append("")
            lines.append(f"Confusion Matrix — {name}:")
            lines.append(f"  {'':20} {'SUPPORTS':>10} {'REFUTES':>10} {'NEI':>10}")
            for i, label in enumerate(VERDICT_LABELS):
                short = label if label != "NOT ENOUGH INFO" else "NEI"
                lines.append(
                    f"  {'Actual ' + short:20} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}"
                )

        return "\n".join(lines)

    @staticmethod
    def _save_metrics_json(report: EvaluationReport, path: Path) -> None:
        data = {
            "total_claims": report.total_claims,
            "skipped_claims": report.skipped_claims,
            "label_distribution": report.label_distribution,
            "krr": {
                "accuracy": report.krr_accuracy,
                "f1": report.krr_f1,
                "precision": report.krr_precision,
                "recall": report.krr_recall,
                "confusion_matrix": report.krr_confusion,
            } if report.krr_accuracy is not None else None,
            "baseline": {
                "accuracy": report.baseline_accuracy,
                "f1": report.baseline_f1,
                "precision": report.baseline_precision,
                "recall": report.baseline_recall,
                "confusion_matrix": report.baseline_confusion,
            } if report.baseline_accuracy is not None else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Metrics written to %s", path)

    @staticmethod
    def _save_confusion_matrix_csv(report: EvaluationReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for name, cm in [("KRR", report.krr_confusion),
                              ("Baseline", report.baseline_confusion)]:
                if cm is None:
                    continue
                writer.writerow([f"Pipeline: {name}"])
                writer.writerow(["Actual \\ Predicted"] + VERDICT_LABELS)
                for i, label in enumerate(VERDICT_LABELS):
                    writer.writerow([label] + cm[i])
                writer.writerow([])
        logger.info("Confusion matrix written to %s", path)
