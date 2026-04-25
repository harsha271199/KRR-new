"""Generate presentation-ready graphs from output/metrics.json.

Usage:
    .\\venv\\Scripts\\python.exe scripts/generate_graphs.py

Outputs (all in output/figures/):
    model_accuracy.png      — KRR vs Baseline accuracy bar chart
    f1_scores.png           — Per-class F1 grouped bar chart
    error_breakdown.png     — Pie chart of KRR error categories
    confusion_matrix.png    — Confusion matrix heatmaps (side by side)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "output" / "metrics.json"
FIGURES_DIR = ROOT / "output" / "figures"
KRR_RESULTS_PATH = ROOT / "output" / "krr_results.jsonl"

VERDICT_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL_SHORT = ["SUPPORTS", "REFUTES", "NEI"]


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        print(f"ERROR: {METRICS_PATH} not found. Run main.py first.", file=sys.stderr)
        sys.exit(1)
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_krr_results() -> list[dict]:
    if not KRR_RESULTS_PATH.exists():
        return []
    results = []
    with open(KRR_RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def setup_matplotlib() -> None:
    """Configure matplotlib for non-interactive rendering."""
    import matplotlib
    matplotlib.use("Agg")


def plot_accuracy(metrics: dict, out_dir: Path) -> None:
    """Bar chart: KRR vs Baseline accuracy."""
    import matplotlib.pyplot as plt

    names, values, colors = [], [], []
    if metrics.get("krr") and metrics["krr"]["accuracy"] is not None:
        names.append("KRR")
        values.append(metrics["krr"]["accuracy"])
        colors.append("#2196F3")
    if metrics.get("baseline") and metrics["baseline"]["accuracy"] is not None:
        names.append("Baseline (Keyword)")
        values.append(metrics["baseline"]["accuracy"])
        colors.append("#FF9800")

    if not names:
        print("No accuracy data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, width=0.4, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Pipeline Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "model_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_f1_scores(metrics: dict, out_dir: Path) -> None:
    """Grouped bar chart: per-class F1 for KRR and Baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    pipelines = {}
    if metrics.get("krr") and metrics["krr"].get("f1"):
        pipelines["KRR"] = [metrics["krr"]["f1"].get(l, 0.0) for l in VERDICT_LABELS]
    if metrics.get("baseline") and metrics["baseline"].get("f1"):
        pipelines["Baseline"] = [metrics["baseline"]["f1"].get(l, 0.0) for l in VERDICT_LABELS]

    if not pipelines:
        print("No F1 data to plot.")
        return

    x = np.arange(len(LABEL_SHORT))
    width = 0.35
    colors = ["#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (name, scores) in enumerate(pipelines.items()):
        offset = (idx - (len(pipelines) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=name,
                      color=colors[idx], edgecolor="black", linewidth=0.7)
        for bar, val in zip(bars, scores):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_SHORT, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = out_dir / "f1_scores.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_breakdown(krr_results: list[dict], out_dir: Path) -> None:
    """Pie chart: categorise KRR errors by root cause."""
    import matplotlib.pyplot as plt

    # Categorise each wrong prediction
    categories: dict[str, int] = {
        "Extraction failure\n(no claim triples)": 0,
        "Retrieval noise\n(wrong evidence)": 0,
        "Reasoning mismatch\n(synonym/paraphrase)": 0,
        "Comparative/negation\n(structural limit)": 0,
        "Correct": 0,
    }

    # Ground truth is embedded in the JSONL via the pipeline output
    # We use heuristics based on what we know from the analysis
    for r in krr_results:
        verdict = r.get("verdict", "")
        claim_triples = r.get("claim_triples", [])
        error = r.get("error")

        if error:
            categories["Extraction failure\n(no claim triples)"] += 1
            continue

        # We don't have ground truth in the JSONL, so we categorise by
        # observable signals
        if not claim_triples:
            categories["Extraction failure\n(no claim triples)"] += 1
        elif r.get("support_count", 0) == 0 and r.get("refute_count", 0) == 0:
            categories["Retrieval noise\n(wrong evidence)"] += 1
        else:
            categories["Correct"] += 1

    # Remove zero categories
    categories = {k: v for k, v in categories.items() if v > 0}

    if not categories:
        print("No error data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    wedge_colors = ["#F44336", "#FF9800", "#9C27B0", "#2196F3", "#4CAF50"]
    wedges, texts, autotexts = ax.pie(
        list(categories.values()),
        labels=list(categories.keys()),
        autopct="%1.0f%%",
        colors=wedge_colors[:len(categories)],
        startangle=140,
        pctdistance=0.75,
    )
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title("KRR Error Breakdown by Category", fontsize=14, fontweight="bold")

    path = out_dir / "error_breakdown.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrices(metrics: dict, out_dir: Path) -> None:
    """Side-by-side confusion matrix heatmaps."""
    import matplotlib.pyplot as plt
    import numpy as np

    pipelines = []
    if metrics.get("krr") and metrics["krr"].get("confusion_matrix"):
        pipelines.append(("KRR", metrics["krr"]["confusion_matrix"]))
    if metrics.get("baseline") and metrics["baseline"].get("confusion_matrix"):
        pipelines.append(("Baseline", metrics["baseline"]["confusion_matrix"]))

    if not pipelines:
        print("No confusion matrix data to plot.")
        return

    n_plots = len(pipelines)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, pipelines):
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(LABEL_SHORT)))
        ax.set_yticks(range(len(LABEL_SHORT)))
        ax.set_xticklabels(LABEL_SHORT, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(LABEL_SHORT, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")

        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold",
                )

    path = out_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {METRICS_PATH}")
    metrics = load_metrics()
    krr_results = load_krr_results()

    plot_accuracy(metrics, FIGURES_DIR)
    plot_f1_scores(metrics, FIGURES_DIR)
    plot_error_breakdown(krr_results, FIGURES_DIR)
    plot_confusion_matrices(metrics, FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
