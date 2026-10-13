from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_score_summary(
    *,
    title: str,
    rule: str,
    metrics: Mapping[str, float],
    confusion_matrix: object,
    caveat: str,
    output_path: Path,
) -> None:
    """Render and save a presentation-ready rule-based score summary."""
    matrix = np.asarray(confusion_matrix, dtype=int)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix; got {matrix.shape}.")

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    ) * 100
    record_count = int(matrix.sum())
    positive_rate = matrix[1].sum() / record_count if record_count else 0.0

    fig, (ax_matrix, ax_metrics) = plt.subplots(
        1,
        2,
        figsize=(13, 6.5),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.90,
        f"{rule}  |  {record_count:,} hourly records  |  "
        f"SepsisLabel positive: {positive_rate:.2%}",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    image = ax_matrix.imshow(matrix, cmap="Blues")
    ax_matrix.set_title("Confusion matrix", fontsize=14, fontweight="bold", pad=12)
    class_names = ["No sepsis label", "Sepsis label"]
    ax_matrix.set_xticks([0, 1], labels=class_names)
    ax_matrix.set_yticks([0, 1], labels=class_names)
    ax_matrix.set_xlabel("Predicted by score rule", labelpad=10)
    ax_matrix.set_ylabel("Dataset SepsisLabel", labelpad=10)
    ax_matrix.tick_params(axis="x", rotation=12)

    text_threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            text_color = "white" if matrix[row, column] > text_threshold else "#0f172a"
            ax_matrix.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{row_percent[row, column]:.1f}% of row",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04, label="Records")

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_values = [float(metrics[name]) for name in metric_names]
    bars = ax_metrics.barh(
        metric_names,
        metric_values,
        color=["#64748b", "#0f766e", "#2563eb", "#7c3aed"],
        height=0.56,
    )
    ax_metrics.set_title("Detection performance", fontsize=14, fontweight="bold", pad=12)
    ax_metrics.set_xlim(0, 1.08)
    ax_metrics.set_xlabel("Score (0–1)")
    ax_metrics.grid(axis="x", linestyle="--", alpha=0.25)
    ax_metrics.set_axisbelow(True)
    ax_metrics.invert_yaxis()
    for bar, value in zip(bars, metric_values):
        ax_metrics.text(
            min(value + 0.025, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.055,
        f"Hourly records (not unique patients). {caveat} Compared with the dataset label; "
        "this is a rule-based baseline, not a clinical diagnosis.",
        ha="center",
        fontsize=9,
        color="#475569",
        wrap=True,
    )
    fig.tight_layout(rect=[0.03, 0.12, 0.97, 0.86])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Presentation figure saved to: {output_path}")
