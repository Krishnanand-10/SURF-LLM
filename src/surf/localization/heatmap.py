"""
Layer & Module Sensitivity Heatmap Generator for SURF-LLM.
Produces visual heatmaps and reports detailing knowledge localization across layers.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


def save_attribution_report(
    layer_scores: Dict[int, float],
    top_k_layers: List[int],
    module_scores: Optional[Dict[int, Dict[str, float]]] = None,
    output_path: str = "layer_scores.json",
    method: str = "gradient_saliency",
) -> Dict[str, Any]:
    """
    Saves layer and module attribution scores to a JSON report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    report = {
        "method": method,
        "top_k_layers": top_k_layers,
        "layer_scores": {str(k): float(v) for k, v in sorted(layer_scores.items())},
        "module_scores": (
            {str(k): {m: float(s) for m, s in v.items()} for k, v in sorted(module_scores.items())}
            if module_scores
            else {}
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def generate_layer_heatmaps(
    layer_scores: Dict[int, float],
    top_k_layers: Optional[List[int]] = None,
    module_scores: Optional[Dict[int, Dict[str, float]]] = None,
    save_path: str = "layer_sensitivity_heatmap.png",
    title: str = "SURF-LLM: Transformer Layer Sensitivity Profiling",
) -> Optional[str]:
    """
    Generates and saves visual heatmaps showing layer and module-wise sensitivity.
    If matplotlib is available, renders high-resolution plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for headless environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SURF-LLM] Matplotlib not available; skipping plot generation.")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    sorted_layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in sorted_layers]
    top_k = set(top_k_layers if top_k_layers is not None else [])

    has_module_breakdown = module_scores and any(len(v) > 0 for v in module_scores.values())

    if has_module_breakdown:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1.4]}
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4.5))
        ax2 = None

    # Plot 1: Layer-wise sensitivity bar chart
    bar_colors = ["#e74c3c" if l in top_k else "#3498db" for l in sorted_layers]
    bars = ax1.bar(
        [str(l) for l in sorted_layers],
        scores,
        color=bar_colors,
        edgecolor="#2c3e50",
        linewidth=0.8,
        alpha=0.85,
    )
    ax1.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attribution Score", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title} (Red = Selected Top-K Layers)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight top-K values with score labels
    max_score = max(scores) if scores else 1.0
    for bar, layer_idx, score in zip(bars, sorted_layers, scores):
        if layer_idx in top_k:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01 * max_score,
                f"L{layer_idx}\n({score:.3f})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#c0392b",
            )

    # Plot 2: 2D Layer x Module Heatmap Matrix (if module scores available)
    if ax2 is not None and module_scores:
        all_modules = sorted(list({m for l_mods in module_scores.values() for m in l_mods.keys()}))
        if all_modules:
            matrix = np.zeros((len(all_modules), len(sorted_layers)))
            for j, l in enumerate(sorted_layers):
                for i, m in enumerate(all_modules):
                    matrix[i, j] = module_scores.get(l, {}).get(m, 0.0)

            cax = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
            ax2.set_xticks(range(len(sorted_layers)))
            ax2.set_xticklabels([str(l) for l in sorted_layers])
            ax2.set_yticks(range(len(all_modules)))
            ax2.set_yticklabels(all_modules, fontsize=9)
            ax2.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Sub-Module Type", fontsize=11, fontweight="bold")
            ax2.set_title("Module-wise Attribution Breakdown Across Layers", fontsize=12, fontweight="bold")

            cbar = fig.colorbar(cax, ax=ax2, orientation="horizontal", pad=0.2, shrink=0.6)
            cbar.set_label("Sensitivity Intensity", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SURF-LLM] Heatmap saved to: {save_path}")
    return save_path
