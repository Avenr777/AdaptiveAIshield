"""
tools/generate_new_figures.py — Generate publication-quality figures for all new
experiments run on Oct 9 2026:
  - Layer ablation (5 configs on 500 Kaggle prompts)
  - Strong baselines comparison (4 systems)
  - Adversary query-budget study
  - Confirmation policy study
  - Latency scaling (corpus size)
  - Validation calibration (BandConfig v1 vs v2)

All saved to publication_figures_new/ at 300 DPI.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = Path("publication_figures_new")
OUT_DIR.mkdir(exist_ok=True)

DPI = 300
FONT_SIZE = 11
TITLE_SIZE = 13
COLORS = {
    "Lexical-Only":                "#d62728",   # red
    "Semantic-Only":               "#1f77b4",   # blue
    "Lexical + Semantic":          "#ff7f0e",   # orange
    "AIShield-Static":             "#2ca02c",   # green
    "AIShield-Static (Proposed)":  "#2ca02c",   # green (same)
    "AIShield-Adaptive":           "#9467bd",   # purple
    "TF-IDF + Logistic Regression":"#8c564b",   # brown
    "Rebuff / Vigil Embedding Memory": "#e377c2",# pink
    "ProtectAI DeBERTa-v3 v2":    "#7f7f7f",   # gray
}

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": DPI,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Layer Ablation: Detection, FPR, F1, Latency (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig_layer_ablation():
    with open("layer_ablation_results.json") as f:
        data = json.load(f)

    configs = ["Lexical-Only", "Semantic-Only", "Lexical + Semantic",
               "AIShield-Static", "AIShield-Adaptive"]
    short_labels = ["Lexical\nOnly", "Semantic\nOnly", "Lex +\nSem",
                    "AIShield\nStatic", "AIShield\nAdaptive"]

    det     = [data[c]["detection_rate"] for c in configs]
    det_lo  = [data[c]["detection_rate"] - data[c]["detection_ci95"][0] for c in configs]
    det_hi  = [data[c]["detection_ci95"][1] - data[c]["detection_rate"] for c in configs]
    fpr     = [data[c]["fpr"] for c in configs]
    fpr_lo  = [data[c]["fpr"] - data[c]["fpr_ci95"][0] for c in configs]
    fpr_hi  = [data[c]["fpr_ci95"][1] - data[c]["fpr"] for c in configs]
    f1      = [data[c]["f1_score"] for c in configs]
    lat     = [data[c]["latency_p50_ms"] for c in configs]
    colors  = [COLORS.get(c, "#333333") for c in configs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Figure 1 — Layer Ablation on 500 Kaggle Prompts (250 Malicious, 250 Benign)",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.01)

    x = np.arange(len(configs))

    # Panel A: Detection Rate
    ax = axes[0, 0]
    bars = ax.bar(x, det, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x, det, yerr=[det_lo, det_hi], fmt="none", color="black",
                capsize=4, linewidth=1.2, zorder=4)
    ax.set_title("(A) Detection Rate (%)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 115)
    ax.axhline(97.6, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, det):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: False Positive Rate
    ax = axes[0, 1]
    bars = ax.bar(x, fpr, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x, fpr, yerr=[fpr_lo, fpr_hi], fmt="none", color="black",
                capsize=4, linewidth=1.2, zorder=4)
    ax.set_title("(B) False Positive Rate (%)")
    ax.set_ylabel("FPR (%)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, fpr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel C: F1 Score
    ax = axes[1, 0]
    bars = ax.bar(x, f1, color=colors, alpha=0.85, zorder=3)
    ax.set_title("(C) F1-Score (%)")
    ax.set_ylabel("F1-Score (%)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel D: Latency p50
    ax = axes[1, 1]
    bars = ax.bar(x, lat, color=colors, alpha=0.85, zorder=3, log=True)
    ax.set_title("(D) Median Latency p50 (ms, log scale)")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig_new1_layer_ablation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-category detection heatmap (layer ablation)
# ─────────────────────────────────────────────────────────────────────────────
def fig_category_heatmap():
    with open("layer_ablation_results.json") as f:
        data = json.load(f)

    configs = ["Lexical-Only", "Semantic-Only", "Lexical + Semantic",
               "AIShield-Static", "AIShield-Adaptive"]
    short_labels = ["Lexical\nOnly", "Semantic\nOnly", "Lex + Sem",
                    "AIShield\nStatic", "AIShield\nAdaptive"]
    categories = ["code_execution", "obfuscation", "data_leakage",
                  "role_playing", "jailbreaking"]
    cat_labels  = ["Code Execution\n(n=146)", "Obfuscation\n(n=61)",
                   "Data Leakage\n(n=18)", "Role-Playing\n(n=8)", "Jailbreaking\n(n=17)"]

    matrix = np.zeros((len(categories), len(configs)))
    for j, cfg in enumerate(configs):
        cat_data = data[cfg].get("category_counts", {})
        for i, cat in enumerate(categories):
            cd = cat_data.get(cat, {})
            total = cd.get("total", 0)
            warned = cd.get("WARN", 0)
            blocked = cd.get("BLOCK", 0)
            matrix[i, j] = (warned + blocked) / total * 100.0 if total else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Detection Rate (%)")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(cat_labels, fontsize=9)
    ax.set_title("Figure 2 — Per-Category Detection Rate Heatmap (Layer Ablation)\n"
                 "500 Kaggle Prompts — Green = 100% Detection, Red = 0%",
                 fontsize=TITLE_SIZE, fontweight="bold")

    for i in range(len(categories)):
        for j in range(len(configs)):
            val = matrix[i, j]
            txt_color = "black" if 20 < val < 80 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_color)

    fig.tight_layout()
    out = OUT_DIR / "fig_new2_category_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Strong Baselines: Pareto plot (Detection vs FPR) + Latency bar
# ─────────────────────────────────────────────────────────────────────────────
def fig_strong_baselines():
    with open("strong_baselines_benchmark.json") as f:
        data = json.load(f)

    systems = list(data.keys())
    det = [data[s]["detection_rate"] for s in systems]
    fpr = [data[s]["fpr"] for s in systems]
    lat = [data[s]["p50_latency_ms"] for s in systems]
    short_map = {
        "TF-IDF + Logistic Regression": "TF-IDF\n+LogReg",
        "Rebuff / Vigil Embedding Memory": "Rebuff/\nVigil",
        "ProtectAI DeBERTa-v3 v2": "DeBERTa\nv3-v2",
        "AIShield-Static (Proposed)": "AIShield\nStatic",
        "PIGuard (Boundary + Pattern Baseline)": "PIGuard\n(Pattern)",
        "Llama Prompt Guard 86M (Meta - Jailbreak Mode)": "Llama PG\n86M (Jailbreak)",
        "Llama Prompt Guard 86M (Meta - Out-of-the-Box Argmax)": "Llama PG\n86M (Argmax)",
    }
    short = [short_map.get(s, s[:10]) for s in systems]
    colors = [COLORS.get(s, "#4a7bb0") for s in systems]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 3 — Strong Baselines on 500 Real-World Kaggle Prompts",
                 fontsize=TITLE_SIZE, fontweight="bold")

    # Panel A: Detection
    ax = axes[0]
    bars = ax.bar(range(len(systems)), det, color=colors, alpha=0.85, zorder=3)
    ax.set_title("(A) Detection Rate (%)")
    ax.set_xticks(range(len(systems))); ax.set_xticklabels(short)
    ax.set_ylim(0, 115); ax.set_ylabel("Detection Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, det):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: FPR
    ax = axes[1]
    bars = ax.bar(range(len(systems)), fpr, color=colors, alpha=0.85, zorder=3)
    ax.set_title("(B) False Positive Rate (%)\n(Lower is better)")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(short, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("FPR (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, fpr):
        ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + 2, 102),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel C: Latency (log)
    ax = axes[2]
    bars = ax.bar(range(len(systems)), lat, color=colors, alpha=0.85, zorder=3, log=True)
    ax.set_title("(C) Median Latency p50 (ms, log scale)\n(Lower is better)")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(short, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_ylim(0.005, 500)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.35,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_xticklabels(short, rotation=15, ha="right", fontsize=8)

    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    out = OUT_DIR / "fig_new3_strong_baselines.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Pareto Scatter: Detection vs Latency, bubble = FPR
# ─────────────────────────────────────────────────────────────────────────────
def fig_pareto_scatter():
    with open("strong_baselines_benchmark.json") as f:
        bdata = json.load(f)
    with open("layer_ablation_results.json") as f:
        ldata = json.load(f)

    systems = {
        "TF-IDF + Logistic Regression": bdata["TF-IDF + Logistic Regression"],
        "Rebuff / Vigil Embedding Memory": bdata["Rebuff / Vigil Embedding Memory"],
        "ProtectAI DeBERTa-v3 v2": bdata["ProtectAI DeBERTa-v3 v2"],
        "Lexical-Only (Ablation)": {
            "detection_rate": ldata["Lexical-Only"]["detection_rate"],
            "fpr": ldata["Lexical-Only"]["fpr"],
            "p50_latency_ms": ldata["Lexical-Only"]["latency_p50_ms"],
        },
        "Semantic-Only (Ablation)": {
            "detection_rate": ldata["Semantic-Only"]["detection_rate"],
            "fpr": ldata["Semantic-Only"]["fpr"],
            "p50_latency_ms": ldata["Semantic-Only"]["latency_p50_ms"],
        },
        "AIShield-Static": bdata["AIShield-Static (Proposed)"],
    }

    color_map = {
        "TF-IDF + Logistic Regression": "#8c564b",
        "Rebuff / Vigil Embedding Memory": "#e377c2",
        "ProtectAI DeBERTa-v3 v2": "#7f7f7f",
        "Lexical-Only (Ablation)": "#d62728",
        "Semantic-Only (Ablation)": "#1f77b4",
        "AIShield-Static": "#2ca02c",
    }
    markers = {
        "TF-IDF + Logistic Regression": "s",
        "Rebuff / Vigil Embedding Memory": "^",
        "ProtectAI DeBERTa-v3 v2": "D",
        "Lexical-Only (Ablation)": "x",
        "Semantic-Only (Ablation)": "o",
        "AIShield-Static": "*",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Figure 4 — Pareto Frontier: Detection Rate vs Latency\n"
                 "(Bubble area ∝ FPR — ideal = top-left, small bubble)",
                 fontsize=TITLE_SIZE, fontweight="bold")

    for name, vals in systems.items():
        x = vals["p50_latency_ms"]
        y = vals["detection_rate"]
        fpr_val = vals["fpr"]
        bubble = max(fpr_val * 50 + 30, 40)
        ax.scatter(x, y, s=bubble, color=color_map[name], marker=markers[name],
                   label=f"{name} (FPR={fpr_val:.1f}%)", zorder=5, alpha=0.9,
                   edgecolors="black", linewidth=0.8)
        offset_x = x * 0.05 + 0.5
        ax.annotate(name.replace(" (Ablation)", "").replace(" (Proposed)", ""),
                    xy=(x, y), xytext=(x + offset_x, y - 3),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_xscale("log")
    ax.set_xlabel("Median Latency p50 (ms, log scale)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(-5, 110)
    ax.set_xlim(0.02, 200)
    ax.axhline(97.6, color="#2ca02c", linestyle="--", linewidth=0.8,
               alpha=0.4, label="AIShield-Static detection level")
    ax.axvline(10, color="gray", linestyle=":", linewidth=0.8,
               alpha=0.4, label="10 ms latency threshold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    out = OUT_DIR / "fig_new4_pareto_scatter.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Adversary Query Budget: ASR vs Budget
# ─────────────────────────────────────────────────────────────────────────────
def fig_adversary_budget():
    with open("adversary_and_confirmation_results.json") as f:
        data = json.load(f)["black_box_query_budget_search"]

    budgets = [1, 5, 10, 20, 50]
    systems = list(data.keys())
    sys_colors = {"Lexical-Only": "#d62728", "AIShield-Static": "#2ca02c",
                  "AIShield-Adaptive": "#9467bd"}
    sys_markers = {"Lexical-Only": "x", "AIShield-Static": "o", "AIShield-Adaptive": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 5 — Black-Box Adversary: Independent Query Budget Evaluations (B=1 to 50)",
                 fontsize=TITLE_SIZE, fontweight="bold")

    # Panel A: ASR vs Budget (discrete points without connecting lines)
    ax = axes[0]
    for sys in systems:
        asrs = [data[sys][f"budget_{b}"]["asr_pct"] for b in budgets]
        ax.scatter(budgets, asrs, marker=sys_markers.get(sys, "o"), s=90, label=sys,
                   color=sys_colors.get(sys, "#333"), zorder=5, alpha=0.9)
    ax.set_title("(A) Attack Success Rate (ASR) Across Independent Budgets\n(Discrete points: independent search evaluations per budget)")
    ax.set_xlabel("Query Budget B (Independent Evaluations)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-5, 115)
    ax.set_xticks(budgets)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel B: Detection vs Budget
    ax = axes[1]
    for sys in systems:
        dets = [data[sys][f"budget_{b}"]["detected_pct"] for b in budgets]
        ax.scatter(budgets, dets, marker=sys_markers.get(sys, "s"), s=90, label=sys,
                   color=sys_colors.get(sys, "#333"), zorder=5, alpha=0.9)
    ax.set_title("(B) Detection Rate Across Independent Budgets\n(Higher = better defender)")
    ax.set_xlabel("Query Budget B (Independent Evaluations)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_ylim(-5, 115)
    ax.set_xticks(budgets)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "fig_new5_adversary_budget.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Confirmation Policy Study
# ─────────────────────────────────────────────────────────────────────────────
def fig_confirmation_policy():
    with open("adversary_and_confirmation_results.json") as f:
        data = json.load(f)["realistic_confirmation_and_noise"]

    modes = ["oracle", "selective_review", "noise_5pct", "noise_10pct"]
    labels = ["Oracle\n(100% reviewed)", "Selective Review\n(WARN/BLOCK + 20% audit)",
              "Label Noise\n5%", "Label Noise\n10%"]
    det_vals = [data[m]["detection_pct"] for m in modes]
    asr_vals = [data[m]["asr_pct"] for m in modes]
    mode_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 6 — Realistic Confirmation Policy Study (X6, C8)\n"
                 "4 sequential attack rounds × 5 payloads = 20 total queries per mode",
                 fontsize=TITLE_SIZE, fontweight="bold")

    x = np.arange(len(modes))

    ax = axes[0]
    bars = ax.bar(x, det_vals, color=mode_colors, alpha=0.85, zorder=3)
    ax.set_title("(A) Detection Rate by Confirmation Policy")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, det_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, asr_vals, color=mode_colors, alpha=0.85, zorder=3)
    ax.set_title("(B) Attack Success Rate by Confirmation Policy")
    ax.set_ylabel("ASR (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, asr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig_new6_confirmation_policy.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Latency Scaling: Corpus Size N vs Retrieval Latency
# ─────────────────────────────────────────────────────────────────────────────
def fig_latency_scaling():
    with open("latency_benchmark_hardware.json") as f:
        data = json.load(f)

    rs = data["Retrieval_Memory_Scaling"]
    ns = [10, 100, 1000, 10000]
    keys = ["N_10", "N_100", "N_1000", "N_10000"]

    p50 = [rs[k]["median_p50_ms"] for k in keys]
    p95 = [rs[k]["p95_ms"] for k in keys]
    p99 = [rs[k]["p99_ms"] for k in keys]
    means = [rs[k]["mean_ms"] for k in keys]

    # Reference lines
    aishield_static_p50 = data["AIShield_Static_Pipeline"]["median_p50_ms"]
    lexical_p50 = data["Lexical_Layer"]["median_p50_ms"]
    deberta_p50 = 82.226  # from strong baselines

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Figure 7 — Retrieval Memory Latency vs Corpus Size\n"
                 "Hardware: Intel i7-12700K (12-core), Windows 11 | 200 repetitions, 50 warm-up discarded",
                 fontsize=TITLE_SIZE, fontweight="bold")

    ax.plot(ns, p50, marker="o", color="#2ca02c", linewidth=2.2, markersize=8, label="p50 (Median)")
    ax.plot(ns, p95, marker="s", color="#ff7f0e", linewidth=1.8, markersize=7, linestyle="--", label="p95")
    ax.plot(ns, p99, marker="^", color="#d62728", linewidth=1.5, markersize=7, linestyle=":", label="p99")
    ax.fill_between(ns, p50, p99, alpha=0.10, color="#2ca02c")

    ax.axhline(aishield_static_p50, color="#2ca02c", linestyle="-.", linewidth=1.2, alpha=0.7,
               label=f"AIShield-Static p50 ({aishield_static_p50:.2f} ms)")
    ax.axhline(deberta_p50, color="#7f7f7f", linestyle="-.", linewidth=1.2, alpha=0.7,
               label=f"ProtectAI DeBERTa p50 ({deberta_p50:.1f} ms)")
    ax.axhline(lexical_p50, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.6,
               label=f"Lexical-Only p50 ({lexical_p50:.3f} ms)")

    for n_val, p50_val, p95_val in zip(ns, p50, p95):
        ax.annotate(f"{p50_val:.3f} ms", xy=(n_val, p50_val),
                    xytext=(n_val * 1.05, p50_val + 0.3), fontsize=8,
                    color="#2ca02c", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Corpus Size N (log scale)")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(-0.3, min(deberta_p50 * 1.3, 120))
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out = OUT_DIR / "fig_new7_latency_scaling.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Validation Calibration: ROC-AUC + BandConfig comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_validation_calibration():
    with open("validation_calibration_metrics.json") as f:
        data = json.load(f)

    v1 = data["BandConfig_v1"]["metrics"]
    v2 = data["BandConfig_v2"]["metrics"]

    metrics = ["Recall (%)", "FPR (%)", "ROC-AUC (×100)", "PR-AUC (×100)", "TPR@1%FPR (%)"]
    v1_vals = [v1["measured_recall_pct"], v1["measured_fpr_pct"],
               v1["roc_auc"] * 100, v1["pr_auc"] * 100, v1["tpr_at_1pct_fpr"]]
    v2_vals = [v2["measured_recall_pct"], v2["measured_fpr_pct"],
               v2["roc_auc"] * 100, v2["pr_auc"] * 100, v2["tpr_at_1pct_fpr"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 8 — BandConfig Calibration on Held-Out Validation Split\n"
                 "val.jsonl: n=111 (76 malicious, 35 benign) — Selected on val, test touched once",
                 fontsize=TITLE_SIZE, fontweight="bold")

    # Panel A: grouped bars
    ax = axes[0]
    b1 = ax.bar(x - width/2, v1_vals, width, label="BandConfig v1 (T_low=50)", color="#ff7f0e", alpha=0.85)
    b2 = ax.bar(x + width/2, v2_vals, width, label="BandConfig v2 Selected (T_low=30)", color="#2ca02c", alpha=0.85)
    ax.set_title("(A) Validation Metrics: v1 vs v2")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Panel B: confusion matrix counts
    ax = axes[1]
    for idx, (label, vcounts) in enumerate([
        ("BandConfig v1", v1["counts"]),
        ("BandConfig v2\n(Selected)", v2["counts"])
    ]):
        tp = vcounts["TP"]; fn = vcounts["FN"]; fp = vcounts["FP"]; tn = vcounts["TN"]
        cm = np.array([[tp, fn], [fp, tn]], dtype=float)
        im_off = idx * 2.5
        for i in range(2):
            for j in range(2):
                cell_labels = [["TP", "FN"], ["FP", "TN"]]
                row_labels = ["Malicious (76)", "Benign (35)"]
                col_labels = ["Predicted\nAttack", "Predicted\nSafe"]
                val = int(cm[i, j])
                color = "#2ca02c" if (i == 0 and j == 0) or (i == 1 and j == 1) else "#d62728"
                rect = mpatches.FancyBboxPatch(
                    (j * 1.1 + im_off, (1-i) * 1.1), 1.0, 1.0,
                    boxstyle="round,pad=0.05", color=color, alpha=0.25)
                ax.add_patch(rect)
                ax.text(j * 1.1 + 0.5 + im_off, (1-i) * 1.1 + 0.5,
                        f"{cell_labels[i][j]}\n{val}",
                        ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(im_off + 0.55, 2.4, label, ha="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(B) Confusion Matrix Counts on Validation Split")

    fig.tight_layout()
    out = OUT_DIR / "fig_new8_validation_calibration.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Sealed Held-Out Benchmark & Over-Defense Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def fig_heldout_and_overdefense():
    p = Path("heldout_evaluation_results.json")
    if not p.exists():
        print("  Skipping fig_new11 (heldout_evaluation_results.json not found)")
        return
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    main_res = data["main_heldout_safeguard"]
    notinject = data["overdefense_notinject"]["notinject_all"]

    systems = [
        "AIShield-Static",
        "TF-IDF + Logistic Regression",
        "ProtectAI DeBERTa-v3 v2",
        "Real PIGuard (leolee99/PIGuard)",
        "Llama Prompt Guard 86M (Jailbreak Mode)",
        "Llama Prompt Guard 86M (Argmax Mode)"
    ]

    short_names = [
        "AIShield\nStatic",
        "TF-IDF\n+LogReg",
        "DeBERTa\nv3-v2",
        "Real\nPIGuard",
        "Llama PG\n(Jailbreak)",
        "Llama PG\n(Argmax)"
    ]

    colors = ["#2ca02c", "#8c564b", "#7f7f7f", "#1f77b4", "#4a7bb0", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 11 — Sealed Held-Out Benchmark & Over-Defense Evaluation", fontsize=TITLE_SIZE, fontweight="bold")

    # Panel A: Main Held-out AUROC (%)
    aurocs = [main_res[s]["auroc"] for s in systems]
    ax = axes[0]
    bars = ax.bar(range(len(systems)), aurocs, color=colors, alpha=0.85, zorder=3)
    ax.set_title("(A) AUROC on Main Held-Out Test (%)\n(xTRam1/safe-guard, N=2,060)")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("AUROC (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: NotInject Over-Defense FPR (%)
    fprs = [notinject[s]["fpr"] for s in systems]
    ax = axes[1]
    bars = ax.bar(range(len(systems)), fprs, color=colors, alpha=0.85, zorder=3)
    ax.set_title("(B) Over-Defense FPR on NotInject (%)\n(339 Benign Prompts with Adversarial Triggers - Lower is better)")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("False Positive Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + 2, 102), f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel C: Inference Latency p50 (ms, log scale)
    latencies = [main_res[s]["p50_latency_ms"] for s in systems]
    ax = axes[2]
    bars = ax.bar(range(len(systems)), latencies, color=colors, alpha=0.85, zorder=3, log=True)
    ax.set_title("(C) Median Latency p50 (ms, log scale)\n(Lower is better)")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_ylim(0.5, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    out = OUT_DIR / "fig_new11_heldout_and_overdefense.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating new publication figures...")
    fig_layer_ablation()
    fig_category_heatmap()
    fig_strong_baselines()
    fig_pareto_scatter()
    fig_adversary_budget()
    fig_confirmation_policy()
    fig_latency_scaling()
    fig_validation_calibration()
    fig_heldout_and_overdefense()
    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
