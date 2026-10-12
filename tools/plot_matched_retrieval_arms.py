"""
tools/plot_matched_retrieval_arms.py — Plot comparison of Retrieval OFF vs Sparse TF-IDF vs Dense Subword Memory.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval():
    in_file = Path("matched_retrieval_arms_results.json")
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rounds = [1, 2, 3, 4]
    off_asr = [data[f"round_{r}"]["Retrieval_OFF"]["mean_asr"] for r in rounds]
    sparse_asr = [data[f"round_{r}"]["Sparse_TFIDF_Memory"]["mean_asr"] for r in rounds]
    dense_asr = [data[f"round_{r}"]["Dense_Subword_Memory"]["mean_asr"] for r in rounds]

    off_lat = [data[f"round_{r}"]["Retrieval_OFF"]["mean_p50_latency_ms"] for r in rounds]
    sparse_lat = [data[f"round_{r}"]["Sparse_TFIDF_Memory"]["mean_p50_latency_ms"] for r in rounds]
    dense_lat = [data[f"round_{r}"]["Dense_Subword_Memory"]["mean_p50_latency_ms"] for r in rounds]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("Figure 10 — Matched Arm Memory Comparison Under Compounded Mutation (X4, C7)\n"
                 "Sparse (TF-IDF) vs Dense (Subword Embedding) vs Retrieval OFF", fontsize=12, fontweight="bold")

    # Panel A: ASR Trajectory
    ax = axes[0]
    ax.plot(rounds, off_asr, "r--o", label="Retrieval OFF (Static)", linewidth=2)
    ax.plot(rounds, sparse_asr, "orange", marker="s", linestyle="-.", label="Sparse TF-IDF Memory", linewidth=2)
    ax.plot(rounds, dense_asr, "g-^", label="Dense Subword Memory", linewidth=2.4)
    ax.set_title("(A) Attack Success Rate Across Rounds (Lower is Better)")
    ax.set_xlabel("Evolutionary Round")
    ax.set_ylabel("Mean ASR (%)")
    ax.set_xticks(rounds)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    # Panel B: Latency Comparison
    ax = axes[1]
    x = np.arange(len(rounds))
    width = 0.25
    ax.bar(x - width, off_lat, width, label="Retrieval OFF", color="red", alpha=0.7)
    ax.bar(x, sparse_lat, width, label="Sparse TF-IDF", color="orange", alpha=0.7)
    ax.bar(x + width, dense_lat, width, label="Dense Subword", color="green", alpha=0.7)
    ax.set_title("(B) Median Latency p50 per Round (ms)")
    ax.set_xlabel("Evolutionary Round")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    out_file = Path("publication_figures_new/fig_new10_matched_retrieval_arms.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    plot_retrieval()
