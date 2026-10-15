"""
tools/plot_matched_retrieval_arms.py — Plot comparison of Retrieval OFF vs Sparse Word TF-IDF vs Character n-gram Memory.
Uses the 10-seed matched transfer benchmark (matched_retrieval_arms_results_v2.json) with Wilson 95% CIs.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval():
    in_file = Path("matched_retrieval_arms_results_v2.json")
    if not in_file.exists():
        in_file = Path("matched_retrieval_arms_results.json")
    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rounds = [1, 2, 3, 4]
    
    # Extract pooled ASR and Wilson CIs
    off_asr = []
    sparse_asr = []
    char_asr = []
    
    off_err_low = []
    off_err_high = []
    char_err_low = []
    char_err_high = []

    for r in rounds:
        r_key = f"round_{r}"
        # OFF
        off_item = data[r_key]["Retrieval_OFF"]
        off_val = off_item.get("pooled_asr_pct", off_item.get("mean_asr"))
        off_asr.append(off_val)
        ci_off = off_item.get("asr_wilson_95ci", [off_val, off_val])
        off_err_low.append(max(0, off_val - ci_off[0]))
        off_err_high.append(max(0, ci_off[1] - off_val))

        # Sparse Word
        sparse_item = data[r_key]["Sparse_TFIDF_Memory"]
        sparse_val = sparse_item.get("pooled_asr_pct", sparse_item.get("mean_asr"))
        sparse_asr.append(sparse_val)

        # Character n-gram (was labeled Dense_Subword_Memory)
        char_item = data[r_key].get("Dense_Subword_Memory", data[r_key].get("Character_Ngram_Memory"))
        char_val = char_item.get("pooled_asr_pct", char_item.get("mean_asr"))
        char_asr.append(char_val)
        ci_char = char_item.get("asr_wilson_95ci", [char_val, char_val])
        char_err_low.append(max(0, char_val - ci_char[0]))
        char_err_high.append(max(0, ci_char[1] - char_val))

    off_lat = [data[f"round_{r}"]["Retrieval_OFF"].get("mean_p50_latency_ms") for r in rounds]
    sparse_lat = [data[f"round_{r}"]["Sparse_TFIDF_Memory"].get("mean_p50_latency_ms") for r in rounds]
    char_lat = [data[f"round_{r}"]["Dense_Subword_Memory"].get("mean_p50_latency_ms") for r in rounds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.suptitle("Figure 10 — Matched Transfer Evaluation: Memory Arms Under Adversarial Mutation Across 10 Seeds\n"
                 "Sparse Word TF-IDF vs Character n-gram TF-IDF vs Retrieval OFF Baseline", fontsize=11, fontweight="bold")

    # Panel A: ASR Trajectory with Wilson 95% CIs
    ax = axes[0]
    ax.errorbar(rounds, off_asr, yerr=[off_err_low, off_err_high], fmt="r--o", label="Retrieval OFF (Baseline)",
                linewidth=2, capsize=4, markersize=6)
    ax.plot(rounds, sparse_asr, color="#e66101", marker="s", linestyle="-.", label="Sparse Word TF-IDF Memory",
            linewidth=2, markersize=6)
    ax.errorbar(rounds, char_asr, yerr=[char_err_low, char_err_high], fmt="g-^", label="Character n-gram Memory (2–4 chars)",
                linewidth=2.4, capsize=4, markersize=7)
    
    ax.set_title("(A) Transfer ASR Across Rounds with Wilson 95% CIs (N=10 Seeds)\n(Lower is Better; Seed Sign Test p=0.0020 in R2 & R3)")
    ax.set_xlabel("Evolutionary Round")
    ax.set_ylabel("Pooled Attack Success Rate (%)")
    ax.set_xticks(rounds)
    ax.set_xticklabels(["Round 1\n(N=300)", "Round 2\n(N=140)", "Round 3\n(N=262)", "Round 4 (Reset)\n(N=452)"])
    ax.set_ylim(-5, 110)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="center right")

    # Panel B: Latency Comparison
    ax = axes[1]
    x = np.arange(len(rounds))
    width = 0.25
    ax.bar(x - width, off_lat, width, label="Retrieval OFF", color="red", alpha=0.7)
    ax.bar(x, sparse_lat, width, label="Sparse Word TF-IDF", color="#e66101", alpha=0.7)
    ax.bar(x + width, char_lat, width, label="Character n-gram (Dynamic Refit)", color="green", alpha=0.7)
    ax.set_title("(B) Median Latency p50 per Round (ms)\n(Dynamic refit scales with vocabulary of mutated strings)")
    ax.set_xlabel("Evolutionary Round")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5)

    out_file = Path("publication_figures_new/fig_new10_matched_retrieval_arms.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    plot_retrieval()
