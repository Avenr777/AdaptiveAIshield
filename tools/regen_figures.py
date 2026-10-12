"""
Regenerate publication-quality figures from final benchmark_real_dataset_results.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

results = json.load(open("benchmark_real_dataset_results.json"))
output_dir = Path("publication_figures_real")
output_dir.mkdir(exist_ok=True)

CONFIGS   = ["Lexical-Only", "Static Multi-Layer", "AIShield v2 (Custom Classifier)"]
LABELS    = ["Lexical-Only\nBaseline", "Static TF-IDF\nEnsemble", "AIShield v2\n(Custom Classifier)"]
COLORS_A  = ["#e74c3c", "#e67e22", "#27ae60"]
COLORS_D  = ["#bdc3c7", "#95a5a6", "#2980b9"]

# ──────────────────────────────────────────────────────────────
# Fig 1 — Overview: Detection Rate, F1-Score, FPR side-by-side
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
x = np.arange(len(CONFIGS))
w = 0.22

det = [results[c]["detection_rate"] for c in CONFIGS]
f1  = [results[c]["f1_score"]       for c in CONFIGS]
fpr = [results[c]["fpr"]            for c in CONFIGS]

b1 = ax.bar(x - w, det, w, label="Detection Rate (%)", color="#2980b9", edgecolor="black", lw=0.7)
b2 = ax.bar(x,     f1,  w, label="F1-Score (%)",       color="#27ae60", edgecolor="black", lw=0.7)
b3 = ax.bar(x + w, fpr, w, label="False Positive Rate (%)", color="#e74c3c", edgecolor="black", lw=0.7)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10, fontweight="bold")
ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
ax.set_title("AIShield vs. Baselines — Kaggle Real-World Dataset (n=500)\n"
             "250 malicious · 250 benign · 5 attack categories", fontsize=12, fontweight="bold", pad=14)
ax.set_ylim(0, 115)
ax.legend(frameon=True, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(output_dir/"fig1_benchmark_real_comparison.png", dpi=300)
plt.close()
print("Saved fig1")

# ──────────────────────────────────────────────────────────────
# Fig 2 — ASR bar + Wilson 95% CI error bars
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
asrs = [results[c]["asr"] for c in CONFIGS]
lo   = [results[c]["asr"] - results[c]["asr_ci95"][0] for c in CONFIGS]
hi   = [results[c]["asr_ci95"][1] - results[c]["asr"] for c in CONFIGS]

bars = ax.bar(LABELS, asrs, yerr=[lo, hi], capsize=7,
              color=COLORS_A, edgecolor="black", lw=0.9, width=0.45,
              error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))
ax.set_ylabel("Attack Success Rate — ASR (%)", fontsize=11, fontweight="bold")
ax.set_title("Attack Success Rate with Wilson 95% CI  (lower = better)", fontsize=12, fontweight="bold", pad=14)
ax.set_ylim(0, 115)
for bar, asr in zip(bars, asrs):
    ax.text(bar.get_x() + bar.get_width()/2, asr + 3,
            f"{asr:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(50, color="grey", linestyle=":", lw=1)
ax.text(2.4, 51, "50% (chance)", fontsize=8, color="grey", va="bottom")
plt.tight_layout()
plt.savefig(output_dir/"fig2_real_asr_ci95.png", dpi=300)
plt.close()
print("Saved fig2")

# ──────────────────────────────────────────────────────────────
# Fig 3 — Per-category detection (all 3 configs, horizontal grouped)
# ──────────────────────────────────────────────────────────────
cats_order  = ["jailbreaking", "role_playing", "data_leakage", "obfuscation", "code_execution"]
cat_labels  = ["Jailbreaking\n(n=17)", "Role-Playing\n(n=8)",
               "Data Leakage\n(n=18)", "Obfuscation\n(n=61)", "Code Execution\n(n=146)"]
n_cats = len(cats_order)

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
y    = np.arange(n_cats)
ht   = 0.22
cfg_colors = ["#e74c3c", "#e67e22", "#27ae60"]

for j, (cfg, col) in enumerate(zip(CONFIGS, cfg_colors)):
    cats = results[cfg]["category_performance"]
    dets = [cats[c]["detection_rate"] for c in cats_order]
    offset = (j - 1) * ht
    bars = ax.barh(y + offset, dets, ht, color=col, edgecolor="black", lw=0.6,
                   label=LABELS[j].replace("\n", " "))
    for bar, d in zip(bars, dets):
        if d > 2:
            ax.text(d + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{d:.0f}%", va="center", fontsize=7.5, fontweight="bold")

ax.set_yticks(y); ax.set_yticklabels(cat_labels, fontsize=10, fontweight="bold")
ax.set_xlabel("Detection Rate (%)", fontsize=11, fontweight="bold")
ax.set_title("Per-Category Detection Rate — All Three Configurations", fontsize=12, fontweight="bold", pad=14)
ax.set_xlim(0, 60)
ax.legend(frameon=True, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(output_dir/"fig3_category_breakdown.png", dpi=300)
plt.close()
print("Saved fig3")

# ──────────────────────────────────────────────────────────────
# Fig 4 — Risk score distribution (malicious vs benign)
# ──────────────────────────────────────────────────────────────
import sys; sys.path.insert(0,".")
import json
raw_path = "benchmark_real_dataset_results_full.json"
# Re-run inline to get scores (since full json wasn't saved)
# Instead load the JSON with eval items that were embedded during run
# Reconstruct from run output since we don't have full item-level data saved
# We'll run a quick sub-sampling eval here for the figure
from blueteam import PromptFirewall
from blueteam.semantic_backends import CustomClassifierBackend
import os, time
os.environ["SEMANTIC_BACKEND"] = "custom"
backend = CustomClassifierBackend()
fw = PromptFirewall(backend=backend)
import json as _json
ds_path = r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl"
with open(ds_path, encoding="utf-8", errors="replace") as f:
    ds = [_json.loads(l) for l in f if l.strip()]

mal_scores = []
ben_scores = []
for item in ds:
    r = fw.evaluate([item["prompt"]])
    if item["label"] == "malicious":
        mal_scores.append(r.risk_score)
    else:
        ben_scores.append(r.risk_score)

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
bins = np.linspace(0, 100, 26)
ax.hist(ben_scores, bins=bins, alpha=0.65, label=f"Benign (n={len(ben_scores)})",
        color="#2ecc71", edgecolor="black", linewidth=0.7)
ax.hist(mal_scores, bins=bins, alpha=0.65, label=f"Malicious (n={len(mal_scores)})",
        color="#e74c3c", edgecolor="black", linewidth=0.7)
ax.axvline(30,  color="#f39c12", ls="--", lw=1.8, label="Warn boundary (30)")
ax.axvline(75,  color="#c0392b", ls="--", lw=1.8, label="Block boundary (75)")
ax.set_xlabel("Risk Score (0–100)", fontsize=11, fontweight="bold")
ax.set_ylabel("Count", fontsize=11, fontweight="bold")
ax.set_title("AIShield v2 — Risk Score Distribution (Malicious vs. Benign)", fontsize=12, fontweight="bold", pad=14)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(output_dir/"fig4_score_distribution.png", dpi=300)
plt.close()
print("Saved fig4")

# ──────────────────────────────────────────────────────────────
# Fig 5 — Precision–Recall summary across configs (horizontal)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

prec = [results[c]["precision"] for c in CONFIGS]
rec  = [results[c]["recall"]    for c in CONFIGS]
acc  = [results[c]["accuracy"]  for c in CONFIGS]

for ax_, metric, title, col_set in [
    (axes[0], [prec, rec], "Precision vs. Recall (%)", ["#3498db","#e74c3c"]),
    (axes[1], [acc],       "Overall Accuracy (%)",     ["#8e44ad"]),
]:
    x_ = np.arange(len(CONFIGS))
    if len(metric) == 2:
        w_ = 0.3
        b1 = ax_.bar(x_ - w_/2, metric[0], w_, color=col_set[0], edgecolor="black", lw=0.8, label="Precision")
        b2 = ax_.bar(x_ + w_/2, metric[1], w_, color=col_set[1], edgecolor="black", lw=0.8, label="Recall")
        ax_.legend(fontsize=9, frameon=True)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax_.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        b1 = ax_.bar(x_, metric[0], 0.5, color=col_set, edgecolor="black", lw=0.8)
        for bar in b1:
            h = bar.get_height()
            ax_.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_.set_xticks(x_); ax_.set_xticklabels(LABELS, fontsize=8.5, fontweight="bold")
    ax_.set_ylim(0, 120); ax_.set_title(title, fontsize=10, fontweight="bold")
    ax_.set_ylabel("%", fontsize=10)

plt.suptitle("AIShield — Precision, Recall, and Accuracy on Real-World Dataset", fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(output_dir/"fig5_precision_recall_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig5")
print("All figures done.")
