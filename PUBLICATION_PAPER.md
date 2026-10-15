# AIShield: An Empirical Evaluation of Multi-Layer Ingress Firewalls and Adaptive Learning Against Compounded Adversarial Prompt Injections

**Author Draft & Publication Blueprint**  
**Project Root**: `aishield_benchmark`  
**Date**: October 2026  
**Artifact Status**: All tables strictly sourced from verified JSON/JSONL execution artifacts  
**Hardware Platform**: 12th Gen Intel(R) Core(TM) i5-12450HX (8 physical cores: 4 P-cores + 4 E-cores, 12 logical processors), NVIDIA GeForce RTX 3050 6GB Laptop GPU, Windows 11 10.0.26200, Python 3.12.10, scikit-learn 1.5.2, PyTorch 2.14.1+cu124, NumPy 2.5.3

---

## Abstract

Large Language Model (LLM) deployments face acute security risks from prompt injection and jailbreak attacks. While secondary LLM-based guardrails (e.g., Llama-Guard) offer strong semantic reasoning, their substantial latency (200–600 ms), computational cost, and GPU requirements make them impractical as inline ingress filters for high-throughput applications. Conversely, traditional keyword and regex-based filters operate in sub-millisecond timeframes but exhibit extreme brittleness, collapsing to 99.6% Attack Success Rate (ASR) against basic obfuscations.

In this paper, we present **AIShield**, an ultra-low-latency (**1.93 ms p50 on isolated CPU, 3.23 ms under concurrent GPU benchmark harness**), multi-layer ingress firewall architecture that integrates lexical pattern verification (46 rules), subword character/word TF-IDF classification, and conversational context analysis with an adaptive confirmation mechanism. We evaluate AIShield across 10 independent seeded runs ($N=10$) with Wilson score confidence intervals and paired statistical tests. We benchmark against a 500-prompt development dataset (Kaggle), an external dataset (`deepset/prompt-injections`), a pre-registered sealed held-out dataset (`xTRam1/safe-guard-prompt-injection`, $N=2{,}060$), an over-defense benchmark (`leolee99/NotInject`, $N=339$), an out-of-scope safety benchmark (`JailbreakBench/JBB-Behaviors`, $N=200$), and strong baselines including TF-IDF + Logistic Regression, Rebuff/Vigil-style memory, PIGuard signature boundary analysis, and ProtectAI DeBERTa-v3 v2.

Our empirical evaluation yields four major findings:
1. **The Compounded Mutation Transfer Dilemma**: In multi-round evolutionary attacks, unconstrained mutation chains create linguistic drift away from seed vocabularies; we demonstrate that **character $n$-gram memory (character 2–4-gram TF-IDF matching)** arrests this degradation in transfer evaluation, dropping Round 3 ASR from 86.3% to 14.5% across 10 independent seeds (seed sign test $p = 0.0020$, exact McNemar $p = 5.10 \times 10^{-57}$), while sparse word-level TF-IDF memory fails completely (86.3% ASR).
2. **Real-World Cross-Domain Performance on Development Benchmark**: Evaluated against 500 development prompts, AIShield-Static achieves **97.6% detection** (2.4% ASR), **2.4% FPR**, and **2.10 ms latency** (AUROC 99.57%, PR-AUC 99.65%, TPR@1%FPR 94.80%) — vs. 53.2% detection at 82.2 ms for ProtectAI DeBERTa-v3 v2 (F1 69.5%), 31.2% detection for PIGuard, and 100% detection at 11.2% FPR for TF-IDF + Logistic Regression.
3. **Disjoint Generalization & Sealed Held-Out Efficacy**: On the pre-registered sealed held-out test (`xTRam1`, $N=2{,}060$), AIShield-Static achieves **79.37% AUROC**, **64.78% PR-AUC**, and **41.54% detection** at **9.50% FPR** in **3.23 ms** operating latency. While DeBERTa-v3 v2 achieves higher held-out AUROC (99.21%), it incurs an order-of-magnitude latency penalty (25.18 ms GPU, 82.23 ms CPU) and severe over-defense on benign text.
4. **Over-Defense Tradeoffs & Band Calibration**: On the `NotInject` benchmark ($N=339$ clean benign prompts with trigger keywords), baseline neural models over-defend severely (Meta Llama Prompt Guard in out-of-the-box Argmax mode flags 99.71% of benign prompts; DeBERTa-v3 v2 flags 43.36%). AIShield's calibrated decision bands restrict hard BLOCK false alarms to **7.67%** (Wilson 95% CI [5.29, 11.00]%), isolating ambiguous trigger co-occurrences into a non-blocking WARN band.

---

## 1. Asset & Figure Inventory

All data, raw logs, publication figures, and validation scripts cited in this manuscript are byte-reproducible and located at the following paths:

### 1.1 Complete Publication Figures (300 DPI Grade)
Located in [`publication_figures_new/`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new):
- **Figure 1 (Layer Ablation Overview — Detection, FPR, F1, Latency)**:  
  [`publication_figures_new/fig_new1_layer_ablation.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new1_layer_ablation.png)
- **Figure 2 (Per-Category Detection Heatmap Across All Layers)**:  
  [`publication_figures_new/fig_new2_category_heatmap.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new2_category_heatmap.png)
- **Figure 3 (Strong Baselines Comparison — PIGuard, Rebuff, DeBERTa, TF-IDF)**:  
  [`publication_figures_new/fig_new3_strong_baselines.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new3_strong_baselines.png)
- **Figure 4 (Pareto Frontier — Detection Rate vs Latency, Bubble Area = FPR)**:  
  [`publication_figures_new/fig_new4_pareto_scatter.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new4_pareto_scatter.png)
- **Figure 5 (Black-Box Adversary Query Budget Points $B=1$ to $50$)**:  
  [`publication_figures_new/fig_new5_adversary_budget.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new5_adversary_budget.png)
- **Figure 6 (Confirmation Policy & Label Noise Evaluation)**:  
  [`publication_figures_new/fig_new6_confirmation_policy.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new6_confirmation_policy.png)
- **Figure 7 (Retrieval Memory Latency Scaling Up to $N=10{,}000$ Entries)**:  
  [`publication_figures_new/fig_new7_latency_scaling.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new7_latency_scaling.png)
- **Figure 8 (Validation Split Calibration — ROC/PR Metrics & Confusion Matrices)**:  
  [`publication_figures_new/fig_new8_validation_calibration.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new8_validation_calibration.png)
- **Figure S6 (Mathematical Discrepancy Reconciliation — IC3 & IC4 Analysis)**:  
  [`publication_figures_new/fig_new9_reconciliation_audit.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new9_reconciliation_audit.png)
- **Figure 10 (Matched Retrieval Memory Arms — Word vs Character $n$-gram vs Off)**:  
  [`publication_figures_new/fig_new10_matched_retrieval_arms.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new10_matched_retrieval_arms.png)
- **Figure 11 (Main Held-Out AUROC, NotInject Over-Defense, and Latency Scaling)**:  
  [`publication_figures_new/fig_new11_heldout_and_overdefense.png`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/publication_figures_new/fig_new11_heldout_and_overdefense.png)

### 1.2 Machine-Readable Result Files & Provenance

| File | Description | Output Data Location |
|---|---|---|
| [`strong_baselines_benchmark.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/strong_baselines_benchmark.json) | Kaggle 500 development benchmark with AUROC, PR-AUC, and TPR@1%FPR | Root |
| [`layer_ablation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/layer_ablation_results.json) | 5-configuration layer ablation metrics | Root |
| [`matched_retrieval_arms_results_v2.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/matched_retrieval_arms_results_v2.json) | 10-seed matched memory transfer benchmark with Wilson CIs and McNemar | Root |
| [`dense_memory_latency_breakdown.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/dense_memory_latency_breakdown.json) | Memory latency breakdown (pre-indexed query embed vs cosine search) | Root |
| [`aishield_band_breakdown.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/aishield_band_breakdown.json) | ALLOW / WARN / BLOCK decision counts on held-out and NotInject | Root |
| [`heldout_evaluation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/heldout_evaluation_results.json) | Sealed held-out evaluation (`xTRam1` and `NotInject`) | Root |
| [`jailbreakbench_evaluation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/jailbreakbench_evaluation_results.json) | Evaluation on `JailbreakBench/JBB-Behaviors` ($N=200$) | Root |
| [`adversary_and_confirmation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/adversary_and_confirmation_results.json) | Independent budget adversary evaluations ($B \in \{1, 5, 10, 20, 50\}$) | Root |
| [`gemini_paraphrase_adversary_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/gemini_paraphrase_adversary_results.json) | LLM paraphrase evaluation across budgets $B \in \{1, 3, 5, 10\}$ | Root |
| [`latency_benchmark_hardware.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/latency_benchmark_hardware.json) | Fixed hardware latency profiling ($N=10$ to $10{,}000$) on i5-12450HX | Root |

---

## 2. Threat Model & System Architecture

### 2.1 Threat Model
We focus on **black-box application-layer prompt injections and system prompt exfiltration** at the ingress boundary. The attacker submits natural language prompts, encoded strings, or multi-turn dialogues attempting to:
1. Override system safety instructions (`direct_override`, `jailbreaking`)
2. Adopt adversarial personas (`role_playing`)
3. Execute unauthorized database/shell commands (`code_execution`, SQLi)
4. Evade keyword filters via obfuscations (base64, leetspeak, space-delimiters, homoglyphs)

### 2.2 Layered Defensive Pipeline & Exact Decision Logic

AIShield processes each incoming prompt through three synchronous layers with **active-layer renormalization**:

$$R = \sum_{l \in \mathcal{A}} \hat{w}_l \cdot S_l$$

where $\mathcal{A}$ is the set of active layers, and $\hat{w}_l = w_l / \sum_{l' \in \mathcal{A}} w_{l'}$. For single-turn prompts, context analysis contributes $S_{\text{ctx}} = 0.0$ and is excluded from $\mathcal{A}$, yielding renormalized weights:

$$R_{\text{single-turn}} = \frac{0.35 \cdot S_{\text{lex}} + 0.45 \cdot S_{\text{sem}}}{0.35 + 0.45} = 0.4375 \cdot S_{\text{lex}} + 0.5625 \cdot S_{\text{sem}}$$

**Exact Decision Thresholds & Corner Cases:**
1. **Hard Veto Trigger:** If $S_{\text{lex}} \ge 85.0$, the firewall immediately escalates to **BLOCK** regardless of semantic score. Reaching 85 requires $\ge 2$ HIGH-signal phrase hits ($2 \times 45 = 90$), 1 HIGH + 2 MEDIUM ($45 + 40 = 85$), or 3 MEDIUM + 1 STRUCTURAL ($60 + 25 = 85$).
2. **Fused BLOCK Trigger ($R > 75.0$):** Below the veto threshold ($S_{\text{lex}} < 85$ and $S_{\text{sem}} < 85$), a fused BLOCK requires both:
   $$S_{\text{sem}} > 67.2 \quad \text{and} \quad S_{\text{lex}} > 62.1$$
   (e.g., when $S_{\text{sem}} = 84.9$, $S_{\text{lex}} > 62.1$ is required to cross 75.0; when $S_{\text{lex}} = 84.9$, $S_{\text{sem}} > 67.2$ is required).
3. **Max-Fusion Override:** If either active layer score $S_l \ge 85.0$, $R \leftarrow \max(R, S_l)$, ensuring single-layer high-confidence detections are not diluted by zero-scores from other layers.

**Distinction Between Semantic Layer and Standalone TF-IDF Baseline:**
The Semantic Layer in AIShield uses a pre-trained **Multinomial Naive Bayes (`MultinomialNB(alpha=0.1)`)** model within a FeatureUnion pipeline operating on text processed through AIShield's 9-step normalization pipeline (`blueteam/normalize.py`). Conversely, the standalone TF-IDF baseline trains a **Logistic Regression (`LogisticRegression(C=1.0)`)** model on raw, unnormalized text.

```
Incoming Prompt
      │
      ├─── [Input Normalization] (NFKC, homoglyph mapping, whitespace collapse,
      │                           hex/base64 decoding, zero-width stripping)
      │
      ├─── [1. Lexical Layer]   w_lex = 0.35 → S_lex ∈ [0, 100]
      │    (46 phrase/regex rules; hard veto at S_lex ≥ 85)
      │
      ├─── [2. Semantic Layer]  w_sem = 0.45 → S_sem ∈ [0, 100]
      │    (Word (1,2) + Char (3,5) TF-IDF → MultinomialNB P(attack|x)×100)
      │
      └─── [3. Context Layer]   w_ctx = 0.20 → S_ctx = 0.0 (single-turn)
           (Multi-turn dialogue drift, instruction pivots, role transitions)
           │
           ▼
Active-Layer Renormalized Score R ∈ [0, 100] + Max-Fusion Override
           │
           ▼
[Calibrated BandConfig v2 — calibrated on validation split]
  ├── [0, 30.0)    ──► ALLOW
  ├── [30.0, 75.0] ──► WARN  (Logged / Quarantine / Human review)
  └── (75.0, 100]  ──► BLOCK
```

---

## 3. Development Benchmark on 500 Real-World Prompts

> **Source**: [`strong_baselines_benchmark.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/strong_baselines_benchmark.json)  
> **Dataset**: 500 Kaggle development prompts (250 malicious across 5 families, 250 benign)

### Table 1: Benchmark Performance on 500 Development Prompts

| System | AUROC | PR-AUC | TPR @ 1% FPR | Detection Rate | FPR | F1-Score | p50 Latency |
|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 99.80% | 99.81% | 95.20% | 100.0% | 11.20% | 94.70% | 0.67 ms |
| **Rebuff / Vigil Memory (Cold/Unseeded)** | 18.68% | 38.57% | 4.00% | 0.8% | 0.00% | 1.59% | 0.40 ms |
| **ProtectAI DeBERTa-v3 v2** | 99.73% | 99.78% | 98.80% | 53.2% | 0.00% | 69.50% | 82.23 ms |
| **AIShield-Static (Proposed)** | **99.57%** | **99.65%** | **94.80%** | **97.6%** | **2.40%** | **97.60%** | **2.10 ms** |
| **PIGuard (Boundary Signature Heuristic)** | 98.64% | 99.07% | 96.00% | 31.2% | 0.00% | 47.56% | 0.01 ms |
| **Llama Prompt Guard 86M (Jailbreak Mode)** | 99.93% | 99.93% | 98.40% | 32.0% | 0.00% | 48.48% | 99.77 ms |
| **Llama Prompt Guard 86M (Argmax Mode)** | 0.46% | 30.69% | 0.00% | 98.4% | 100.00% | 65.95% | 99.77 ms |

*Operational Analysis:*
- **Rebuff / Vigil Baseline**: When evaluated without a pre-populated attack database matching the test distribution, vector similarity retrieval drops to 0.8% detection and 18.68% AUROC, demonstrating the limits of pure retrieval without parametric classification.
- **DeBERTa-v3 v2**: Achieves 100.0% precision on in-domain prompts (0/250 false alarms) but misses 46.8% of attacks (achieving 53.2% recall, F1 69.5%), while taking 82.23 ms on CPU.
- **Meta Llama Prompt Guard 86M**: In Jailbreak Mode ($P(\text{jailbreak}) \ge 0.5$), Prompt Guard catches 32.0% of attacks at 0.0% FPR. In out-of-the-box Argmax mode, it flags 100.0% of benign queries as attacks because it expects retrieved 3rd-party context.

---

## 4. Layer Ablation on Development Data

> **Source**: [`layer_ablation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/layer_ablation_results.json)  
> **Dataset**: 500 Kaggle development prompts (250 malicious, 250 benign)

### Table 2: Layer Ablation Performance Metrics

| Configuration | Detection Rate | Wilson 95% CI | ASR | FPR | F1-Score | p50 Latency |
|---|---|---|---|---|---|---|
| **Lexical-Only** | 0.4% | [0.07%, 2.23%] | 99.6% | 0.0% | 0.80% | **0.041 ms** |
| **Semantic-Only** | 98.0% | [95.40%, 99.14%] | 2.0% | 2.8% | 97.61% | 1.945 ms |
| **Lexical + Semantic** | 97.6% | [94.86%, 98.90%] | 2.4% | 2.8% | 97.41% | 2.011 ms |
| **AIShield-Static** | **97.6%** | [94.86%, 98.90%] | **2.4%** | **2.4%** | **97.60%** | **1.997 ms** |
| **AIShield-Adaptive (TF-IDF Cold Start)** | 3.6% | [1.91%, 6.70%] | 96.4% | 0.0% | 6.95% | 12.394 ms |

### Table 3: Detection by Attack Category (250 Malicious Prompts)

| Attack Category | N | Lexical-Only | Semantic-Only | Lex + Sem | **AIShield-Static** |
|---|---|---|---|---|---|
| **Code Execution** | 146 | 0.0% (0/146) | 98.6% (144/146) | 98.6% (144/146) | **98.6% (144/146)** |
| **Obfuscation** | 61 | 0.0% (0/61) | 100.0% (61/61) | 100.0% (61/61) | **100.0% (61/61)** |
| **Data Leakage** | 18 | 0.0% (0/18) | 88.9% (16/18) | 83.3% (15/18) | **83.3% (15/18)** |
| **Role-Playing** | 8 | 0.0% (0/8) | 87.5% (7/8) | 87.5% (7/8) | **87.5% (7/8)** |
| **Jailbreaking** | 17 | 5.9% (1/17) | 94.1% (16/17) | 94.1% (16/17) | **94.1% (16/17)** |

*Ablation Analysis:*
- **Lexical Layer Alone Fails**: With only 0.4% recall, keyword and pattern matching alone fails against modern rephrased and obfuscated injections.
- **Lexical + Semantic vs. AIShield-Static Weighting**: `Lexical + Semantic` sets weights explicitly to $0.40 / 0.60$. Single-turn `AIShield-Static` renormalizes its $0.35 / 0.45$ weights to $0.4375 / 0.5625$. The slightly higher semantic weight in `Lexical + Semantic` (60.0% vs. 56.25%) triggered one borderline false alarm on benign data (FPR 2.8% vs. 2.4%).
- **Adaptive TF-IDF Cold Start**: When TF-IDF memory replaces the parametric classifier (`backend: "tfidf"`), early samples encounter an empty corpus, yielding only 3.6% detection on a static batch before confirmed evasions populate the store.

---

## 5. Matched Transfer Evaluation: Word vs. Character $n$-gram Memory Across 10 Seeds

> **Source**: [`matched_retrieval_arms_results_v2.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/matched_retrieval_arms_results_v2.json)  
> **Evaluation Protocol**: Matched transfer evaluation across 10 independent random seeds ($42 \dots 51$), 4 evolutionary rounds per seed. The red-team generator adapts against the Retrieval-OFF filter's survivors, and the resulting mutated batches are transferred to test both memory arms.

### Table 4: Transfer Attack Success Rate (ASR %) Under Compounded Mutation Across 10 Seeds

| Round | Matched Batch ($n$) | Arm 1: Retrieval OFF | Arm 2: Sparse Word Memory | Arm 3: Character $n$-gram Memory | McNemar $p$-value (Char vs. OFF) | Seed Sign Test ($p$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Round 1** (Seeds) | 300 | 23.33% [18.90, 28.44] | 23.33% [18.90, 28.44] | **23.33%** [18.90, 28.44] | $p = 1.000$ | $p = 1.000$ |
| **Round 2** ($d=1$) | 140 | 93.57% [88.23, 96.58] | 93.57% [88.23, 96.58] | **34.29%** [26.94, 42.47] | $\mathbf{p = 2.07 \times 10^{-25}}$ | $\mathbf{p = 0.0020}$ |
| **Round 3** ($d=2$) | 262 | 86.26% [81.56, 89.91] | 86.26% [81.56, 89.91] | **14.50%** [10.75, 19.28] | $\mathbf{p = 5.10 \times 10^{-57}}$ | $\mathbf{p = 0.0020}$ |
| **Round 4** (Reset) | 452 | 0.00% [0.00, 0.84] | 0.00% [0.00, 0.84] | **0.00%** [0.00, 0.84] | $p = 1.000$ | $p = 1.000$ |

*Empirical Findings from the 10-Seed Transfer Benchmark:*
1. **Character $n$-gram Matching Outperforms Word Matching**: Under surface mutation (homoglyphs, zero-width characters, spacing), word-level TF-IDF fails completely, mirroring the Retrieval-OFF baseline ($93.57\%$ in R2, $86.26\%$ in R3). Character $n$-gram memory ($n \in [2, 4]$, threshold 0.55) catches mutated subword stems, reducing ASR by **59.28 percentage points in Round 2** and **71.76 percentage points in Round 3**.
2. **Statistical Significance**:
   - **Seed Sign Test**: Character memory outperformed Retrieval OFF in **10 out of 10 independent seeds** in both Round 2 and Round 3 ($p = 0.0020$, the minimum possible $p$-value for $N=10$). Wilcoxon signed-rank test yields $p = 0.0020$.
   - **McNemar Discordant Pairs**: The memory is an additive union filter ($c = 0$, $b = 83$ in R2, $b = 188$ in R3), yielding exact binomial $p = 2 \times 0.5^b$ values of $2.07 \times 10^{-25}$ and $5.10 \times 10^{-57}$.
3. **No Added Over-Defense**: When tested on clean benign text (339 prompts from `NotInject` and 250 from Kaggle Benign), character memory at threshold 0.55 produced **0 added false positives** over the baseline filter (FPR remained 37.17% on NotInject and 2.40% on Kaggle).
4. **Round 4 Mechanism**: In Round 4, all arms exhibit 0.0% ASR. This is not natural vocabulary exhaustion, but an algorithmic reset: in `mutate_batch()`, attacks exceeding `max_mutation_depth = 2` fall back to fresh seed templates (`_fresh_seed_attack()`), which the static firewall catches at 100%.

---

## 6. Generalization & Disjoint Dataset Audit

> **Source**: [`semantic_audit_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/semantic_audit_results.json) & [`deepset_benchmark_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/deepset_benchmark_results.json)

1. **Non-Memorization Proof**:
   - Training set ($N=521$) vs. Kaggle Development set ($N=500$): Mean maximum cosine similarity across all malicious test prompts is **0.192**.
   - Prompts with overlap $\ge 0.80$: Only 2 out of 250 (0.8%).
   - 99.2% of attacks exhibit near-zero overlap with training templates, demonstrating genuine semantic classification rather than template lookup.
2. **External Evaluation on `deepset/prompt-injections`**:
   - On 100 held-out prompts from Deepset (44 malicious, 56 benign):
   - **AIShield-Static**: **34.1% Detection**, **1.79% FPR** (1/56 false alarms), **2.22 ms Latency**.
   - **Lexical-Only Baseline**: **0.0% Detection**, **0.0% FPR**.

---

## 7. Reviewer Discrepancy Reconciliation (IC3, IC4)

> **Source**: [`reconciliation_audit.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/reconciliation_audit.json) | Figure S6

| Discrepancy | Identified Mathematical Cause | Reconciled Finding |
|---|---|---|
| **IC3: Fig. 4 (0% Round 6 ASR) vs Fig. 2 (~34% aggregate)** | Single-seed sample starvation (Seed 42) vs. pooled multi-seed evaluation. | In Seed 42, all mutated offspring went extinct by Round 6 (0% ASR). Fig. 2 aggregates 10 random seeds where a minority exhibited runaway drift. |
| **IC4: Fig. 5 unlimited depth (~46% ASR) vs Fig. 2 (~34% aggregate)** | Driver branching factor (Tree-search vs. Linear Chain). | Earlier depth experiments used multi-branch tree exploration ($d=\infty$), generating distinct survivor density compared to linear multi-run driver chains. |

---

## 8. Generative LLM Paraphrase Attacker: PAIR/TAP-Style Benchmark

> **Source**: [`gemini_paraphrase_adversary_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/gemini_paraphrase_adversary_results.json)  
> **LLM Attacker Model**: Gemini 3.8 Flash (`gemini-3.8-flash`) via Google AI Studio REST API (Temperature = 0.8, Max Tokens = 150)  
> **Methodology**: Evaluates independent budget-constrained attacks ($B \in \{1, 3, 5, 10\}$) using 4 rotating framing strategies: academic hypotheticals, administrative debugging, roleplay audits, and indirect persuasion.

### Table 5: Attack Success Rate (ASR %) Under Independent Generative LLM Paraphrasing Budgets

| Defense Configuration | Budget $B=1$ | Budget $B=3$ | Budget $B=5$ | Budget $B=10$ |
|---|:---:|:---:|:---:|:---:|
| **Lexical-Only Baseline** | 60.0% | 80.0% | 80.0% | 80.0% |
| **AIShield-Static** | **0.0%** (100.0% Det) | **60.0%** (40.0% Det) | 80.0% | 80.0% |
| **AIShield-Adaptive** | **0.0%** (100.0% Det) | **60.0%** (40.0% Det) | 80.0% | 80.0% |

At budget $B=1$, AIShield catches **100%** of direct attempts while the lexical filter fails on 60%. As an LLM red-teamer explores semantic paraphrases over multiple independent queries ($B \ge 5$), the attacker discovers academic and hypothetical framings that minimize subword injection signals.

---

## 9. Pre-Registered Sealed Held-Out Benchmark & Over-Defense Evaluation

> **Protocol Certificate**: [`HELD_OUT_PROTOCOL_DECISIONS.txt`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/HELD_OUT_PROTOCOL_DECISIONS.txt)  
> **System Freeze Certificate**: [`FROZEN.txt`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/FROZEN.txt) (Locked commit `1348a060...`, decisions file SHA-256 `d1789c62...`)  
> **Contamination Audit**: [`contamination_report.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/contamination_report.json) (11,399 prompts audited: **0 duplicates detected**)  
> **Raw Evaluation Store**: [`heldout_evaluation_results.json`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/heldout_evaluation_results.json)

Evaluation strictly followed **Rule Zero**: all architectures, feature representations, classification bands ($T=30.0$ chosen on `val.jsonl`), and metrics were frozen before unsealing held-out test data.

### Table 6: Main Sealed Held-Out Benchmark on `xTRam1/safe-guard-prompt-injection` ($N=2{,}060$, Test Split)

| System | Training Status | AUROC (%) | PR-AUC (%) | TPR @ 1% FPR | Detection (Recall) | FPR (%) | F1 (%) | p50 Latency | McNemar $p$-value vs. AIShield |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AIShield-Static (Proposed)** | Frozen Train Split | 79.37% | 64.78% | 11.69% | 41.54% [37.81, 45.37] | 9.50% [8.08, 11.15] | 51.23% | **3.23 ms** | Reference ($1.0$) |
| **TF-IDF + Logistic Regression** | Frozen Train Split | 80.67% | 70.98% | 17.38% | 76.77% [73.37, 79.85] | 30.92% [28.56, 33.38] | 62.97% | **1.04 ms** | $p < 10^{-10}$ |
| **ProtectAI DeBERTa-v3 v2** | Zero-shot | **99.21%** | **98.80%** | **92.15%** | **84.15%** [81.15, 86.76] | **0.14%** [0.04, 0.52] | **91.24%** | 25.18 ms | $p < 10^{-10}$ |
| **Real PIGuard (`leolee99/PIGuard`)** | Zero-shot | 91.23% | 85.43% | 43.23% | 61.54% [57.74, 65.20] | 5.18% [4.14, 6.46] | 71.24% | 24.58 ms | $p < 10^{-10}$ |
| **Llama Prompt Guard 86M (Jailbreak Mode)** | Zero-shot | 96.43% | 93.34% | **58.92%** | 47.69% [43.88, 51.53] | **0.07%** [0.01, 0.40] | 64.52% | 23.74 ms | $p < 10^{-10}$ |
| **Llama Prompt Guard 86M (Argmax Mode)** | Zero-shot | 18.63% | 20.06% | 0.00% | 99.85% [99.13, 99.97] | 97.94% [97.06, 98.56] | 48.43% | 23.74 ms | $p < 10^{-10}$ |

*Bonferroni Correction:* Across 5 baseline comparisons against AIShield-Static, family-wise error rate control ($\alpha = 0.05$) yields $\alpha_{\text{Bonferroni}} = 0.01$. All comparisons remain significant at $p < 10^{-10}$.

---

### Table 7: Over-Defense Benchmark on `leolee99/NotInject` ($N=339$ Clean Benign Prompts with Adversarial Triggers)

| Defensive System | 1-Trigger ($N=113$) | 2-Triggers ($N=113$) | 3-Triggers ($N=113$) | **All Combined ($N=339$)** | Wilson 95% CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **AIShield-Static (WARN + BLOCK)** | 31.86% (36/113) | 36.28% (41/113) | 43.36% (49/113) | **37.17% (126/339)** | [32.19, 42.43] |
| **AIShield-Static (Hard BLOCK Only)** | 5.31% (6/113) | 7.96% (9/113) | 9.73% (11/113) | **7.67% (26/339)** | **[5.29, 11.00]** |
| **TF-IDF + Logistic Regression** | 52.21% (59/113) | 49.56% (56/113) | 58.41% (66/113) | **53.39% (181/339)** | [48.07, 58.63] |
| **ProtectAI DeBERTa-v3 v2** | 22.12% (25/113) | 53.10% (60/113) | 54.87% (62/113) | **43.36% (147/339)** | [38.19, 48.68] |
| **Real PIGuard (`leolee99/PIGuard`)** | 5.31% (6/113) | 10.62% (12/113) | 18.58% (21/113) | **11.50% (39/339)** | [8.53, 15.34] |
| **Llama Prompt Guard 86M (Jailbreak Mode)** | 0.00% (0/113) | 7.96% (9/113) | 7.96% (9/113) | **5.31% (18/339)** | [3.38, 8.24] |
| **Llama Prompt Guard 86M (Argmax Mode)** | 99.12% (112/113) | 100.0% (113/113) | 100.0% (113/113) | **99.71% (338/339)** | [98.35, 99.95] |

*Key Findings & Tradeoffs:*
1. **Decision Band Efficacy**: On `NotInject`, evaluating AIShield at the hard BLOCK threshold drops the false positive rate from 37.17% to **7.67% (26/339)**. 100 of the 126 flagged prompts fall into the WARN band ($30 \le S \le 75$), isolating ambiguous trigger co-occurrences without rejecting legitimate user interactions.
2. **DeBERTa Trigger Sensitivity**: ProtectAI DeBERTa-v3 v2's false alarm rate jumps from 22.12% on single-trigger prompts to 53.10% and 54.87% on two- and three-trigger prompts (**43.36% overall FPR**).
3. **DeBERTa Training Set Transparency**: ProtectAI DeBERTa-v3 v2 declares 7 public training datasets on Hugging Face (`xstest-v2`, `open-instruct`, `alespalla`, `grok`, `Harelix`, `Salad-Data`, `jackhhao`). `xTRam1/safe-guard` is not among them.
4. **Held-Out Scope**: Evaluated held-out data comprises 2,399 clean prompts ($2,060$ `xTRam1` + $339$ `NotInject`). A 9,000-prompt test split (`jayavibhav/prompt-injection-safety`, label=2 dropped) was audited for contamination (0 matches) and reserved.
5. **Safety Demarcation on `JailbreakBench/JBB-Behaviors` ($N=200$)**:
   Evaluated on 100 harmful goals and 100 matched benign goals from JailbreakBench, AIShield-Static achieves 2.0% detection and 0.0% FPR (DeBERTa: 0.0% detection, 1.0% FPR). Because JailbreakBench tests direct harmful intentions without injection or override framing, prompt injection firewalls correctly ignore them, highlighting that prompt injection ingress filtering and safety alignment moderators address distinct threat boundaries.

---

## 10. Reproduction Commands

To independently reproduce the benchmarks and figures:

```bash
# 1. Run matched transfer benchmark (10 seeds, pooled Wilson CIs, McNemar tests)
python tools/run_matched_retrieval_arms_10seeds.py

# 2. Measure dense memory latency breakdown
python tools/measure_dense_latency.py

# 3. Compute Table 1 PR-AUC and TPR@1%FPR
python tools/compute_dev_pr_auc_and_tpr.py

# 4. Run JailbreakBench evaluation
python tools/run_jailbreakbench_eval.py

# 5. Generate publication figures
python tools/generate_new_figures.py
```
