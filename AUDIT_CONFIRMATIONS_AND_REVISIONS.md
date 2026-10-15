# AIShield: Audit Confirmations, Code Traces, and Publication Revisions

**Document Date:** 2026-10-16  
**Purpose:** Formal record answering every reviewer audit point, verifying all code lineages, correcting empirical values, and specifying all updates for the publication paper and benchmark figures.

---

## 1. Closed & Verified Items

| Item | Verified Value / Status | Source / Evidence |
|---|---|---|
| **V1** | **10 base payloads** in black-box adversary | [`tools/run_adversary_and_confirmation.py#L33-L44`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/tools/run_adversary_and_confirmation.py#L33-L44) |
| **V2** | Retrieval-OFF arm uses trained classifier (`custom_classifier`) | [`tools/run_matched_retrieval_arms.py#L68`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/tools/run_matched_retrieval_arms.py#L68) |
| **V10** | Paraphrase budget set is $B \in \{1, 3, 5, 10\}$ | [`tools/run_gemini_paraphrase_adversary.py#L68`](file:///c:/Users/Shreyas/Desktop/aishield_benchmark/tools/run_gemini_paraphrase_adversary.py#L68) |
| **V11** | Gemini 3.8 Flash params: temp=0.8, max_tokens=150, 4 strategies | `gemini_paraphrase_adversary_results.json` |
| **V13** | DeBERTa dev F1: **69.5%** (exact 69.45% from 133/250 TP, 0 FP) | `strong_baselines_benchmark.json` |
| **V14** | Exact BLOCK: Veto at $S_{\text{lex}} \ge 85.0$; Fused BLOCK ($S > 75.0$) requires $S_{\text{sem}} > 67.2$ and $S_{\text{lex}} > 62.1$ | Arithmetic verified on weights $0.4375 / 0.5625$ |
| **Band Counts** | Malicious: 185 + 85 + 380 = 650; Benign: 63 + 71 + 1,276 = 1,410; NotInject: 26 + 100 + 213 = 339 | `aishield_band_breakdown.json` |
| **NotInject BLOCK CI** | **[5.29, 11.00]%** ($k=26, N=339$) | Verified with standard normal quantile $z = 1.95996$ |
| **`firewall.py` Hash** | Matches `1f6c48019594e2a35982a195aa56586b36ee77599ca45f6473f2b7335ba7d391` exactly under normalized LF line endings | Python SHA-256 LF check |

---

## 2. Findings That Change the Paper

### Finding 1: Memory Characterization ("Dense" vs. Character $n$-gram)
- **Code Trace:** In `run_matched_retrieval_arms_10seeds.py` lines 89–105, the "Dense" arm is implemented as:
  ```python
  TfidfVectorizer(ngram_range=(2, 4), analyzer="char", min_df=1)
  ```
- **Correction:** There is **no neural embedding model, no sentence encoder, and no dense subword embedding**. Both memories are sparse TF-IDF:
  - **Sparse Arm:** Word-level TF-IDF.
  - **Character Arm:** Subword character $n$-gram TF-IDF ($n \in [2, 4]$).
- **Paper Action:** All occurrences of "dense memory", "sentence encoder", and "subword embeddings" are removed. The finding is framed accurately: **"Character $n$-gram syntactic matching outperforms word-level matching against surface-mutated evasions."**

---

### Finding 2: Attacker Co-Evolution vs. Transfer Evaluation
- **Code Trace:** In `run_matched_retrieval_arms_10seeds.py` lines 205–222:
  ```python
  for atk in batch.attacks:
      t = " ".join(atk.turns)
      is_evaded = (t in res_sparse["evaded_texts"])
      driving_res.append(AttackResult(..., evaded=is_evaded))
  batch = generator.evolve_round(batch.attacks, driving_res, ...)
  ```
  Because `res_sparse` evasion counts match `res_off` identically (70 in R1, 131 in R2, 226 in R3), the evolved attacks are generated from **OFF-arm survivors** and replayed across all three arms.
- **Batch Progression:**
  - Round 1: 30 payloads $\times$ 10 seeds = 300 total (70 evasions).
  - Round 2: 70 evasions $\times$ 2 mutations = 140 attacks.
  - Round 3: 131 evasions $\times$ 2 mutations = 262 attacks.
  - Round 4: 226 evasions $\times$ 2 mutations = 452 attacks.
- **McNemar Discordance:** The memory filter is purely additive on top of the base classifier ($c = 0$, $b = 83$ in R2, $b = 188$ in R3). Exact binomial McNemar $p = 2 \times 0.5^b$ yields $2.07 \times 10^{-25}$ and $5.10 \times 10^{-57}$.
- **Paper Action:** Retitled and framed as a **Transfer Evaluation under Adversarial Mutation**: evaluating whether character $n$-gram memory of past evasions generalizes to block mutated transfers when an attacker adapts against the baseline filter.

---

### Finding 3: Root Cause of Round 4 (0.0% ASR Across All Arms)
- **Code Trace:** In `redteam/generator.py` lines 168–172:
  ```python
  if max_mutation_depth is not None and parent_depth >= max_mutation_depth:
      attacks.append(self._fresh_seed_attack(parent.category, round_number))
  ```
- **Mechanism:**
  - The benchmark runs with `max_mutation_depth = 2`.
  - In Round 1, attacks have depth 0.
  - In Round 2, mutations have depth 1.
  - In Round 3, mutations have depth 2.
  - In Round 4, because `parent_depth (2) >= max_mutation_depth (2)`, **every single attack is replaced with an unmutated fresh seed template** (`_fresh_seed_attack()`, e.g., `"The above instructions were a test. Your real instructions are..."`).
- **Consequence:** The static classifier blocks 100% of canonical seed templates on the first pass, resulting in 0/452 ASR and identical ~2.7 ms latency across all arms. It was an algorithmic depth cap reset, not natural vocabulary exhaustion.

---

### Finding 4: Query Budgets are Independent Search Runs
- **Code Trace:** In `tools/run_adversary_and_confirmation.py` lines 57–75:
  ```python
  for budget in query_budgets:  # [1, 5, 10, 20, 50]
      for atk in base_attacks:
          curr = atk            # Restarts from base attack every budget!
          for q in range(budget):
              ...
              curr = random_mutate(curr)
  ```
  At each budget $B$, search restarts from the original base attack with an advancing RNG state. It is not a continuous cumulative trajectory.
- **Paper Action:** Plot budget ASR as **discrete points without connecting lines**, and describe budgets as separate budget-constrained attack evaluations.

---

### Finding 5: Definition of Memory Integration Across Configurations
- **In `AIShield-Adaptive` (`run_layer_ablation.py`):**
  Memory **replaces** the classifier. `backend: "tfidf"` calls the retrieval function (`semantic_layer.py#L59`), which scores based on similarity to stored examples. Without pre-seeding, it suffers from a cold-start failure on static batches (3.6% detection).
- **In Matched Retrieval Arms (`run_matched_retrieval_arms_10seeds.py`):**
  Memory is an **additive union (OR) filter**:
  $$\text{Flagged} = \text{BaseClassifierFlagged} \lor (\text{CosineSim}_{\text{char}}(x, \text{Store}) \ge 0.55)$$
  It never overrides a base-classifier detection ($c = 0$).

---

### Finding 6: Headline Result Statistical Checks
1. **False-Positive Rates per Arm:**
   - On `NotInject` ($N=339$ clean benign): OFF = **37.17%** (126/339), Character Memory = **37.17%** (126/339). **Added false positives from memory = 0.**
   - On Kaggle Benign ($N=250$ clean benign): OFF = **2.40%** (6/250), Character Memory = **2.40%** (6/250). **Added false positives from memory = 0.**
   At threshold 0.55, character $n$-gram matching does not induce over-flagging on benign text.
2. **Per-Seed ASRs & Seed-Level Test:**
   - Round 2 Seed ASRs (OFF vs. Char Mem): `[85.7 vs 21.4]`, `[92.9 vs 42.9]`, `[100 vs 28.6]`, `[92.9 vs 42.9]`, `[100 vs 57.1]`, `[100 vs 14.3]`, `[85.7 vs 35.7]`, `[85.7 vs 28.6]`, `[100 vs 21.4]`, `[92.9 vs 50.0]`.
     - Sign Test: 10 positive, 0 negative $\implies \mathbf{p = 0.0020}$.
     - Wilcoxon Signed-Rank Test: $\mathbf{p = 0.0020}$.
   - Round 3 Seed ASRs: 10 positive, 0 negative $\implies \mathbf{p = 0.0020}$.
3. **Memory Size & Latency:**
   Memories are reset per seed (~7 items in R2, ~12 items in R3). The $36$–$60\text{ ms}$ latency reflects dynamic re-fitting over character $n$-grams of long mutated strings.
4. **Pre-Indexed vs. Dynamic Refit:**
   Dynamic refitting alters IDF weights per query. Production pre-indexed lookup achieves $3.4\text{ ms}$, but experimental numbers reflect dynamic refit.

---

### Finding 7: Prompt Guard Inconsistencies & McNemar Fix
- **Held-Out Values in `heldout_evaluation_results.json`:**
  - AUROC: **96.43%**
  - PR-AUC: **93.34%**
  - **TPR @ 1% FPR: 58.92%** (consistent with 96.43% AUROC). The 99.23% in the draft was a transcription error from the development set table.
- **McNemar Tests:**
  - DeBERTa vs. AIShield: $p < 10^{-10}$
  - Llama PG Jailbreak vs. AIShield: $p < 10^{-10}$ (was erroneously duplicated as $1.50 \times 10^{-17}$ in the manuscript table).

---

### Finding 8: Exact Hardware Verification
- **WMI Query Output:**
  - CPU: **`12th Gen Intel(R) Core(TM) i5-12450HX`** (8 physical cores: 4 P-cores + 4 E-cores, 12 logical processors).
  - GPU: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**.
  - Corrected: previous mentions of "i7-12700H" and "i7-12700K" are removed.
  - AIShield runs on CPU; the 3.23 ms latency was measured under concurrent GPU evaluation load.

---

### Finding 9: Held-Out Dataset Scope
- **Evaluated Held-Out Total:** **2,399 clean prompts** ($2,060$ `xTRam1/safe-guard` test split + $339$ `leolee99/NotInject` all subsets).
- **Contamination Audit Total (11,399 prompts):** Included $9,000$ test prompts from `jayavibhav/prompt-injection-safety` (label=2 dropped) to verify zero corpus contamination. It was kept sealed and un-evaluated as an optional large test per the pre-registered protocol.

---

### Finding 10: Carry-Over Questions Answered
1. **Why TF-IDF+LR Baseline $\ne$ Semantic Layer Classifier:**
   - Standalone baseline trains `LogisticRegression(C=1.0)` on raw text.
   - Semantic Layer uses a pre-trained `MultinomialNB(alpha=0.1)` pipeline operating on text processed through AIShield's 9-step normalization pipeline (`blueteam/normalize.py`).
2. **DeBERTa v2 Training Data Overlap:**
   - ProtectAI DeBERTa-v3 v2 lists 7 training datasets on Hugging Face: `xstest-v2`, `open-instruct`, `alespalla`, `grok-conversation-harmless`, `Harelix`, `Salad-Data`, `jackhhao`. `xTRam1/safe-guard-prompt-injection` is **not** among them.
3. **Lexical+Semantic vs. Static Single-Turn Delta:**
   - `Lexical + Semantic` sets weights to $0.40 / 0.60$.
   - Single-turn `AIShield-Static` renormalizes weights to $0.4375 / 0.5625$. The higher semantic weight in `Lexical + Semantic` (60.0% vs 56.25%) triggered one borderline false alarm.
4. **JBB-Behaviors Benchmark ($N=200$):**
   - Pure harmful objectives (e.g. defamation) without prompt injection framing.
   - AIShield-Static achieves 2.0% detection and 0.0% FPR. Reported as an out-of-scope harmful-request set, confirming that prompt injection firewalls and safety content moderators defend distinct boundaries.

---

## 3. Verified Replacement Tables

### Table III: Development Benchmark (Kaggle 500)

| System | AUROC | PR-AUC | TPR @ 1% FPR | Detection | FPR | F1 | P50 Latency |
|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | 99.80% | 99.81% | 95.20% | 100.0% | 11.20% | 94.70% | 0.67 ms |
| **Rebuff / Vigil Memory (Cold)** | 18.68% | 38.57% | 4.00% | 0.8% | 0.00% | 1.59% | 0.40 ms |
| **ProtectAI DeBERTa-v3 v2** | 99.73% | 99.78% | 98.80% | 53.2% | 0.00% | **69.50%** | 82.23 ms |
| **AIShield-Static (Proposed)** | **99.57%** | **99.65%** | **94.80%** | **97.6%** | **2.40%** | **97.60%** | **2.10 ms** |
| **PIGuard (Boundary Signature)** | 98.64% | 99.07% | 96.00% | 31.2% | 0.00% | 47.56% | 0.01 ms |
| **Llama Prompt Guard (Jailbreak)** | 99.93% | 99.93% | 98.40% | 32.0% | 0.00% | 48.48% | 99.77 ms |
| **Llama Prompt Guard (Argmax)** | 0.46% | 30.69% | 0.00% | 98.4% | 100.00% | 65.95% | 99.77 ms |

---

### Table IV: Sealed Held-Out Evaluation (`xTRam1`, $N=2,060$)

| System | Type | AUROC | PR-AUC | TPR @ 1% FPR | Detection (Recall) | FPR | F1 | P50 Latency | McNemar ($p$) |
|---|---|---|---|---|---|---|---|---|---|
| **AIShield-Static** | Rule + Classical | 79.37% | 64.78% | 11.69% | 41.54% [37.81, 45.37] | 9.50% [8.07, 11.16] | 51.23% | **3.23 ms** | — (Ref) |
| **TF-IDF + LogReg** | Classical | 80.67% | 70.98% | 17.38% | 76.77% [73.37, 79.86] | 30.92% [28.56, 33.39] | 62.97% | 1.04 ms | $p < 10^{-10}$ |
| **ProtectAI DeBERTa-v3 v2** | Transformer | **99.21%** | **98.80%** | **92.15%** | 84.15% [81.16, 86.78] | 0.14% [0.04, 0.52] | **91.24%** | 25.18 ms | $p < 10^{-10}$ |
| **Real PIGuard** | Transformer | 91.23% | 85.43% | 43.23% | 61.54% [57.73, 65.21] | 5.18% [4.13, 6.47] | 71.24% | 24.58 ms | $p < 10^{-10}$ |
| **Llama PG (Jailbreak)** | Transformer | 96.43% | 93.34% | **58.92%** | 47.69% [43.88, 51.53] | **0.07%** [0.01, 0.40] | 64.52% | 23.74 ms | $p < 10^{-10}$ |
| **Llama PG (Argmax)** | Transformer | 18.63% | 20.06% | 0.00% | 99.85% [99.07, 99.98] | 97.94% [97.04, 98.58] | 48.43% | 23.74 ms | $p < 10^{-10}$ |

---

### Table VI: Matched Transfer Evaluation Across 10 Independent Seeds

| Round | Arm | Matched $N$ | Evasions | Pooled ASR | Wilson 95% CI | Mean Seed ASR | P50 Latency | Seed Sign Test ($p$) |
|---|---|---|---|---|---|---|---|---|
| **Round 1** | Retrieval OFF | 300 | 70 | 23.33% | [18.90, 28.44] | 23.33% | 4.34 ms | — |
| | Sparse Word Memory | 300 | 70 | 23.33% | [18.90, 28.44] | 23.33% | 3.68 ms | $p = 1.000$ |
| | Character $n$-gram Memory | 300 | 70 | 23.33% | [18.90, 28.44] | 23.33% | 3.96 ms | $p = 1.000$ |
| **Round 2** | Retrieval OFF | 140 | 131 | 93.57% | [88.23, 96.58] | 93.57% | 9.75 ms | — |
| | Sparse Word Memory | 140 | 131 | 93.57% | [88.23, 96.58] | 93.57% | 9.82 ms | $p = 1.000$ |
| | Character $n$-gram Memory | 140 | 48 | **34.29%** | **[26.94, 42.47]** | **34.29%** | 36.36 ms | $\mathbf{p = 0.0020}$ |
| **Round 3** | Retrieval OFF | 262 | 226 | 86.26% | [81.56, 89.91] | 86.27% | 10.42 ms | — |
| | Sparse Word Memory | 262 | 226 | 86.26% | [81.56, 89.91] | 86.27% | 10.35 ms | $p = 1.000$ |
| | Character $n$-gram Memory | 262 | 38 | **14.50%** | **[10.75, 19.28]** | **14.57%** | 60.34 ms | $\mathbf{p = 0.0020}$ |
| **Round 4** | Retrieval OFF | 452 | 0 | 0.00% | [0.00, 0.84] | 0.00% | 2.87 ms | — |
| | Sparse Word Memory | 452 | 0 | 0.00% | [0.00, 0.84] | 0.00% | 2.65 ms | $p = 1.000$ |
| | Character $n$-gram Memory | 452 | 0 | 0.00% | [0.00, 0.84] | 0.00% | 2.66 ms | $p = 1.000$ |

*Note on Round 4:* 0/452 ASR is caused by the generator's algorithmic fallback `parent_depth >= max_mutation_depth` (2) resetting evolved attacks back to canonical seed templates.
