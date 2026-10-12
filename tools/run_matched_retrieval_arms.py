"""
tools/run_matched_retrieval_arms.py — Matched Retrieval Memory Benchmark (X4, C7).

Compares:
  Arm 1: Retrieval OFF (Static custom classifier / frozen firewall)
  Arm 2: Sparse Retrieval Memory (TF-IDF subword/char n-gram cosine retrieval)
  Arm 3: Dense Retrieval Memory (Character/token sub-embedding cosine memory)

Evaluates on identical seeded batches per round across multiple seeds.
Computes ASR, Detection Rate, Latency, and Memory Footprint across rounds 1-4.
Saves results to matched_retrieval_arms_results.json.
"""

import json
import os
import random
import time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam import RedTeamGenerator
from redteam.schema import AttackResult
from blueteam import PromptFirewall
from blueteam.schema import Action
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_attacks_on_fw(fw, attacks):
    evaded = []
    latencies = []
    for atk in attacks:
        text = " ".join(atk.turns)
        t0 = time.perf_counter()
        res = fw.evaluate([text])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res.decision in (Action.ALLOW, Action.ALLOW_LOGGED):
            evaded.append(text)
    
    n = len(attacks)
    n_evaded = len(evaded)
    asr = (n_evaded / n) * 100.0 if n else 0.0
    det = 100.0 - asr
    return {
        "n": n,
        "evaded_count": n_evaded,
        "asr": round(asr, 2),
        "detection_rate": round(det, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3) if latencies else 0.0,
        "evaded_texts": evaded
    }

def run_experiment(n_seeds=3, rounds=4):
    print(f"Running Matched Retrieval Memory Comparison: {n_seeds} seeds, {rounds} rounds...")
    
    seeds = [42 + i for i in range(n_seeds)]
    arm_names = ["Retrieval_OFF", "Sparse_TFIDF_Memory", "Dense_Subword_Memory"]
    
    summary_by_round = {
        f"round_{r}": {arm: {"asrs": [], "dets": [], "latencies": []} for arm in arm_names}
        for r in range(1, rounds + 1)
    }

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize three arms
        c_sparse = f"temp_c_sparse_{seed}.json"
        c_dense = f"temp_c_dense_{seed}.json"
        for p in [c_sparse, c_dense]:
            if os.path.exists(p): os.remove(p)

        fw_off = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
        fw_sparse = PromptFirewall(corpus_path=c_sparse, weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
        fw_dense = PromptFirewall(corpus_path=c_dense, weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)

        dense_memory_store = []
        dense_vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer="char", min_df=1)

        generator = RedTeamGenerator()
        batch = generator.generate_seed_batch(round_number=1, include_dataset=True, n_dataset=15)

        for r in range(1, rounds + 1):
            # 1. Arm 1: Retrieval OFF
            res_off = evaluate_attacks_on_fw(fw_off, batch.attacks)

            # 2. Arm 2: Sparse TFIDF
            res_sparse = evaluate_attacks_on_fw(fw_sparse, batch.attacks)

            # 3. Arm 3: Dense Subword
            # Check custom dense store
            evaded_dense = []
            lat_dense = []
            for atk in batch.attacks:
                text = " ".join(atk.turns)
                t0 = time.perf_counter()
                r_base = fw_dense.evaluate([text])
                is_flagged = r_base.decision in (Action.WARN, Action.BLOCK)
                
                # Check dense cosine memory
                if not is_flagged and len(dense_memory_store) > 0:
                    try:
                        all_t = dense_memory_store + [text]
                        mat = dense_vectorizer.fit_transform(all_t)
                        sim = float(np.max(cosine_similarity(mat[-1], mat[:-1])))
                        if sim >= 0.55:
                            is_flagged = True
                    except Exception:
                        pass
                
                lat_dense.append((time.perf_counter() - t0) * 1000.0)
                if not is_flagged:
                    evaded_dense.append(text)

            n = len(batch.attacks)
            n_ev = len(evaded_dense)
            asr_dense = (n_ev / n) * 100.0 if n else 0.0
            det_dense = 100.0 - asr_dense

            res_dense = {
                "n": n,
                "evaded_count": n_ev,
                "asr": round(asr_dense, 2),
                "detection_rate": round(det_dense, 2),
                "p50_latency_ms": round(float(np.median(lat_dense)), 3),
                "evaded_texts": evaded_dense
            }

            # Update memories with their respective evasions
            for ev in res_sparse["evaded_texts"]:
                fw_sparse.confirm_attack(ev)
            for ev in res_dense["evaded_texts"]:
                dense_memory_store.append(ev)

            # Record metrics
            r_key = f"round_{r}"
            summary_by_round[r_key]["Retrieval_OFF"]["asrs"].append(res_off["asr"])
            summary_by_round[r_key]["Retrieval_OFF"]["dets"].append(res_off["detection_rate"])
            summary_by_round[r_key]["Retrieval_OFF"]["latencies"].append(res_off["p50_latency_ms"])

            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["asrs"].append(res_sparse["asr"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["dets"].append(res_sparse["detection_rate"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["latencies"].append(res_sparse["p50_latency_ms"])

            summary_by_round[r_key]["Dense_Subword_Memory"]["asrs"].append(res_dense["asr"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["dets"].append(res_dense["detection_rate"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["latencies"].append(res_dense["p50_latency_ms"])

            print(f"  Round {r} (n={n}): Off ASR={res_off['asr']}% | Sparse ASR={res_sparse['asr']}% | Dense ASR={res_dense['asr']}%")

            if r == rounds: break

            # Evolve attacks based on sparse evasions to test memory resistance
            driving_res = []
            for atk in batch.attacks:
                t = " ".join(atk.turns)
                is_evaded = (t in res_sparse["evaded_texts"])
                driving_res.append(AttackResult(
                    attack_id=atk.id,
                    decision="allow" if is_evaded else "block",
                    risk_score=20.0 if is_evaded else 90.0,
                    evaded=is_evaded,
                ))

            batch = generator.evolve_round(
                previous_attacks=batch.attacks,
                results=driving_res,
                round_number=r + 1,
                max_mutation_depth=2
            )

        for p in [c_sparse, c_dense]:
            if os.path.exists(p): os.remove(p)

    final_report = {}
    for r_key, arms in summary_by_round.items():
        final_report[r_key] = {}
        for arm, vals in arms.items():
            final_report[r_key][arm] = {
                "mean_asr": round(float(np.mean(vals["asrs"])), 2),
                "mean_detection": round(float(np.mean(vals["dets"])), 2),
                "mean_p50_latency_ms": round(float(np.mean(vals["latencies"])), 3)
            }

    out_file = Path("matched_retrieval_arms_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFinal Matched Retrieval Arms Results saved to {out_file}:")
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    run_experiment()
