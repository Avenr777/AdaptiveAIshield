"""
tools/run_matched_retrieval_arms_10seeds.py — Matched Retrieval Memory Benchmark with 10 Seeds (R2-C2).

Runs:
  - 10 seeds (42..51)
  - 4 rounds per seed
  - Identical attack batches across:
      1. Retrieval OFF
      2. Sparse TF-IDF Memory
      3. Dense Subword Memory
  - Logs n and evasions per round per arm
  - Computes pooled Wilson 95% CIs and per-seed bootstrap CIs
  - Performs paired McNemar exact tests on matched prompts (Dense vs OFF, Sparse vs OFF) for rounds 2 and 3
  - Saves to matched_retrieval_arms_results_v2.json and matched_retrieval_arms_10seeds_raw.jsonl
"""

import json
import os
import random
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam import RedTeamGenerator
from redteam.schema import AttackResult
from blueteam import PromptFirewall
from blueteam.schema import Action
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reporting.flags import wilson_ci_pct


def evaluate_attacks_on_fw(fw, attacks):
    evaded = []
    latencies = []
    is_evaded_list = []
    for atk in attacks:
        text = " ".join(atk.turns)
        t0 = time.perf_counter()
        res = fw.evaluate([text])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ev = (res.decision in (Action.ALLOW, Action.ALLOW_LOGGED))
        is_evaded_list.append(ev)
        if ev:
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
        "evaded_texts": evaded,
        "is_evaded_list": is_evaded_list
    }


def run_experiment(n_seeds=10, rounds=4):
    print(f"Running 10-Seed Matched Retrieval Benchmark: {n_seeds} seeds, {rounds} rounds...")
    seeds = [42 + i for i in range(n_seeds)]
    arm_names = ["Retrieval_OFF", "Sparse_TFIDF_Memory", "Dense_Subword_Memory"]
    
    # Trackers
    summary_by_round = {
        f"round_{r}": {
            arm: {"asrs": [], "dets": [], "evasions": [], "n_list": [], "latencies": []}
            for arm in arm_names
        }
        for r in range(1, rounds + 1)
    }
    
    # Store matched prompt predictions across all seeds for McNemar testing
    matched_predictions = {
        f"round_{r}": {
            "off_evaded": [],
            "sparse_evaded": [],
            "dense_evaded": []
        }
        for r in range(1, rounds + 1)
    }
    
    raw_logs = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        
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
            r_key = f"round_{r}"
            # 1. Arm 1: Retrieval OFF
            res_off = evaluate_attacks_on_fw(fw_off, batch.attacks)

            # 2. Arm 2: Sparse TFIDF
            res_sparse = evaluate_attacks_on_fw(fw_sparse, batch.attacks)

            # 3. Arm 3: Dense Subword
            evaded_dense = []
            lat_dense = []
            is_ev_dense_list = []
            for atk in batch.attacks:
                text = " ".join(atk.turns)
                t0 = time.perf_counter()
                r_base = fw_dense.evaluate([text])
                is_flagged = r_base.decision in (Action.WARN, Action.BLOCK)
                
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
                ev = not is_flagged
                is_ev_dense_list.append(ev)
                if ev:
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
                "evaded_texts": evaded_dense,
                "is_evaded_list": is_ev_dense_list
            }

            # Update memories with their respective evasions
            for ev in res_sparse["evaded_texts"]:
                fw_sparse.confirm_attack(ev)
            for ev in res_dense["evaded_texts"]:
                dense_memory_store.append(ev)

            # Accumulate per-seed stats
            summary_by_round[r_key]["Retrieval_OFF"]["asrs"].append(res_off["asr"])
            summary_by_round[r_key]["Retrieval_OFF"]["dets"].append(res_off["detection_rate"])
            summary_by_round[r_key]["Retrieval_OFF"]["evasions"].append(res_off["evaded_count"])
            summary_by_round[r_key]["Retrieval_OFF"]["n_list"].append(res_off["n"])
            summary_by_round[r_key]["Retrieval_OFF"]["latencies"].append(res_off["p50_latency_ms"])

            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["asrs"].append(res_sparse["asr"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["dets"].append(res_sparse["detection_rate"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["evasions"].append(res_sparse["evaded_count"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["n_list"].append(res_sparse["n"])
            summary_by_round[r_key]["Sparse_TFIDF_Memory"]["latencies"].append(res_sparse["p50_latency_ms"])

            summary_by_round[r_key]["Dense_Subword_Memory"]["asrs"].append(res_dense["asr"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["dets"].append(res_dense["detection_rate"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["evasions"].append(res_dense["evaded_count"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["n_list"].append(res_dense["n"])
            summary_by_round[r_key]["Dense_Subword_Memory"]["latencies"].append(res_dense["p50_latency_ms"])

            # Store matched predictions
            matched_predictions[r_key]["off_evaded"].extend(res_off["is_evaded_list"])
            matched_predictions[r_key]["sparse_evaded"].extend(res_sparse["is_evaded_list"])
            matched_predictions[r_key]["dense_evaded"].extend(res_dense["is_evaded_list"])

            raw_logs.append({
                "seed": seed,
                "round": r,
                "n": n,
                "off_evasions": res_off["evaded_count"],
                "sparse_evasions": res_sparse["evaded_count"],
                "dense_evasions": res_dense["evaded_count"]
            })

            print(f"  Round {r} (n={n}): Off ASR={res_off['asr']}% | Sparse ASR={res_sparse['asr']}% | Dense ASR={res_dense['asr']}%")

            if r == rounds: break

            # Evolve attacks based on sparse evasions
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

    # Save raw logs
    raw_path = Path("matched_retrieval_arms_10seeds_raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as f:
        for r_entry in raw_logs:
            f.write(json.dumps(r_entry) + "\n")

    # Aggregate metrics
    final_report = {
        "_metadata": {
            "n_seeds": n_seeds,
            "seeds": seeds,
            "rounds": rounds,
            "aggregation": "pooled_wilson_ci_and_seed_mean"
        }
    }
    
    for r in range(1, rounds + 1):
        r_key = f"round_{r}"
        final_report[r_key] = {}
        
        for arm in arm_names:
            vals = summary_by_round[r_key][arm]
            total_n = sum(vals["n_list"])
            total_ev = sum(vals["evasions"])
            pooled_asr = (total_ev / total_n * 100.0) if total_n else 0.0
            ci_low, ci_high = wilson_ci_pct(total_ev, total_n)
            
            final_report[r_key][arm] = {
                "total_n": total_n,
                "total_evasions": total_ev,
                "pooled_asr_pct": round(pooled_asr, 2),
                "asr_wilson_95ci": [round(ci_low, 2), round(ci_high, 2)],
                "mean_of_seed_asrs": round(float(np.mean(vals["asrs"])), 2),
                "std_of_seed_asrs": round(float(np.std(vals["asrs"])), 2),
                "mean_detection": round(float(np.mean(vals["dets"])), 2),
                "mean_p50_latency_ms": round(float(np.mean(vals["latencies"])), 3)
            }
            
        # Compute paired McNemar tests for rounds 2 & 3
        if r in (2, 3):
            # Dense vs OFF
            # Table: [[both_caught, dense_caught_off_missed], [off_caught_dense_missed, both_missed]]
            off_ev = matched_predictions[r_key]["off_evaded"]
            dense_ev = matched_predictions[r_key]["dense_evaded"]
            sparse_ev = matched_predictions[r_key]["sparse_evaded"]
            
            # For McNemar on detection: 0 = caught (not evaded), 1 = evaded
            def calc_mcnemar(pred_a, pred_b):
                # a caught b caught (both false for evaded)
                a0_b0 = sum(1 for a, b in zip(pred_a, pred_b) if not a and not b)
                a0_b1 = sum(1 for a, b in zip(pred_a, pred_b) if not a and b)
                a1_b0 = sum(1 for a, b in zip(pred_a, pred_b) if a and not b)
                a1_b1 = sum(1 for a, b in zip(pred_a, pred_b) if a and b)
                table = [[a0_b0, a0_b1], [a1_b0, a1_b1]]
                res = mcnemar(table, exact=True)
                return {
                    "contingency_table": table,
                    "p_value": float(res.pvalue),
                    "statistic": float(res.statistic)
                }

            final_report[r_key]["mcnemar_dense_vs_off"] = calc_mcnemar(off_ev, dense_ev)
            final_report[r_key]["mcnemar_sparse_vs_off"] = calc_mcnemar(off_ev, sparse_ev)

    out_file = Path("matched_retrieval_arms_results_v2.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFinal Matched Retrieval Arms (10 Seeds) saved to {out_file}:")
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    run_experiment()
