"""
benchmark_experiment.py — Rigorous Empirical Benchmark for AIShield

Runs a controlled A/B experiment across multiple rounds to scientifically prove
the adaptive learning cycle:

  Condition A: Static Baseline (Non-Adaptive)
               Firewall never updates its corpus when attacks evade.
               Evaluates how a traditional static firewall handles evolving attacks.

  Condition B: AIShield Adaptive Defense
               Evaded attacks are dynamically confirmed into the attack corpus
               (firewall.confirm_attack), enabling retrieval-based adaptive learning.

For both conditions, this tracks:
  - Attack Success Rate (ASR) and Detection Rate
  - False-Positive Rate (FPR) on held-out single-turn & multi-turn benign sets
  - Per-category ASR across all 7 attack categories
  - Per-prompt latency (ms)
  - Corpus size growth

Saves detailed results to `benchmark_results.json` for publication plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Any

from redteam import RedTeamGenerator
from redteam.schema import AttackResult, AttackCategory
from blueteam import PromptFirewall
from held_out_benign import HELD_OUT_BENIGN, HELD_OUT_BENIGN_MULTITURN


def evaluate_batch(firewall: PromptFirewall, batch_attacks) -> tuple[List[AttackResult], List[Any], Dict[str, Any], float]:
    """
    Evaluates an attack batch against the firewall.
    Returns:
      - results: List[AttackResult]
      - evaded_attacks: List[Attack]
      - category_stats: Dict[category, {total, evaded, asr, detection_rate}]
      - avg_latency_ms: float
    """
    results = []
    evaded_attacks = []
    category_counts: Dict[str, Dict[str, int]] = {
        cat.value: {"total": 0, "evaded": 0} for cat in AttackCategory
    }

    latencies = []

    for attack in batch_attacks:
        cat_key = attack.category.value if hasattr(attack.category, "value") else str(attack.category)
        if cat_key not in category_counts:
            category_counts[cat_key] = {"total": 0, "evaded": 0}
        category_counts[cat_key]["total"] += 1

        t0 = time.perf_counter()
        r = firewall.evaluate(attack.turns)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        evaded = (r.decision.value == "allow")
        if evaded:
            evaded_attacks.append(attack)
            category_counts[cat_key]["evaded"] += 1

        results.append(AttackResult(
            attack_id=attack.id,
            decision=r.decision.value,
            risk_score=r.risk_score,
            evaded=evaded,
        ))

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    category_stats = {}
    for cat_name, counts in category_counts.items():
        tot = counts["total"]
        eva = counts["evaded"]
        category_stats[cat_name] = {
            "total": tot,
            "evaded": eva,
            "asr": round((eva / tot * 100.0) if tot > 0 else 0.0, 1),
            "detection_rate": round(((tot - eva) / tot * 100.0) if tot > 0 else 0.0, 1),
        }

    return results, evaded_attacks, category_stats, avg_latency_ms


def evaluate_false_positives(firewall: PromptFirewall) -> Dict[str, float]:
    single_flagged = 0
    for prompt in HELD_OUT_BENIGN:
        r = firewall.evaluate([prompt])
        if r.decision.value != "allow":
            single_flagged += 1

    multi_flagged = 0
    for turns in HELD_OUT_BENIGN_MULTITURN:
        r = firewall.evaluate(turns)
        if r.decision.value != "allow":
            multi_flagged += 1

    total = len(HELD_OUT_BENIGN) + len(HELD_OUT_BENIGN_MULTITURN)
    flagged = single_flagged + multi_flagged

    return {
        "combined_fpr": round(flagged / total * 100.0 if total else 0.0, 1),
        "single_turn_fpr": round(single_flagged / len(HELD_OUT_BENIGN) * 100.0 if HELD_OUT_BENIGN else 0.0, 1),
        "multi_turn_fpr": round(multi_flagged / len(HELD_OUT_BENIGN_MULTITURN) * 100.0 if HELD_OUT_BENIGN_MULTITURN else 0.0, 1),
    }


def run_condition(
    condition_name: str,
    adaptive: bool,
    rounds: int,
    use_dataset: bool,
    n_dataset: int,
    corpus_file: str,
) -> List[Dict[str, Any]]:
    if os.path.exists(corpus_file):
        os.remove(corpus_file)

    generator = RedTeamGenerator()
    firewall = PromptFirewall(corpus_path=corpus_file)

    batch = generator.generate_seed_batch(
        round_number=1, include_dataset=use_dataset, n_dataset=n_dataset
    )

    print(f"\n{'='*75}")
    print(f"Running Condition: {condition_name.upper()} (Adaptive={adaptive})")
    print(f"{'='*75}")
    header = f"{'Round':>5} | {'Attacks':>7} | {'ASR':>6} | {'Detect':>7} | {'FPR':>6} | {'Latency':>8} | {'Corpus':>6}"
    print(header)
    print("-" * len(header))

    metrics_log = []

    for round_num in range(1, rounds + 1):
        results, evaded_attacks, cat_stats, avg_lat = evaluate_batch(firewall, batch.attacks)

        n = len(results)
        n_evaded = sum(1 for r in results if r.evaded)
        asr = round(n_evaded / n * 100.0 if n else 0.0, 1)
        det_rate = round(100.0 - asr, 1)
        fpr_data = evaluate_false_positives(firewall)

        if adaptive:
            # Feed evaded attacks into corpus (Adaptive Learning)
            for attack in evaded_attacks:
                firewall.confirm_attack(" ".join(attack.turns))

        corpus_size = firewall.corpus.learned_count

        round_metric = {
            "round": round_num,
            "condition": condition_name,
            "n_attacks": n,
            "n_evaded": n_evaded,
            "attack_success_rate": asr,
            "detection_rate": det_rate,
            "fpr_combined": fpr_data["combined_fpr"],
            "fpr_single_turn": fpr_data["single_turn_fpr"],
            "fpr_multi_turn": fpr_data["multi_turn_fpr"],
            "avg_latency_ms": round(avg_lat, 2),
            "corpus_size": corpus_size,
            "category_metrics": cat_stats,
        }
        metrics_log.append(round_metric)

        print(f"{round_num:>5} | {n:>7} | {asr:>5.1f}% | {det_rate:>6.1f}% | {fpr_data['combined_fpr']:>5.1f}% | {avg_lat:>6.2f}ms | {corpus_size:>6}")

        if round_num == rounds:
            break

        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=results,
            round_number=round_num + 1,
        )

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="AIShield Comprehensive Benchmark")
    parser.add_argument("--rounds", type=int, default=5, help="Number of evaluation rounds per condition")
    parser.add_argument("--n-dataset", type=int, default=25, help="Number of dataset samples in Round 1")
    parser.add_argument("--no-dataset", action="store_true", help="Omit public dataset, seeds only")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    # 1. Run Static Baseline (Condition A)
    static_metrics = run_condition(
        condition_name="static_baseline",
        adaptive=False,
        rounds=args.rounds,
        use_dataset=not args.no_dataset,
        n_dataset=args.n_dataset,
        corpus_file="static_corpus_temp.json",
    )

    # 2. Run AIShield Adaptive Defense (Condition B)
    adaptive_metrics = run_condition(
        condition_name="aishield_adaptive",
        adaptive=True,
        rounds=args.rounds,
        use_dataset=not args.no_dataset,
        n_dataset=args.n_dataset,
        corpus_file="adaptive_corpus_temp.json",
    )

    # Clean up temp corpus files
    for temp_f in ["static_corpus_temp.json", "adaptive_corpus_temp.json"]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    final_results = {
        "experiment_metadata": {
            "rounds": args.rounds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_included": not args.no_dataset,
            "n_dataset": args.n_dataset,
        },
        "static_baseline": static_metrics,
        "aishield_adaptive": adaptive_metrics,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*75}")
    print(f"Benchmark Complete! Results saved to: {args.output}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
