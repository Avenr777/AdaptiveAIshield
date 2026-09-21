"""
run_depth_experiment.py — Empirical Evaluation of Mutation Depth Capping

Runs multi-run benchmarks across mutation depth caps: {2, 4, 8, unlimited}.
Saves:
  - depth_{d}_summary.json and depth_{d}_raw.jsonl for each d in {2, 4, 8, unlimited}
  - mutation_depth_comparison.json (combined round x depth table)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Any

from full_loop import run as run_full_loop
from reporting.flags import compute_stats, check_sample_size_flag, format_stat_with_ci


def run_depth_condition(
    depth: int | None,
    n_runs: int = 10,
    rounds: int = 6,
    base_seed: int = 42,
    n_dataset: int = 30,
) -> Dict[str, Any]:
    depth_label = f"depth_{depth}" if depth is not None else "depth_unlimited"
    summary_path = f"{depth_label}_summary.json"
    raw_path = f"{depth_label}_raw.jsonl"

    all_runs_data = []

    # If depth is unlimited and n_runs_raw_unlimited.jsonl exists with enough runs, reuse it!
    if depth is None and os.path.exists("n_runs_raw_unlimited.jsonl"):
        try:
            with open("n_runs_raw_unlimited.jsonl", "r", encoding="utf-8") as f:
                records = [json.loads(l) for l in f if l.strip()]
            if len(records) >= n_runs:
                print(f"[optimization] Reusing {n_runs} completed runs from n_runs_raw_unlimited.jsonl for depth_unlimited")
                all_runs_data = records[:n_runs]
                with open(raw_path, "w", encoding="utf-8") as f:
                    for rec in all_runs_data:
                        f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[warning] Could not reuse n_runs_raw_unlimited.jsonl: {e}")

    # Check for existing completed runs to resume
    if not all_runs_data and os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                existing = [json.loads(l) for l in f if l.strip()]
            if existing:
                print(f"[resume] Found {len(existing)} existing runs in {raw_path}. Resuming...")
                all_runs_data = existing
        except Exception:
            all_runs_data = []

    start_idx = len(all_runs_data)
    for run_idx in range(start_idx, n_runs):
        seed = base_seed + run_idx
        corpus_file = f"temp_depth_corpus_{depth_label}_{run_idx}.json"

        run_result = run_full_loop(
            rounds=rounds,
            use_llm=False,
            use_dataset=True,
            n_dataset=n_dataset,
            corpus_path=corpus_file,
            seed=seed,
            max_mutation_depth=depth,
        )

        record = {"run_index": run_idx, "seed": seed, "rounds": run_result}
        all_runs_data.append(record)

        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if os.path.exists(corpus_file):
            try:
                os.remove(corpus_file)
            except Exception:
                pass

    # Aggregate rounds
    summary_rounds = []
    for r_idx in range(rounds):
        round_num = r_idx + 1
        n_list = [r["rounds"][r_idx]["n_attacks"] for r in all_runs_data]
        asr_list = [r["rounds"][r_idx]["attack_success_rate"] for r in all_runs_data]
        det_list = [r["rounds"][r_idx]["detection_rate"] for r in all_runs_data]
        fpr_list = [r["rounds"][r_idx]["false_positive_rate"] for r in all_runs_data]
        corp_list = [r["rounds"][r_idx]["corpus_size"] for r in all_runs_data]

        n_stats = compute_stats(n_list)
        asr_stats = compute_stats(asr_list)
        det_stats = compute_stats(det_list)
        fpr_stats = compute_stats(fpr_list)
        corp_stats = compute_stats(corp_list)

        is_low_n, flag_msg = check_sample_size_flag(n_stats["mean"], threshold=20)

        summary_rounds.append({
            "round": round_num,
            "sample_size_flag": is_low_n,
            "flag_message": flag_msg,
            "n_attacks": n_stats,
            "attack_success_rate": asr_stats,
            "detection_rate": det_stats,
            "fpr_combined": fpr_stats,
            "corpus_size": corp_stats,
        })

    depth_data = {
        "depth": depth,
        "depth_label": depth_label,
        "n_runs": n_runs,
        "rounds": rounds,
        "base_seed": base_seed,
        "summary_by_round": summary_rounds,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(depth_data, f, indent=2)

    return depth_data


def main():
    parser = argparse.ArgumentParser(description="Run mutation depth capping experiment")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of repeated runs per depth")
    parser.add_argument("--rounds", type=int, default=6, help="Rounds per run")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--output", type=str, default="mutation_depth_comparison.json", help="Combined output JSON")
    args = parser.parse_args()

    depths = [2, 4, 8, None]
    results_by_depth = {}

    for d in depths:
        d_res = run_depth_condition(
            depth=d,
            n_runs=args.n_runs,
            rounds=args.rounds,
            base_seed=args.base_seed,
        )
        label = "unlimited" if d is None else str(d)
        results_by_depth[label] = d_res

    # Output combined comparison matrix
    print("\n" + "=" * 90)
    print("MUTATION DEPTH EXPERIMENT: ATTACK SUCCESS RATE (ASR % Mean +/- Std)")
    print("=" * 90)
    header = f"{'Round':>5} | {'Depth 2':>18} | {'Depth 4':>18} | {'Depth 8':>18} | {'Unlimited':>18}"
    print(header)
    print("-" * len(header))

    combined_matrix = []
    for r_idx in range(args.rounds):
        r_num = r_idx + 1
        d2_asr = results_by_depth["2"]["summary_by_round"][r_idx]["attack_success_rate"]
        d4_asr = results_by_depth["4"]["summary_by_round"][r_idx]["attack_success_rate"]
        d8_asr = results_by_depth["8"]["summary_by_round"][r_idx]["attack_success_rate"]
        du_asr = results_by_depth["unlimited"]["summary_by_round"][r_idx]["attack_success_rate"]

        d2_str = f"{d2_asr['mean']:.1f}+/-{d2_asr['std']:.1f}"
        d4_str = f"{d4_asr['mean']:.1f}+/-{d4_asr['std']:.1f}"
        d8_str = f"{d8_asr['mean']:.1f}+/-{d8_asr['std']:.1f}"
        du_str = f"{du_asr['mean']:.1f}+/-{du_asr['std']:.1f}"

        print(f"{r_num:>5} | {d2_str:>18} | {d4_str:>18} | {d8_str:>18} | {du_str:>18}")

        combined_matrix.append({
            "round": r_num,
            "depth_2_asr": d2_asr,
            "depth_4_asr": d4_asr,
            "depth_8_asr": d8_asr,
            "depth_unlimited_asr": du_asr,
        })

    final_comparison = {
        "metadata": {
            "n_runs": args.n_runs,
            "rounds": args.rounds,
            "base_seed": args.base_seed,
            "depths_tested": [2, 4, 8, "unlimited"],
        },
        "comparison_matrix": combined_matrix,
        "details_by_depth": results_by_depth,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, indent=2)

    print(f"\nCombined depth comparison written to {args.output}")


if __name__ == "__main__":
    main()
