"""
run_n_times.py — Multi-run statistical evaluation harness for AIShield

Runs the full Red Team <-> Blue Team loop N times with seeds
base_seed, base_seed+1, ..., base_seed+N-1.

Computes mean ± std and 95% confidence intervals across runs for:
  - Attack Success Rate (ASR)
  - Detection Rate
  - False-Positive Rate (combined, 1-turn, multi-turn)
  - Attack Corpus Size
  - Attack batch size (n)

Flags rounds where batch size n falls below a threshold (default 20),
warning that confidence intervals and percentage estimates are statistically
unreliable at low sample sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any

from full_loop import run as run_full_loop
from reporting.flags import compute_stats, check_sample_size_flag, format_stat_with_ci


def main():
    parser = argparse.ArgumentParser(description="Run full_loop N times with varying seeds for statistical significance.")
    parser.add_argument("--backend", type=str, required=True, choices=["tfidf", "custom", "anthropic"],
                        help="Semantic similarity backend")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of repeated runs")
    parser.add_argument("--rounds", type=int, default=6, help="Evaluation rounds per run")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM generator")
    parser.add_argument("--no-dataset", action="store_true", help="Skip public dataset")
    parser.add_argument("--n-dataset", type=int, default=30, help="Max dataset prompts")
    parser.add_argument("--max-mutation-depth", type=int, default=None,
                        help="Max mutation depth before fallback to seeds")
    parser.add_argument("--min-sample-size", type=int, default=20,
                        help="Minimum attack batch size for reliable percentage reporting")
    parser.add_argument("--output", type=str, default="n_runs_summary.json", help="Summary JSON output")
    parser.add_argument("--raw-output", type=str, default="n_runs_raw.jsonl", help="Raw per-run JSONL output")
    args = parser.parse_args()

    os.environ["SEMANTIC_BACKEND"] = args.backend
    print(f"Semantic similarity backend: {args.backend}")
    print(f"Starting {args.n_runs} runs x {args.rounds} rounds (base_seed={args.base_seed}, max_mutation_depth={args.max_mutation_depth})...")

    # Clear raw output file
    if os.path.exists(args.raw_output):
        os.remove(args.raw_output)

    # Collect round-by-round metrics across all runs
    # run_metrics[run_idx] = list of round metric dicts
    all_runs_data = []

    for run_idx in range(args.n_runs):
        seed = args.base_seed + run_idx
        corpus_file = f"temp_corpus_run_{run_idx}.json"
        print(f"\n>>> Run {run_idx + 1}/{args.n_runs} (seed={seed})")

        # Suppress noisy intermediate prints from full_loop run if desired, or keep concise
        run_result = run_full_loop(
            rounds=args.rounds,
            use_llm=args.use_llm,
            use_dataset=not args.no_dataset,
            n_dataset=args.n_dataset,
            corpus_path=corpus_file,
            seed=seed,
            max_mutation_depth=args.max_mutation_depth,
            backend=args.backend,
        )

        all_runs_data.append({"run_index": run_idx, "seed": seed, "rounds": run_result})

        # Append to raw jsonl
        with open(args.raw_output, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_index": run_idx, "seed": seed, "rounds": run_result}) + "\n")

        # Cleanup temp corpus
        if os.path.exists(corpus_file):
            try:
                os.remove(corpus_file)
            except Exception:
                pass

    # Aggregate by round
    summary_by_round = []
    print("\n" + "=" * 95)
    print(f"STATISTICAL SUMMARY OVER {args.n_runs} RUNS (Mean ± Std [95% CI])")
    print("=" * 95)

    header = f"{'Round':>5} | {'Attacks (n)':>15} | {'ASR (%)':>18} | {'Detection (%)':>18} | {'FPR (%)':>15} | {'Corpus Size':>12}"
    print(header)
    print("-" * len(header))

    for r_idx in range(args.rounds):
        round_num = r_idx + 1
        n_list = [run_data["rounds"][r_idx]["n_attacks"] for run_data in all_runs_data]
        asr_list = [run_data["rounds"][r_idx]["attack_success_rate"] for run_data in all_runs_data]
        det_list = [run_data["rounds"][r_idx]["detection_rate"] for run_data in all_runs_data]
        fpr_list = [run_data["rounds"][r_idx]["false_positive_rate"] for run_data in all_runs_data]
        fpr_1t_list = [run_data["rounds"][r_idx]["false_positive_rate_single_turn"] for run_data in all_runs_data]
        fpr_mt_list = [run_data["rounds"][r_idx]["false_positive_rate_multi_turn"] for run_data in all_runs_data]
        corp_list = [run_data["rounds"][r_idx]["corpus_size"] for run_data in all_runs_data]

        n_stats = compute_stats(n_list)
        asr_stats = compute_stats(asr_list)
        det_stats = compute_stats(det_list)
        fpr_stats = compute_stats(fpr_list)
        fpr_1t_stats = compute_stats(fpr_1t_list)
        fpr_mt_stats = compute_stats(fpr_mt_list)
        corp_stats = compute_stats(corp_list)

        is_low_n, flag_msg = check_sample_size_flag(n_stats["mean"], threshold=args.min_sample_size)
        flag = f"  {flag_msg}" if is_low_n else ""

        asr_str = f"{asr_stats['mean']:.1f}+/-{asr_stats['std']:.1f} [+/-{asr_stats['ci95']:.1f}]"
        det_str = f"{det_stats['mean']:.1f}+/-{det_stats['std']:.1f} [+/-{det_stats['ci95']:.1f}]"
        fpr_str = f"{fpr_stats['mean']:.1f}+/-{fpr_stats['std']:.1f}"
        n_str = f"{n_stats['mean']:.1f}+/-{n_stats['std']:.1f}"
        corp_str = f"{corp_stats['mean']:.1f}+/-{corp_stats['std']:.1f}"

        print(f"{round_num:>5} | {n_str:>15} | {asr_str:>18} | {det_str:>18} | {fpr_str:>15} | {corp_str:>12}{flag}")

        summary_by_round.append({
            "round": round_num,
            "sample_size_flag": is_low_n,
            "n_attacks": n_stats,
            "attack_success_rate": asr_stats,
            "detection_rate": det_stats,
            "fpr_combined": fpr_stats,
            "fpr_single_turn": fpr_1t_stats,
            "fpr_multi_turn": fpr_mt_stats,
            "corpus_size": corp_stats,
        })

    summary_data = {
        "n_runs": args.n_runs,
        "rounds": args.rounds,
        "base_seed": args.base_seed,
        "max_mutation_depth": args.max_mutation_depth,
        "min_sample_size": args.min_sample_size,
        "semantic_backend": args.backend,
        "summary_by_round": summary_by_round,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary metrics written to {args.output}")
    print(f"Raw run-by-run logs written to {args.raw_output}")


if __name__ == "__main__":
    main()
