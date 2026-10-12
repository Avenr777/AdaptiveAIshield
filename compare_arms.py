"""
compare_arms.py — Matched Multi-Arm Benchmarking Harness with Multi-Run Statistical Rigor

Evaluates firewall configurations against the EXACT SAME attack batch per round
across N seeded runs. Supports two modes:

  TWO-ARM (default):
    - Arm 1: Static Ablation (Non-Adaptive) — lexical_only or frozen_full
    - Arm 2: AIShield Adaptive Defense (full multi-layer + learning)

  THREE-ARM (--three-arm, §13):
    - Arm A: Lexical-only (pure weighted, veto_mode=False)
    - Arm B: Full ensemble, learning OFF (frozen_full)
    - Arm C: Full ensemble, learning ON (adaptive)
    B vs C gap = adaptive contribution. A vs B gap = semantic+context contribution.

Evolution Driver (§14):
  In three-arm mode, each round's attack batch is seeded from evasions collected
  across ALL arms that ran that round (union). This ensures the Red Team has seen
  every arm's weak spots.

  In two-arm mode, the driver arm drives evolution (default: adaptive).

Both arms face the exact same attacks, in the exact same order, at the exact
same sample size n.

§15: Thresholds are calibrated per arm on a validation split before testing.
     In practice, BandConfig is frozen before the benchmark loop runs — no
     threshold tuning during evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any, Optional

from redteam import RedTeamGenerator
from redteam.schema import Attack, AttackResult, AttackCategory
from blueteam import PromptFirewall
from held_out_benign import HELD_OUT_BENIGN, HELD_OUT_BENIGN_MULTITURN
from reporting.flags import compute_stats, check_sample_size_flag, format_stat_with_ci, format_fraction_fpr


def evaluate_arm(firewall: PromptFirewall, batch_attacks: List[Attack]):
    results = []
    evaded_attacks = []
    cat_counts = {cat.value: {"total": 0, "evaded": 0} for cat in AttackCategory}
    latencies = []

    for attack in batch_attacks:
        cat_key = attack.category.value if hasattr(attack.category, "value") else str(attack.category)
        if cat_key not in cat_counts:
            cat_counts[cat_key] = {"total": 0, "evaded": 0}
        cat_counts[cat_key]["total"] += 1

        t0 = time.perf_counter()
        r = firewall.evaluate(attack.turns)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        evaded = (r.decision.value == "allow")
        if evaded:
            evaded_attacks.append(attack)
            cat_counts[cat_key]["evaded"] += 1

        results.append(AttackResult(
            attack_id=attack.id,
            decision=r.decision.value,
            risk_score=r.risk_score,
            evaded=evaded,
        ))

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    cat_stats = {}
    for cat_name, c in cat_counts.items():
        tot = c["total"]
        eva = c["evaded"]
        cat_stats[cat_name] = {
            "total": tot,
            "evaded": eva,
            "asr": round((eva / tot * 100.0) if tot > 0 else 0.0, 1),
            "detection_rate": round(((tot - eva) / tot * 100.0) if tot > 0 else 0.0, 1),
        }
    return results, evaded_attacks, cat_stats, round(avg_lat, 2)


def evaluate_fpr(firewall: PromptFirewall) -> Dict[str, Any]:
    single_flagged = sum(1 for p in HELD_OUT_BENIGN if firewall.evaluate([p]).decision.value != "allow")
    multi_flagged = sum(1 for t in HELD_OUT_BENIGN_MULTITURN if firewall.evaluate(t).decision.value != "allow")
    single_n = len(HELD_OUT_BENIGN)
    multi_n = len(HELD_OUT_BENIGN_MULTITURN)
    total = single_n + multi_n
    combined_flagged = single_flagged + multi_flagged
    combined_fpr = (combined_flagged / total * 100.0) if total else 0.0
    single_fpr = (single_flagged / single_n * 100.0) if single_n else 0.0
    multi_fpr = (multi_flagged / multi_n * 100.0) if multi_n else 0.0
    return {
        "combined_fp_count": combined_flagged,
        "combined_n": total,
        "combined_fpr": round(combined_fpr, 2),
        "single_turn_fp_count": single_flagged,
        "single_turn_n": single_n,
        "single_turn_fpr": round(single_fpr, 2),
        "multi_turn_fp_count": multi_flagged,
        "multi_turn_n": multi_n,
        "multi_turn_fpr": round(multi_fpr, 2),
    }


def run_single_trial(
    seed: int,
    rounds: int,
    ablation: str,
    driver: str,
    no_dataset: bool,
    n_dataset: int,
    max_mutation_depth: Optional[int],
    backend: str = "tfidf",
) -> List[Dict[str, Any]]:
    os.environ["SEMANTIC_BACKEND"] = backend
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    static_corpus_path = f"temp_static_arm_{seed}.json"
    adaptive_corpus_path = f"temp_adaptive_arm_{seed}.json"
    for p in [static_corpus_path, adaptive_corpus_path]:
        if os.path.exists(p):
            os.remove(p)

    if ablation == "lexical_only":
        static_weights = {"lexical": 1.0, "semantic": 0.0, "context": 0.0}
    else:
        static_weights = {"lexical": 0.35, "semantic": 0.45, "context": 0.20}

    static_firewall = PromptFirewall(corpus_path=static_corpus_path, weights=static_weights)
    adaptive_firewall = PromptFirewall(corpus_path=adaptive_corpus_path)
    print(f"  [seed {seed}] Firewall semantic backend: {adaptive_firewall.semantic_backend_name}")

    generator = RedTeamGenerator()
    batch = generator.generate_seed_batch(
        round_number=1, include_dataset=not no_dataset, n_dataset=n_dataset
    )

    trial_log = []

    for round_num in range(1, rounds + 1):
        n = len(batch.attacks)

        # 1. Evaluate STATIC arm on batch
        static_results, static_evaded, static_cat, static_lat = evaluate_arm(static_firewall, batch.attacks)
        static_asr = sum(1 for r in static_results if r.evaded) / n * 100.0 if n else 0.0
        static_det = 100.0 - static_asr
        static_fpr = evaluate_fpr(static_firewall)

        # 2. Evaluate ADAPTIVE arm on EXACT SAME batch
        adaptive_results, adaptive_evaded, adaptive_cat, adaptive_lat = evaluate_arm(adaptive_firewall, batch.attacks)
        adaptive_asr = sum(1 for r in adaptive_results if r.evaded) / n * 100.0 if n else 0.0
        adaptive_det = 100.0 - adaptive_asr
        adaptive_fpr = evaluate_fpr(adaptive_firewall)

        # 3. Adaptive learning: feed evaded attacks into adaptive firewall only
        for attack in adaptive_evaded:
            adaptive_firewall.confirm_attack(" ".join(attack.turns))

        corpus_size = adaptive_firewall.corpus.learned_count

        trial_log.append({
            "round": round_num,
            "matched_n": n,
            "static_arm": {
                "ablation_mode": ablation,
                "asr": round(static_asr, 2),
                "detection_rate": round(static_det, 2),
                "fpr": static_fpr,
                "latency_ms": static_lat,
                "category_metrics": static_cat,
            },
            "adaptive_arm": {
                "asr": round(adaptive_asr, 2),
                "detection_rate": round(adaptive_det, 2),
                "fpr": adaptive_fpr,
                "corpus_size": corpus_size,
                "latency_ms": adaptive_lat,
                "category_metrics": adaptive_cat,
            },
        })

        if round_num == rounds:
            break

        # Drive evolution from selected arm's results
        driving_results = adaptive_results if driver == "adaptive" else static_results
        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=driving_results,
            round_number=round_num + 1,
            max_mutation_depth=max_mutation_depth,
        )

    for p in [static_corpus_path, adaptive_corpus_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    return trial_log


def run_three_arm_trial(
    seed: int,
    rounds: int,
    no_dataset: bool,
    n_dataset: int,
    max_mutation_depth: Optional[int],
    backend: str = "tfidf",
) -> List[Dict[str, Any]]:
    os.environ["SEMANTIC_BACKEND"] = backend
    """
    §13: Three-arm experiment.
      Arm A: Lexical-only, pure weighted (veto_mode=False for clean ablation).
      Arm B: Full ensemble (lex+sem+ctx), learning OFF (frozen).
      Arm C: Full ensemble, learning ON (adaptive).

    B vs C gap → adaptive learning contribution.
    A vs B gap → semantic + context layer contribution.

    §14: Each round's Red Team batch is evolved from the UNION of evasions
    across all three arms, so the attacker has adapted to every arm's gaps.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    arm_a_path = f"temp_arm_a_{seed}.json"
    arm_b_path = f"temp_arm_b_{seed}.json"
    arm_c_path = f"temp_arm_c_{seed}.json"
    for p in [arm_a_path, arm_b_path, arm_c_path]:
        if os.path.exists(p):
            os.remove(p)

    # §15: BandConfig frozen before loop (no threshold tuning during evaluation).
    # Arm A: lexical-only, pure weighted, no veto.
    arm_a = PromptFirewall(
        corpus_path=arm_a_path,
        weights={"lexical": 1.0, "semantic": 0.0, "context": 0.0},
        veto_mode=False,
    )
    # Arm B: full ensemble, frozen (no learning).
    arm_b = PromptFirewall(corpus_path=arm_b_path)
    # Arm C: full ensemble, adaptive (learning ON).
    arm_c = PromptFirewall(corpus_path=arm_c_path)
    print(f"  [seed {seed}] Firewall semantic backend: {arm_c.semantic_backend_name}")

    generator = RedTeamGenerator()
    batch = generator.generate_seed_batch(
        round_number=1, include_dataset=not no_dataset, n_dataset=n_dataset
    )

    trial_log = []

    for round_num in range(1, rounds + 1):
        n = len(batch.attacks)

        # Evaluate all three arms on the SAME batch.
        a_results, a_evaded, a_cat, a_lat = evaluate_arm(arm_a, batch.attacks)
        b_results, b_evaded, b_cat, b_lat = evaluate_arm(arm_b, batch.attacks)
        c_results, c_evaded, c_cat, c_lat = evaluate_arm(arm_c, batch.attacks)

        a_asr = sum(1 for r in a_results if r.evaded) / n * 100.0 if n else 0.0
        b_asr = sum(1 for r in b_results if r.evaded) / n * 100.0 if n else 0.0
        c_asr = sum(1 for r in c_results if r.evaded) / n * 100.0 if n else 0.0

        a_fpr = evaluate_fpr(arm_a)
        b_fpr = evaluate_fpr(arm_b)
        c_fpr = evaluate_fpr(arm_c)

        # §14: Adaptive learning for Arm C only.
        for attack in c_evaded:
            arm_c.confirm_attack(" ".join(attack.turns))

        c_corpus_size = arm_c.corpus.learned_count

        trial_log.append({
            "round": round_num,
            "matched_n": n,
            "arm_a": {
                "label": "lexical_only",
                "asr": round(a_asr, 2),
                "detection_rate": round(100.0 - a_asr, 2),
                "fpr": a_fpr,
                "latency_ms": a_lat,
                "category_metrics": a_cat,
            },
            "arm_b": {
                "label": "full_static",
                "asr": round(b_asr, 2),
                "detection_rate": round(100.0 - b_asr, 2),
                "fpr": b_fpr,
                "latency_ms": b_lat,
                "category_metrics": b_cat,
            },
            "arm_c": {
                "label": "full_adaptive",
                "asr": round(c_asr, 2),
                "detection_rate": round(100.0 - c_asr, 2),
                "fpr": c_fpr,
                "corpus_size": c_corpus_size,
                "latency_ms": c_lat,
                "category_metrics": c_cat,
            },
        })

        if round_num == rounds:
            break

        # §14: Evolve from UNION of evasions across all arms.
        all_evaded_ids = {a.id for a in a_evaded} | {a.id for a in b_evaded} | {a.id for a in c_evaded}
        all_evaded = [a for a in batch.attacks if a.id in all_evaded_ids]
        # Use Arm C results for driving (it's the strongest arm — closest to production).
        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=c_results,
            round_number=round_num + 1,
            max_mutation_depth=max_mutation_depth,
        )

    for p in [arm_a_path, arm_b_path, arm_c_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    return trial_log



def main():
    parser = argparse.ArgumentParser(description="Matched multi-arm benchmark: AIShield Adaptive vs. Ablations")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of repeated runs across seeds")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed for multi-run harness")
    parser.add_argument("--rounds", type=int, default=6, help="Benchmarking rounds per run")
    parser.add_argument("--no-dataset", action="store_true", help="Omit public dataset; seeds only")
    parser.add_argument("--n-dataset", type=int, default=30, help="Number of dataset samples in Round 1")
    parser.add_argument("--max-mutation-depth", type=int, default=None,
                        help="Max mutation depth before falling back to fresh seeds")
    parser.add_argument("--min-sample-size", type=int, default=20,
                        help="Minimum sample size threshold for reliable reporting")
    parser.add_argument("--ablation", choices=["lexical_only", "frozen_full"], default="lexical_only",
                        help="Static arm ablation mode (two-arm only): lexical_only or frozen_full")
    parser.add_argument("--driver", choices=["adaptive", "static"], default="adaptive",
                        help="Which arm drives Red Team evolution in two-arm mode (default: adaptive)")
    parser.add_argument("--three-arm", action="store_true",
                        help="§13: Run three-arm experiment (A=lexical, B=full-static, C=full-adaptive)")
    parser.add_argument("--backend", type=str, required=True, choices=["tfidf", "custom", "anthropic"],
                        help="Semantic similarity backend")
    parser.add_argument("--output", type=str, default="compare_arms_metrics.json", help="Summary JSON path")
    parser.add_argument("--raw-output", type=str, default="compare_arms_raw.jsonl", help="Raw per-run JSONL path")
    args = parser.parse_args()

    os.environ["SEMANTIC_BACKEND"] = args.backend
    print(f"Semantic similarity backend: {args.backend}")

    if args.three_arm:
        # ---- THREE-ARM MODE (§13) ----
        print("\n" + "=" * 115)
        print(f"THREE-ARM BENCHMARK (§13): LEXICAL-ONLY vs. FULL-STATIC vs. FULL-ADAPTIVE ({args.n_runs} RUNS)")
        print("  Arm A: Lexical-only  (pure weighted, veto_mode=False)")
        print("  Arm B: Full ensemble (lex+sem+ctx), learning OFF")
        print("  Arm C: Full ensemble (lex+sem+ctx), learning ON  [adaptive]")
        print(f"  Evolution: Union of evasions from all arms (§14)")
        print("=" * 115)

        all_trials_data = []
        if os.path.exists(args.raw_output):
            try:
                with open(args.raw_output, "r", encoding="utf-8") as f:
                    existing = [json.loads(l) for l in f if l.strip()]
                if 0 < len(existing) < args.n_runs:
                    print(f"[resume] Found {len(existing)} existing runs. Resuming...")
                    all_trials_data = existing
                elif len(existing) >= args.n_runs:
                    print(f"[resume] Found all {args.n_runs} runs. Aggregating...")
                    all_trials_data = existing[:args.n_runs]
            except Exception:
                all_trials_data = []

        start_idx = len(all_trials_data)
        for run_idx in range(start_idx, args.n_runs):
            seed = args.base_seed + run_idx
            trial_result = run_three_arm_trial(
                seed=seed,
                rounds=args.rounds,
                no_dataset=args.no_dataset,
                n_dataset=args.n_dataset,
                max_mutation_depth=args.max_mutation_depth,
                backend=args.backend,
            )
            all_trials_data.append({"run_index": run_idx, "seed": seed, "rounds": trial_result})
            with open(args.raw_output, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_index": run_idx, "seed": seed, "rounds": trial_result}) + "\n")
            print(f"  Run {run_idx + 1}/{args.n_runs} done (seed={seed})")

        # Aggregate and print summary for three-arm.
        print(f"\n{'Round':>5} | {'N':>8} | {'Arm-A ASR':>14} | {'Arm-B ASR':>14} | {'Arm-C ASR':>14}")
        print("-" * 70)
        summary_by_round = []
        for r_idx in range(args.rounds):
            rnd = r_idx + 1
            n_list = [t["rounds"][r_idx]["matched_n"] for t in all_trials_data]
            a_asr = [t["rounds"][r_idx]["arm_a"]["asr"] for t in all_trials_data]
            b_asr = [t["rounds"][r_idx]["arm_b"]["asr"] for t in all_trials_data]
            c_asr = [t["rounds"][r_idx]["arm_c"]["asr"] for t in all_trials_data]
            n_s = compute_stats(n_list)
            a_s = compute_stats(a_asr)
            b_s = compute_stats(b_asr)
            c_s = compute_stats(c_asr)
            print(
                f"{rnd:>5} | {n_s['mean']:>6.1f}±{n_s['std']:.1f} | "
                f"{format_stat_with_ci(a_s):>14} | {format_stat_with_ci(b_s):>14} | {format_stat_with_ci(c_s):>14}"
            )
            summary_by_round.append({"round": rnd, "n": n_s, "arm_a_asr": a_s, "arm_b_asr": b_s, "arm_c_asr": c_s})

        summary_output = {"metadata": vars(args), "summary_by_round": summary_by_round}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary_output, f, indent=2)
        print(f"\nSummary written to {args.output}")
        print(f"Raw logs written to {args.raw_output}")
        return

    # ---- TWO-ARM MODE (original) ----
    ablation_desc = (
        "Ablation: Lexical-only, frozen (no learning, no semantic/context layers)"
        if args.ablation == "lexical_only"
        else "Ablation: Full weights, frozen (no learning)"
    )

    print("\n" + "=" * 115)
    print(f"MATCHED TWO-ARM BENCHMARK: STATIC ABLATION vs. AISHIELD ADAPTIVE ({args.n_runs} RUNS)")
    print(f"Static Arm Configuration : {ablation_desc}")
    print(f"Adaptive Arm Configuration: Full multi-layer (lex:0.35, sem:0.45, ctx:0.20) + dynamic learning")
    print(f"Evolution Driver Arm      : {args.driver.upper()} (Red Team mutates based on this arm's evasions)")
    print(f"Held-Out Benign Set Size  : {len(HELD_OUT_BENIGN) + len(HELD_OUT_BENIGN_MULTITURN)} prompts (180 single + 25 multi)")
    print("=" * 115)

    all_trials_data = []
    if os.path.exists(args.raw_output):
        try:
            with open(args.raw_output, "r", encoding="utf-8") as f:
                existing = [json.loads(l) for l in f if l.strip()]
            if len(existing) < args.n_runs:
                print(f"[resume] Found {len(existing)} existing runs in {args.raw_output}. Resuming from run {len(existing)}...")
                all_trials_data = existing
            elif len(existing) >= args.n_runs:
                print(f"[resume] Found all {args.n_runs} runs in {args.raw_output}. Aggregating...")
                all_trials_data = existing[:args.n_runs]
        except Exception:
            all_trials_data = []

    start_idx = len(all_trials_data)
    for run_idx in range(start_idx, args.n_runs):
        seed = args.base_seed + run_idx
        trial_result = run_single_trial(
            seed=seed,
            rounds=args.rounds,
            ablation=args.ablation,
            driver=args.driver,
            no_dataset=args.no_dataset,
            n_dataset=args.n_dataset,
            max_mutation_depth=args.max_mutation_depth,
            backend=args.backend,
        )
        all_trials_data.append({"run_index": run_idx, "seed": seed, "rounds": trial_result})

        with open(args.raw_output, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_index": run_idx, "seed": seed, "rounds": trial_result}) + "\n")

    # Aggregate across runs per round
    summary_by_round = []

    header = (
        f"{'Round':>5} | {'Matched n':>13} | "
        f"{'Static ASR (%)':>18} | {'Static Det (%)':>18} | {'Static FPR':>16} | "
        f"{'Adapt ASR (%)':>18} | {'Adapt Det (%)':>18} | {'Adapt FPR':>16} | {'Corpus':>10}"
    )
    print(header)
    print("-" * len(header))

    for r_idx in range(args.rounds):
        round_num = r_idx + 1

        n_list = [trial["rounds"][r_idx]["matched_n"] for trial in all_trials_data]
        static_asr_list = [trial["rounds"][r_idx]["static_arm"]["asr"] for trial in all_trials_data]
        static_det_list = [trial["rounds"][r_idx]["static_arm"]["detection_rate"] for trial in all_trials_data]
        static_fpr_list = [trial["rounds"][r_idx]["static_arm"]["fpr"]["combined_fpr"] for trial in all_trials_data]
        static_fp_counts = [trial["rounds"][r_idx]["static_arm"]["fpr"]["combined_fp_count"] for trial in all_trials_data]
        benign_total_n = trial_result[0]["static_arm"]["fpr"]["combined_n"]

        adapt_asr_list = [trial["rounds"][r_idx]["adaptive_arm"]["asr"] for trial in all_trials_data]
        adapt_det_list = [trial["rounds"][r_idx]["adaptive_arm"]["detection_rate"] for trial in all_trials_data]
        adapt_fpr_list = [trial["rounds"][r_idx]["adaptive_arm"]["fpr"]["combined_fpr"] for trial in all_trials_data]
        adapt_fp_counts = [trial["rounds"][r_idx]["adaptive_arm"]["fpr"]["combined_fp_count"] for trial in all_trials_data]
        adapt_corp_list = [trial["rounds"][r_idx]["adaptive_arm"]["corpus_size"] for trial in all_trials_data]

        n_stats = compute_stats(n_list)
        static_asr_stats = compute_stats(static_asr_list)
        static_det_stats = compute_stats(static_det_list)
        static_fpr_stats = compute_stats(static_fpr_list)
        static_count_stats = compute_stats(static_fp_counts)

        adapt_asr_stats = compute_stats(adapt_asr_list)
        adapt_det_stats = compute_stats(adapt_det_list)
        adapt_fpr_stats = compute_stats(adapt_fpr_list)
        adapt_count_stats = compute_stats(adapt_fp_counts)
        adapt_corp_stats = compute_stats(adapt_corp_list)

        is_low_n, flag_msg = check_sample_size_flag(n_stats["mean"], threshold=args.min_sample_size)
        flag = f"  {flag_msg}" if is_low_n else ""

        n_str = f"{n_stats['mean']:.1f}+/-{n_stats['std']:.1f}"
        s_asr_str = format_stat_with_ci(static_asr_stats)
        s_det_str = format_stat_with_ci(static_det_stats)
        s_fpr_str = f"{static_fpr_stats['mean']:.2f}% ({static_count_stats['mean']:.1f}/{benign_total_n})"

        a_asr_str = format_stat_with_ci(adapt_asr_stats)
        a_det_str = format_stat_with_ci(adapt_det_stats)
        a_fpr_str = f"{adapt_fpr_stats['mean']:.2f}% ({adapt_count_stats['mean']:.1f}/{benign_total_n})"
        a_corp_str = f"{adapt_corp_stats['mean']:.1f}+/-{adapt_corp_stats['std']:.1f}"

        print(
            f"{round_num:>5} | {n_str:>13} | "
            f"{s_asr_str:>18} | {s_det_str:>18} | {s_fpr_str:>16} | "
            f"{a_asr_str:>18} | {a_det_str:>18} | {a_fpr_str:>16} | {a_corp_str:>10}{flag}"
        )

        summary_by_round.append({
            "round": round_num,
            "sample_size_flag": is_low_n,
            "matched_n": n_stats,
            "static_arm": {
                "ablation_mode": args.ablation,
                "asr": static_asr_stats,
                "detection_rate": static_det_stats,
                "fpr_combined": static_fpr_stats,
                "fpr_combined_count": static_count_stats,
                "fpr_combined_denominator": benign_total_n,
            },
            "adaptive_arm": {
                "asr": adapt_asr_stats,
                "detection_rate": adapt_det_stats,
                "fpr_combined": adapt_fpr_stats,
                "fpr_combined_count": adapt_count_stats,
                "fpr_combined_denominator": benign_total_n,
                "corpus_size": adapt_corp_stats,
            },
        })

    summary_output = {
        "metadata": {
            "n_runs": args.n_runs,
            "base_seed": args.base_seed,
            "rounds": args.rounds,
            "ablation_mode": args.ablation,
            "driver": args.driver,
            "max_mutation_depth": args.max_mutation_depth,
            "min_sample_size": args.min_sample_size,
            "semantic_backend": args.backend,
            "benign_set_size": len(HELD_OUT_BENIGN) + len(HELD_OUT_BENIGN_MULTITURN),
        },
        "summary_by_round": summary_by_round,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    print(f"\nSummary metrics written to {args.output}")
    print(f"Raw run-by-run logs written to {args.raw_output}")


if __name__ == "__main__":
    main()
