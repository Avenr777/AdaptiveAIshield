"""
The complete, closed Red Team <-> Blue Team loop, matching the
benchmarking methodology in Section 13 of the report:

  1. Red Team generates a batch (Round 1: seed templates + public
     dataset; later rounds: mutations/LLM variants of whatever evaded).
  2. Blue Team evaluates the batch AND a held-out benign set in the
     same pass, so both Attack Success Rate and False-Positive Rate
     are measured every round.
  3. Attacks that evaded are fed back into the Blue Team's corpus via
     firewall.confirm_attack() — the Adaptive Learning Mechanism. This
     is the step that was missing before: previously the Blue Team
     never actually got smarter between rounds.
  4. Round-over-round metrics are recorded and printed as a table.
  5. Repeat.

Run:
    python full_loop.py --rounds 6
"""

from __future__ import annotations

import argparse
import json
import os

from redteam import RedTeamGenerator, AttackGeneratorLLM
from redteam.schema import AttackResult
from redteam.local_llm_client import LocalModelAttackGenerator, is_configured as local_model_configured
from blueteam import PromptFirewall
from held_out_benign import HELD_OUT_BENIGN, HELD_OUT_BENIGN_MULTITURN


def pick_llm_client(use_llm: bool):
    if not use_llm:
        return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("(using Anthropic API for LLM-generated variants)")
        return AttackGeneratorLLM()
    if local_model_configured():
        print("(using local model server for LLM-generated variants)")
        return LocalModelAttackGenerator()
    print("(--use-llm set but no model configured — continuing without it)")
    return None


def evaluate_round(firewall: PromptFirewall, batch_attacks):
    """Evaluate the attack batch, returning (AttackResult list, evaded_attacks)."""
    results = []
    evaded_attacks = []
    for attack in batch_attacks:
        r = firewall.evaluate(attack.turns)
        evaded = (r.decision.value == "allow")
        results.append(AttackResult(
            attack_id=attack.id,
            decision=r.decision.value,
            risk_score=r.risk_score,
            evaded=evaded,
        ))
        if evaded:
            evaded_attacks.append(attack)
    return results, evaded_attacks


def evaluate_false_positive_rate(firewall: PromptFirewall) -> dict:
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
        "combined_fpr": flagged / total * 100 if total else 0.0,
        "single_turn_fpr": single_flagged / len(HELD_OUT_BENIGN) * 100,
        "multi_turn_fpr": multi_flagged / len(HELD_OUT_BENIGN_MULTITURN) * 100,
    }


def run(
    rounds: int,
    use_llm: bool,
    use_dataset: bool,
    n_dataset: int,
    corpus_path: str,
    seed: Optional[int] = None,
    max_mutation_depth: Optional[int] = None,
):
    import random
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    print(f"[reproducibility] Using seed: {seed}")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # Fresh corpus each run so results are reproducible/comparable.
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

    llm = pick_llm_client(use_llm)
    generator = RedTeamGenerator(llm=llm)
    firewall = PromptFirewall(corpus_path=corpus_path)

    batch = generator.generate_seed_batch(
        round_number=1, include_dataset=use_dataset, n_dataset=n_dataset
    )

    metrics = []
    header = f"{'Round':>5} | {'Attacks':>7} | {'ASR':>6} | {'Detection':>9} | {'FPR':>6} | {'FPR-1turn':>9} | {'FPR-multi':>9} | {'Corpus':>6}"
    print(header)
    print("-" * len(header))

    for round_number in range(1, rounds + 1):
        results, evaded_attacks = evaluate_round(firewall, batch.attacks)

        n = len(results)
        asr = sum(1 for r in results if r.evaded) / n * 100 if n else 0.0
        detection_rate = 100 - asr
        fpr = evaluate_false_positive_rate(firewall)

        # --- Adaptive learning: confirm evaded attacks into the corpus ---
        for attack in evaded_attacks:
            firewall.confirm_attack(" ".join(attack.turns))

        corpus_size = firewall.corpus.learned_count
        metrics.append({
            "round": round_number,
            "seed": seed,
            "n_attacks": n,
            "attack_success_rate": round(asr, 1),
            "detection_rate": round(detection_rate, 1),
            "false_positive_rate": round(fpr["combined_fpr"], 1),
            "false_positive_rate_single_turn": round(fpr["single_turn_fpr"], 1),
            "false_positive_rate_multi_turn": round(fpr["multi_turn_fpr"], 1),
            "corpus_size": corpus_size,
        })
        print(f"{round_number:>5} | {n:>7} | {asr:>5.1f}% | {detection_rate:>8.1f}% | "
              f"{fpr['combined_fpr']:>5.1f}% | {fpr['single_turn_fpr']:>8.1f}% | "
              f"{fpr['multi_turn_fpr']:>8.1f}% | {corpus_size:>6}")

        if round_number == rounds:
            break

        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=results,
            round_number=round_number + 1,
            max_mutation_depth=max_mutation_depth,
        )

    with open("full_loop_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to full_loop_metrics.json")
    print(f"Learned-attack corpus written to {corpus_path} "
          f"({firewall.corpus.learned_count} confirmed attacks)")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--n-dataset", type=int, default=30)
    parser.add_argument("--corpus-path", type=str, default="full_loop_corpus.json")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-mutation-depth", type=int, default=None,
                        help="Max generations deep an attack can be mutated before falling back to seeds")
    args = parser.parse_args()
    run(
        args.rounds,
        args.use_llm,
        not args.no_dataset,
        args.n_dataset,
        args.corpus_path,
        seed=args.seed,
        max_mutation_depth=args.max_mutation_depth,
    )
