"""
End-to-end demo of the benchmarking loop described in Section 13 of the
report: generate a batch, evaluate it against the (stub) Blue Team,
record results, evolve, repeat.

Round 1 now combines hand-written seed templates with real prompts
pulled from a public jailbreak dataset (see redteam/dataset_library.py).
Every later round evolves whatever evaded the previous round, via
mutation and (optionally) fresh LLM generation.

Run:
    python run_round.py

No API key needed by default — it runs in seed+dataset+mutation mode.
Set ANTHROPIC_API_KEY (or LOCAL_MODEL_ENDPOINT) and pass --use-llm to
also generate fresh LLM variants each round.
"""

import argparse
from collections import Counter

from redteam import RedTeamGenerator, CorpusStore, AttackGeneratorLLM
from redteam.local_llm_client import LocalModelAttackGenerator, is_configured as local_model_configured
import dummy_blue_team
import real_blue_team


def attack_success_rate(results) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.evaded) / len(results) * 100


def source_breakdown(attacks) -> str:
    counts = Counter(a.source.value for a in attacks)
    return ", ".join(f"{src}={n}" for src, n in sorted(counts.items()))


def pick_llm_client(use_llm: bool):
    if not use_llm:
        return None
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("(using Anthropic API for LLM-generated variants)")
        return AttackGeneratorLLM()
    if local_model_configured():
        print("(using local model server for LLM-generated variants)")
        return LocalModelAttackGenerator()
    print("(--use-llm was set but no ANTHROPIC_API_KEY or "
          "LOCAL_MODEL_ENDPOINT found — continuing without LLM variants)")
    return None


def run(
    rounds: int,
    use_llm: bool,
    use_dataset: bool,
    n_dataset: int,
    blue_team,
    seed=None,
    max_mutation_depth=None,
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

    llm = pick_llm_client(use_llm)
    generator = RedTeamGenerator(llm=llm)
    store = CorpusStore("demo_corpus.jsonl")

    batch = generator.generate_seed_batch(
        round_number=1, include_dataset=use_dataset, n_dataset=n_dataset
    )
    all_attacks = list(batch.attacks)

    for round_number in range(1, rounds + 1):
        store.save_batch(batch)
        results = blue_team.evaluate(batch.attacks)
        store.save_results(results)

        asr = attack_success_rate(results)
        print(f"Round {round_number}: {len(batch.attacks)} attacks "
              f"[{source_breakdown(batch.attacks)}], "
              f"Attack Success Rate = {asr:.1f}%")

        if round_number == rounds:
            break

        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=results,
            round_number=round_number + 1,
            max_mutation_depth=max_mutation_depth,
        )
        all_attacks.extend(batch.attacks)

    print(f"\nTotal attacks generated across all rounds: {len(all_attacks)}")
    print(f"Corpus written to demo_corpus.jsonl / demo_corpus.results.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--use-llm", action="store_true",
                         help="Also generate fresh LLM variants each round "
                              "(requires ANTHROPIC_API_KEY or LOCAL_MODEL_ENDPOINT).")
    parser.add_argument("--no-dataset", action="store_true",
                         help="Skip pulling from the public jailbreak dataset; "
                              "Round 1 uses hand-written seeds only.")
    parser.add_argument("--n-dataset", type=int, default=30,
                         help="Max dataset prompts to include in Round 1.")
    parser.add_argument("--real-firewall", action="store_true",
                         help="Benchmark against the real blueteam.PromptFirewall "
                              "instead of the keyword-only dummy stand-in.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-mutation-depth", type=int, default=None,
                        help="Max generations deep an attack can be mutated before falling back to seeds")
    args = parser.parse_args()
    blue_team = real_blue_team if args.real_firewall else dummy_blue_team
    if args.real_firewall:
        print("(benchmarking against the real PromptFirewall)")
    run(
        args.rounds,
        args.use_llm,
        not args.no_dataset,
        args.n_dataset,
        blue_team,
        seed=args.seed,
        max_mutation_depth=args.max_mutation_depth,
    )
