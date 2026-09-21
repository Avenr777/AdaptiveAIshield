"""
Generates three separate JSON files:

  1. dataset_attacks.json   Pulled from a public jailbreak-prompt
                             dataset (falls back to an offline sample
                             if no network / no `datasets` package).
  2. mutator_attacks.json   Programmatic mutations applied on top of
                             the dataset attacks (homoglyphs, base64,
                             leetspeak, etc. — no model involved).
  3. llm_attacks.json       Freshly LLM-generated variants. Tries the
                             Anthropic API first (ANTHROPIC_API_KEY),
                             then a local model server
                             (LOCAL_MODEL_ENDPOINT), and writes an
                             EMPTY list if neither is configured.

Run:
    python generate_three_batches.py
    python generate_three_batches.py --n-dataset 20 --mutations-per-attack 2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from redteam.schema import Attack, GenerationBatch
from redteam.dataset_library import load_public_jailbreak_dataset
from redteam.generator import RedTeamGenerator
from redteam.llm_client import AttackGeneratorLLM
from redteam.local_llm_client import LocalModelAttackGenerator, is_configured as local_model_configured


def save_json(attacks: List[Attack], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in attacks], f, indent=2, ensure_ascii=False)
    print(f"  wrote {len(attacks)} attacks -> {path}")


def build_dataset_batch(n_dataset: int) -> List[Attack]:
    print("[1/3] Loading public jailbreak dataset...")
    attacks = load_public_jailbreak_dataset(limit=n_dataset)
    return attacks


def build_mutator_batch(dataset_attacks: List[Attack], mutations_per_attack: int) -> List[Attack]:
    print("[2/3] Generating mutator variants (no model, programmatic only)...")
    generator = RedTeamGenerator()  # no llm= needed for mutation
    batch = generator.mutate_batch(
        dataset_attacks, round_number=1, mutations_per_attack=mutations_per_attack
    )
    return batch.attacks


def build_llm_batch(n_per_category: int) -> List[Attack]:
    print("[3/3] Generating LLM variants...")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  using Anthropic API (ANTHROPIC_API_KEY found)")
        generator = RedTeamGenerator(llm=AttackGeneratorLLM())
        return generator.generate_llm_batch(round_number=1, n_per_category=n_per_category).attacks

    if local_model_configured():
        print("  using local model server (LOCAL_MODEL_ENDPOINT found)")
        generator = RedTeamGenerator(llm=LocalModelAttackGenerator())
        return generator.generate_llm_batch(round_number=1, n_per_category=n_per_category).attacks

    print("  no ANTHROPIC_API_KEY or LOCAL_MODEL_ENDPOINT set — writing empty list")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dataset", type=int, default=50,
                         help="Max number of prompts to pull from the dataset.")
    parser.add_argument("--mutations-per-attack", type=int, default=2,
                         help="How many mutated variants to create per dataset attack.")
    parser.add_argument("--n-per-category", type=int, default=3,
                         help="LLM variants to generate per category (only used if a model is configured).")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    import random
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    print(f"[reproducibility] Using seed: {seed}")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    os.makedirs(args.outdir, exist_ok=True)

    dataset_attacks = build_dataset_batch(args.n_dataset)
    save_json(dataset_attacks, os.path.join(args.outdir, "dataset_attacks.json"))

    mutator_attacks = build_mutator_batch(dataset_attacks, args.mutations_per_attack)
    save_json(mutator_attacks, os.path.join(args.outdir, "mutator_attacks.json"))

    llm_attacks = build_llm_batch(args.n_per_category)
    save_json(llm_attacks, os.path.join(args.outdir, "llm_attacks.json"))

    total = len(dataset_attacks) + len(mutator_attacks) + len(llm_attacks)
    print(f"\nDone. {total} attacks total across 3 files.")
    if not llm_attacks:
        print("Note: llm_attacks.json is empty — set ANTHROPIC_API_KEY or "
              "LOCAL_MODEL_ENDPOINT to populate it.")


if __name__ == "__main__":
    main()
