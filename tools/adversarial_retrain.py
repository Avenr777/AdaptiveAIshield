"""
tools/adversarial_retrain.py — Closed-loop Adversarial Retraining Harness (Phase 4).

Executes the adversarial research protocol:
  Round 1: Initial model evaluates baseline attacks.
  Evasions are captured -> dynamically appended to the training corpus.
  The classifier is retrained on the augmented dataset.
  Round 2+: Red team mutates against the updated defense, and the retrained
  classifier is reloaded and evaluated on the new attacks and held-out benign set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam import RedTeamGenerator
from redteam.schema import AttackResult
from blueteam import PromptFirewall
from held_out_benign import HELD_OUT_BENIGN, HELD_OUT_BENIGN_MULTITURN


def retrain_classifier_from_dataset(train_texts: List[str], train_labels: List[int], output_dir: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.naive_bayes import MultinomialNB
    import joblib

    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
    ])

    pipeline = Pipeline([
        ("features", union),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    pipeline.fit(train_texts, train_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classical_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    return pipeline


def evaluate_round(firewall: PromptFirewall, batch_attacks):
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


def run_adversarial_loop(
    rounds: int = 4,
    seed: int = 42,
    model_dir: str = "models/aishield_classifier",
    data_dir: str = "data",
    output_json: str = "adversarial_retrain_results.json",
):
    random.seed(seed)
    model_path = Path(model_dir)
    data_path = Path(data_dir)

    # Initial training data
    train_texts, train_labels = [], []
    with (data_path / "train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                train_texts.append(item["text"])
                train_labels.append(int(item["label"]))

    # Train initial round model
    retrain_classifier_from_dataset(train_texts, train_labels, model_path)

    # Configure firewall with custom classifier backend
    os.environ["SEMANTIC_BACKEND"] = "custom"
    firewall = PromptFirewall(corpus_path="temp_adv_corpus.json")

    generator = RedTeamGenerator()
    batch = generator.generate_seed_batch(round_number=1, include_dataset=True, n_dataset=25)

    print("\n" + "=" * 90)
    print(f"ADVERSARIAL RETRAINING BENCHMARK ({rounds} ROUNDS, SEED={seed})")
    print("Each round: evasions are fed into the training pool -> model retrained in real time.")
    print("=" * 90)

    header = f"{'Round':>5} | {'Attacks':>7} | {'ASR (%)':>8} | {'Detection (%)':>13} | {'FPR (%)':>8} | {'Train Size':>10}"
    print(header)
    print("-" * len(header))

    round_metrics = []

    for round_num in range(1, rounds + 1):
        results, evaded_attacks = evaluate_round(firewall, batch.attacks)
        n = len(results)
        asr = sum(1 for r in results if r.evaded) / n * 100 if n else 0.0
        det = 100.0 - asr
        fpr = evaluate_false_positive_rate(firewall)

        train_size_before = len(train_texts)

        print(
            f"{round_num:>5} | {n:>7} | {asr:>8.1f} | {det:>13.1f} | {fpr['combined_fpr']:>8.1f} | {train_size_before:>10}"
        )

        round_metrics.append({
            "round": round_num,
            "n_attacks": n,
            "asr": round(asr, 2),
            "detection_rate": round(det, 2),
            "fpr": round(fpr["combined_fpr"], 2),
            "train_size": train_size_before,
            "n_evasions": len(evaded_attacks),
        })

        if round_num == rounds:
            break

        # Adversarial Retraining Step:
        # Augment training set with evasions
        new_evasions = 0
        for atk in evaded_attacks:
            t = " ".join(atk.turns)
            if t not in train_texts:
                train_texts.append(t)
                train_labels.append(1)  # label 1 = attack
                new_evasions += 1
                firewall.confirm_attack(t)

        # Retrain model with updated adversarial samples
        retrain_classifier_from_dataset(train_texts, train_labels, model_path)
        # Re-instantiate firewall so it reloads the fresh weights
        firewall = PromptFirewall(corpus_path="temp_adv_corpus.json")

        # Evolve attacker
        batch = generator.evolve_round(
            previous_attacks=batch.attacks,
            results=results,
            round_number=round_num + 1,
            max_mutation_depth=4,
        )

    if Path("temp_adv_corpus.json").exists():
        os.remove("temp_adv_corpus.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": round_metrics}, f, indent=2)

    print(f"\nAdversarial retraining results saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run closed-loop adversarial retraining benchmark")
    parser.add_argument("--rounds", type=int, default=4, help="Number of adversarial rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="adversarial_retrain_results.json", help="Output JSON summary")
    args = parser.parse_args()

    run_adversarial_loop(rounds=args.rounds, seed=args.seed, output_json=args.output)
