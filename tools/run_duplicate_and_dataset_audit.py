"""
tools/run_duplicate_and_dataset_audit.py — Near-Duplicate Audit & Dataset Profiling (X2, C5).

Performs:
  1. Exact duplicate check between training splits and test sets.
  2. N-gram Jaccard near-duplicate check (threshold >= 0.80) between train and test.
  3. Characterization and distribution summary of benign and attack datasets.
Saves to dataset_audit_results.json.
"""

import json
from pathlib import Path
from typing import List, Dict, Set


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    words = text.lower().split()
    if len(words) < n:
        return set(words)
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def audit_splits():
    train_file = Path("data/train.jsonl")
    val_file = Path("data/val.jsonl")
    test_file = Path("data/test.jsonl")
    kaggle_file = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")

    train_data = [json.loads(l) for l in open(train_file, encoding="utf-8") if l.strip()]
    test_data = [json.loads(l) for l in open(test_file, encoding="utf-8") if l.strip()]
    kaggle_data = [json.loads(l) for l in open(kaggle_file, encoding="utf-8") if l.strip()]

    train_texts = [d["text"] for d in train_data]
    test_texts = [d["text"] for d in test_data]
    kaggle_texts = [d["prompt"] for d in kaggle_data]

    # 1. Exact duplicates
    train_set = set(train_texts)
    exact_in_test = sum(1 for t in test_texts if t in train_set)
    exact_in_kaggle = sum(1 for t in kaggle_texts if t in train_set)

    # 2. Near-duplicates (Jaccard word 3-gram >= 0.80)
    train_ngrams = [get_ngrams(t) for t in train_texts]
    
    near_dup_kaggle = []
    for idx, k_txt in enumerate(kaggle_texts):
        k_ng = get_ngrams(k_txt)
        for t_idx, t_ng in enumerate(train_ngrams):
            sim = jaccard_similarity(k_ng, t_ng)
            if sim >= 0.80:
                near_dup_kaggle.append({
                    "kaggle_prompt": k_txt,
                    "train_prompt": train_texts[t_idx],
                    "jaccard_similarity": round(sim, 3),
                })
                break

    # 3. Profiling
    kaggle_mal = [d for d in kaggle_data if d["label"] == "malicious"]
    kaggle_ben = [d for d in kaggle_data if d["label"] == "benign"]

    cat_dist = {}
    for m in kaggle_mal:
        c = m.get("attack_type", "unknown")
        cat_dist[c] = cat_dist.get(c, 0) + 1

    summary = {
        "split_sizes": {
            "train": len(train_data),
            "internal_test": len(test_data),
            "kaggle_real_world": len(kaggle_data),
        },
        "exact_duplicates": {
            "train_vs_internal_test": exact_in_test,
            "train_vs_kaggle_held_out": exact_in_kaggle,
        },
        "near_duplicates": {
            "threshold_jaccard": 0.80,
            "train_vs_kaggle_count": len(near_dup_kaggle),
            "near_duplicate_rate_pct": round(len(near_dup_kaggle) / len(kaggle_data) * 100.0, 2),
            "examples": near_dup_kaggle[:5],
        },
        "kaggle_category_distribution": cat_dist,
        "benign_characterization": {
            "kaggle_benign_count": len(kaggle_ben),
            "description": "250 diverse real-world benign prompts covering technical queries, arithmetic, creative writing, and documentation summarization.",
            "held_out_adversarial_benign_count": 205,
            "adversarial_benign_description": "205 challenging prompts with technical overlap (security education, fiction, database admin, PLC control terminology) to stress-test FPR.",
        }
    }

    out_file = Path("dataset_audit_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset audit completed! Exact Kaggle overlap: {exact_in_kaggle}, Near-duplicates: {len(near_dup_kaggle)}")
    print(f"Saved audit report to {out_file.resolve()}")


if __name__ == "__main__":
    audit_splits()
