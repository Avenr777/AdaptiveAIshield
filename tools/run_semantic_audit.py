"""
tools/run_semantic_audit.py — Semantic Near-Duplicate Audit between Training Set and Kaggle Benchmark.

Computes semantic embedding cosine similarity using TF-IDF subword/char ensemble
and analyzes distribution of maximum similarity scores:
- How many test prompts have similarity >= 0.80, >= 0.85, >= 0.90, >= 0.95 with any training prompt?
- Confirms whether high test recall is driven by memorization of identical embeddings.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity

def run_audit():
    train_path = Path("data/train.jsonl")
    train_texts = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_texts.append(json.loads(line)["text"])

    kaggle_path = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")
    kaggle_data = []
    with kaggle_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kaggle_data.append(json.loads(line))

    test_malicious = [d["prompt"] for d in kaggle_data if d["label"] == "malicious"]
    test_benign = [d["prompt"] for d in kaggle_data if d["label"] == "benign"]

    print(f"Loaded {len(train_texts)} training texts, {len(test_malicious)} test attacks, {len(test_benign)} test benign.")

    union = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("char", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
    ])
    
    all_texts = train_texts + test_malicious + test_benign
    print("Fitting feature union vectorizer...")
    union.fit(all_texts)
    
    X_train = union.transform(train_texts)
    X_mal = union.transform(test_malicious)
    X_ben = union.transform(test_benign)

    sim_mal = cosine_similarity(X_mal, X_train)
    max_sim_mal = np.max(sim_mal, axis=1)

    sim_ben = cosine_similarity(X_ben, X_train)
    max_sim_ben = np.max(sim_ben, axis=1)

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    mal_counts = {str(t): int(np.sum(max_sim_mal >= t)) for t in thresholds}
    ben_counts = {str(t): int(np.sum(max_sim_ben >= t)) for t in thresholds}

    results = {
        "n_train": len(train_texts),
        "n_test_malicious": len(test_malicious),
        "n_test_benign": len(test_benign),
        "max_similarity_malicious": {
            "mean": float(np.mean(max_sim_mal)),
            "median": float(np.median(max_sim_mal)),
            "p95": float(np.percentile(max_sim_mal, 95)),
            "max": float(np.max(max_sim_mal)),
            "threshold_exceedances": mal_counts
        },
        "max_similarity_benign": {
            "mean": float(np.mean(max_sim_ben)),
            "median": float(np.median(max_sim_ben)),
            "p95": float(np.percentile(max_sim_ben, 95)),
            "max": float(np.max(max_sim_ben)),
            "threshold_exceedances": ben_counts
        }
    }

    out_file = Path("semantic_audit_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_file.resolve()}:")
    print("Malicious overlap >= 0.80:", mal_counts["0.8"])
    print("Malicious overlap >= 0.90:", mal_counts["0.9"])
    print("Malicious overlap >= 0.95:", mal_counts["0.95"])
    print("Malicious mean max similarity:", round(results["max_similarity_malicious"]["mean"], 3))

if __name__ == "__main__":
    run_audit()
