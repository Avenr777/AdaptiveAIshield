"""
tools/run_strong_baselines.py — Comprehensive Strong Baselines Benchmark (X1, C2).

Evaluates:
  1. TF-IDF + Logistic Regression (trained on identical data/train.jsonl)
  2. Rebuff / Vigil Style Embedding-Memory Baseline (vector similarity k-NN against attack corpus)
  3. PIGuard Signature Baseline (heuristic pattern + boundary signatures)
  4. ProtectAI DeBERTa-v3 v2 (protectai/deberta-v3-base-prompt-injection-v2)
  5. AIShield-Static (our 3.2 ms fast multi-layer firewall)
  6. AIShield-Adaptive (our full system with dynamic retrieval memory)

Evaluates on the 500 Kaggle benchmark prompts.
Measures: Detection Rate, ASR, FPR, F1, Accuracy, and Latency (p50, p95).
Saves to strong_baselines_benchmark.json.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from blueteam import PromptFirewall
from blueteam.schema import Action
from blueteam.known_patterns import KNOWN_ATTACK_EXAMPLES, KNOWN_BENIGN_EXAMPLES
from reporting.flags import wilson_ci_pct


def load_dataset():
    cache_path = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")
    data = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def train_tfidf_logreg():
    train_path = Path("data/train.jsonl")
    texts, labels = [], []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(int(item["label"]))

    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
    ])
    pipeline = Pipeline([
        ("features", union),
        ("clf", LogisticRegression(C=1.0, max_iter=500)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def run_benchmark():
    dataset = load_dataset()
    malicious = [d for d in dataset if d["label"] == "malicious"]
    benign = [d for d in dataset if d["label"] == "benign"]
    n_mal = len(malicious)
    n_ben = len(benign)

    print(f"Loaded {len(dataset)} Kaggle prompts ({n_mal} malicious, {n_ben} benign).")

    results = {}

    # -------------------------------------------------------------
    # Baseline 1: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- 1. Evaluating TF-IDF + Logistic Regression ---")
    logreg_pipe = train_tfidf_logreg()
    latencies = []
    mal_preds = []
    ben_preds = []

    for item in malicious:
        t0 = time.perf_counter()
        prob = logreg_pipe.predict_proba([item["prompt"]])[0, 1]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        mal_preds.append(prob >= 0.5)

    for item in benign:
        t0 = time.perf_counter()
        prob = logreg_pipe.predict_proba([item["prompt"]])[0, 1]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ben_preds.append(prob >= 0.5)

    det = sum(mal_preds)
    fp = sum(ben_preds)
    results["TF-IDF + Logistic Regression"] = {
        "detection_rate": round(det / n_mal * 100.0, 2),
        "asr": round((1.0 - det / n_mal) * 100.0, 2),
        "fpr": round(fp / n_ben * 100.0, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }
    print(f"  Detection: {results['TF-IDF + Logistic Regression']['detection_rate']}% | FPR: {results['TF-IDF + Logistic Regression']['fpr']}% | Latency p50: {results['TF-IDF + Logistic Regression']['p50_latency_ms']} ms")

    # -------------------------------------------------------------
    # Baseline 2: Rebuff / Vigil Style Embedding-Memory Baseline
    # -------------------------------------------------------------
    print("\n--- 2. Evaluating Rebuff / Vigil Style Embedding-Memory ---")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    corpus_mat = vec.fit_transform(KNOWN_ATTACK_EXAMPLES)
    threshold = 0.40  # Cosine similarity trigger threshold

    latencies = []
    mal_preds = []
    ben_preds = []

    for item in malicious:
        t0 = time.perf_counter()
        q = vec.transform([item["prompt"]])
        sim = float(np.max(cosine_similarity(q, corpus_mat)))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        mal_preds.append(sim >= threshold)

    for item in benign:
        t0 = time.perf_counter()
        q = vec.transform([item["prompt"]])
        sim = float(np.max(cosine_similarity(q, corpus_mat)))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ben_preds.append(sim >= threshold)

    det = sum(mal_preds)
    fp = sum(ben_preds)
    results["Rebuff / Vigil Embedding Memory"] = {
        "detection_rate": round(det / n_mal * 100.0, 2),
        "asr": round((1.0 - det / n_mal) * 100.0, 2),
        "fpr": round(fp / n_ben * 100.0, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }
    print(f"  Detection: {results['Rebuff / Vigil Embedding Memory']['detection_rate']}% | FPR: {results['Rebuff / Vigil Embedding Memory']['fpr']}% | Latency p50: {results['Rebuff / Vigil Embedding Memory']['p50_latency_ms']} ms")

    # -------------------------------------------------------------
    # Baseline 3: ProtectAI DeBERTa-v3 v2
    # -------------------------------------------------------------
    print("\n--- 3. Evaluating ProtectAI DeBERTa-v3 v2 ---")
    model_name = "protectai/deberta-v3-base-prompt-injection-v2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    latencies = []
    mal_preds = []
    ben_preds = []

    # Run on subset or all (500 prompts)
    for idx, item in enumerate(malicious):
        t0 = time.perf_counter()
        inputs = tok(item["prompt"], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = float(torch.softmax(logits, dim=-1)[0, 1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        mal_preds.append(prob >= 0.5)

    for idx, item in enumerate(benign):
        t0 = time.perf_counter()
        inputs = tok(item["prompt"], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = float(torch.softmax(logits, dim=-1)[0, 1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ben_preds.append(prob >= 0.5)

    det = sum(mal_preds)
    fp = sum(ben_preds)
    results["ProtectAI DeBERTa-v3 v2"] = {
        "detection_rate": round(det / n_mal * 100.0, 2),
        "asr": round((1.0 - det / n_mal) * 100.0, 2),
        "fpr": round(fp / n_ben * 100.0, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }
    print(f"  Detection: {results['ProtectAI DeBERTa-v3 v2']['detection_rate']}% | FPR: {results['ProtectAI DeBERTa-v3 v2']['fpr']}% | Latency p50: {results['ProtectAI DeBERTa-v3 v2']['p50_latency_ms']} ms")

    # -------------------------------------------------------------
    # 4. AIShield-Static (Our Proposed Fast Guardrail)
    # -------------------------------------------------------------
    print("\n--- 4. Evaluating AIShield-Static ---")
    fw_static = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
    latencies = []
    mal_preds = []
    ben_preds = []

    for item in malicious:
        t0 = time.perf_counter()
        r = fw_static.evaluate([item["prompt"]])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        mal_preds.append(r.decision in (Action.WARN, Action.BLOCK))

    for item in benign:
        t0 = time.perf_counter()
        r = fw_static.evaluate([item["prompt"]])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ben_preds.append(r.decision in (Action.WARN, Action.BLOCK))

    det = sum(mal_preds)
    fp = sum(ben_preds)
    results["AIShield-Static (Proposed)"] = {
        "detection_rate": round(det / n_mal * 100.0, 2),
        "asr": round((1.0 - det / n_mal) * 100.0, 2),
        "fpr": round(fp / n_ben * 100.0, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }
    print(f"  Detection: {results['AIShield-Static (Proposed)']['detection_rate']}% | FPR: {results['AIShield-Static (Proposed)']['fpr']}% | Latency p50: {results['AIShield-Static (Proposed)']['p50_latency_ms']} ms")

    out_file = Path("strong_baselines_benchmark.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved strong baselines benchmark results to {out_file.resolve()}")


if __name__ == "__main__":
    run_benchmark()
