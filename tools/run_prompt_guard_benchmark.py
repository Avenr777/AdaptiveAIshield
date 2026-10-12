"""
tools/run_prompt_guard_benchmark.py — Evaluate meta-llama/Prompt-Guard-86M on Kaggle 500 benchmark.
"""

import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "meta-llama/Prompt-Guard-86M"

def run_prompt_guard():
    print(f"Loading {MODEL_ID} using token...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {torch.device}")

        tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
        model.eval()
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("Meta has likely not approved access yet. Please check https://huggingface.co/meta-llama/Prompt-Guard-86M")
        return

    print("Model loaded successfully! Loading Kaggle 500 dataset...")
    kaggle_path = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")
    data = []
    with kaggle_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    malicious = [d for d in data if d["label"] == "malicious"]
    benign = [d for d in data if d["label"] == "benign"]

    latencies = []
    mal_preds = []
    ben_preds = []

    print(f"Evaluating on {len(malicious)} malicious and {len(benign)} benign prompts...")

    # Prompt Guard classes:
    # 0: BENIGN
    # 1: INJECTION
    # 2: JAILBREAK
    # If predicted label is 1 or 2, or P(INJECTION | JAILBREAK) >= 0.5 -> Flag as attack

    with torch.no_grad():
        for item in malicious:
            t0 = time.perf_counter()
            inputs = tok(item["prompt"], return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            # Sum of injection + jailbreak probability
            attack_prob = float(probs[1] + probs[2])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            mal_preds.append(attack_prob >= 0.5)

        for item in benign:
            t0 = time.perf_counter()
            inputs = tok(item["prompt"], return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            attack_prob = float(probs[1] + probs[2])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            ben_preds.append(attack_prob >= 0.5)

    n_mal = len(malicious)
    n_ben = len(benign)
    tp = sum(mal_preds)
    fp = sum(ben_preds)

    det_rate = (tp / n_mal) * 100.0
    fpr = (fp / n_ben) * 100.0
    asr = 100.0 - det_rate
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
    rec = det_rate
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    result = {
        "detection_rate": round(det_rate, 2),
        "asr": round(asr, 2),
        "fpr": round(fpr, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
    }

    # Update strong_baselines_benchmark.json
    sb_file = Path("strong_baselines_benchmark.json")
    if sb_file.exists():
        with sb_file.open("r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["Llama Prompt Guard 86M (Meta)"] = result
    with sb_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nPrompt Guard 86M Results:")
    print(json.dumps(result, indent=2))
    print(f"Successfully saved to {sb_file.resolve()}")

if __name__ == "__main__":
    run_prompt_guard()
