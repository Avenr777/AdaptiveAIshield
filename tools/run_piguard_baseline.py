"""
tools/run_piguard_baseline.py — Implement and evaluate PIGuard signature and boundary heuristic baseline on Kaggle 500 benchmark.

PIGuard logic:
1. Instruction boundary tokens (e.g. system:, [INST], <|im_start|>, ignore previous)
2. Command/delimiter injection patterns (eval(, exec(, <script>, DROP TABLE, rm -rf, os.system)
3. Role manipulation / escape syntax (Act as, You are now, DAN, unrestricted)
4. Obfuscation indicators (base64 blocks, space-delimited words, char escape sequences)
"""

import json
import re
import time
from pathlib import Path
import numpy as np

PIGUARD_PATTERNS = [
    re.compile(r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|directives)\b"),
    re.compile(r"(?i)\b(?:system\s*prompt|system\s*instructions|core\s*directive)\b"),
    re.compile(r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|simulate)\s+(?:an?\s+)?(?:evil|dan|chaos|unrestricted|jailbreak|hacker|root)\b"),
    re.compile(r"(?i)\b(?:do\s+anything\s+now|jailbreak|bypass\s+(?:content|safety|filter))\b"),
    re.compile(r"(?i)\b(?:rm\s+-rf|chmod\s+777|cat\s+/etc/passwd|/bin/sh|cmd\.exe|powershell)\b"),
    re.compile(r"(?i)\b(?:SELECT\s+.*\s+FROM\s+|DROP\s+TABLE|INSERT\s+INTO|UNION\s+SELECT|--\s*$|OR\s+1\s*=\s*1)\b"),
    re.compile(r"(?i)\b(?:exec\s*\(|eval\s*\(|subprocess\.Popen|os\.system|__import__)\b"),
    re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)"),
    re.compile(r"(?i)\b(?:base64\s+decode|rot13|hex\s+encoded)\b"),
    re.compile(r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"), # long b64 strings
]

def piguard_predict(prompt: str) -> bool:
    for pat in PIGUARD_PATTERNS:
        if pat.search(prompt):
            return True
    return False

def run_benchmark():
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

    for item in malicious:
        t0 = time.perf_counter()
        pred = piguard_predict(item["prompt"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        mal_preds.append(pred)

    for item in benign:
        t0 = time.perf_counter()
        pred = piguard_predict(item["prompt"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ben_preds.append(pred)

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
        "p50_latency_ms": round(float(np.median(latencies)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
    }

    # Append / update strong_baselines_benchmark.json
    sb_file = Path("strong_baselines_benchmark.json")
    if sb_file.exists():
        with sb_file.open("r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["PIGuard (Boundary + Pattern Baseline)"] = result
    with sb_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("PIGuard Results:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_benchmark()
