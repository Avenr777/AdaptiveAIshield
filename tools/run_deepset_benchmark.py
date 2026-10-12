"""
tools/run_deepset_benchmark.py — Download and evaluate deepset/prompt-injections held-out test split.

Tests AIShield-Static and baselines against deepset/prompt-injections:
Contains 100 test prompts (injection=1, benign=0).
"""

import json
import urllib.request
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blueteam import PromptFirewall
from blueteam.schema import Action

def fetch_deepset_test():
    url = "https://datasets-server.huggingface.co/rows?dataset=deepset%2Fprompt-injections&config=default&split=test&offset=0&limit=100"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        rows = [r["row"] for r in data.get("rows", [])]
    return rows

def run_deepset():
    print("Fetching deepset/prompt-injections test split...")
    rows = fetch_deepset_test()
    print(f"Retrieved {len(rows)} samples.")

    # 1 is injection, 0 is benign
    malicious = [r["text"] for r in rows if r["label"] == 1]
    benign = [r["text"] for r in rows if r["label"] == 0]
    print(f"Split: {len(malicious)} malicious, {len(benign)} benign.")

    fw_static = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
    fw_lexical = PromptFirewall(weights={"lexical": 1.0, "semantic": 0.0, "context": 0.0}, veto_mode=False)

    def eval_fw(fw):
        latencies = []
        mal_det = 0
        ben_fp = 0
        for text in malicious:
            t0 = time.perf_counter()
            res = fw.evaluate([text])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.decision in (Action.WARN, Action.BLOCK):
                mal_det += 1
        for text in benign:
            t0 = time.perf_counter()
            res = fw.evaluate([text])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.decision in (Action.WARN, Action.BLOCK):
                ben_fp += 1
        
        n_m = len(malicious)
        n_b = len(benign)
        det_pct = (mal_det / n_m) * 100.0 if n_m else 0.0
        fpr_pct = (ben_fp / n_b) * 100.0 if n_b else 0.0
        return {
            "detection_rate": round(det_pct, 2),
            "asr": round(100.0 - det_pct, 2),
            "fpr": round(fpr_pct, 2),
            "p50_latency_ms": round(float(np.median(latencies)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "n_malicious": n_m,
            "n_benign": n_b
        }

    results = {
        "dataset": "deepset/prompt-injections (test split)",
        "AIShield-Static": eval_fw(fw_static),
        "Lexical-Only": eval_fw(fw_lexical)
    }

    out_file = Path("deepset_benchmark_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Results on deepset/prompt-injections:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_deepset()
