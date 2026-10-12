"""
tools/run_layer_ablation.py — Layer Ablation on Real Kaggle Dataset (X3, C1, C4).

Evaluates:
  1. Lexical-Only (w_lex=1.0, w_sem=0.0, w_ctx=0.0)
  2. Semantic-Only (w_lex=0.0, w_sem=1.0, w_ctx=0.0)
  3. Lexical + Semantic (w_lex=0.40, w_sem=0.60, w_ctx=0.0)
  4. AIShield-Static (w_lex=0.35, w_sem=0.45, w_ctx=0.20 with active-layer renormalization)
  5. AIShield-Adaptive (full system with dynamic retrieval memory enabled)

Reports:
  - Exact ALLOW / WARN / BLOCK counts for both malicious and benign sets
  - Detection Rate, ASR, FPR, Precision, Recall, F1, Accuracy, and p50/p95 Latency
  - Saves to layer_ablation_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blueteam import PromptFirewall
from blueteam.schema import Action, BandConfig
from blueteam.semantic_backends import TfidfBackend, CustomClassifierBackend
from reporting.flags import wilson_ci_pct


def load_dataset():
    cache_path = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")
    data = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_ablation():
    dataset = load_dataset()
    malicious_items = [d for d in dataset if d["label"] == "malicious"]
    benign_items = [d for d in dataset if d["label"] == "benign"]

    configs = {
        "Lexical-Only": {
            "weights": {"lexical": 1.0, "semantic": 0.0, "context": 0.0},
            "backend": "custom",
            "veto": False,
            "adaptive_retrieval": False,
        },
        "Semantic-Only": {
            "weights": {"lexical": 0.0, "semantic": 1.0, "context": 0.0},
            "backend": "custom",
            "veto": True,
            "adaptive_retrieval": False,
        },
        "Lexical + Semantic": {
            "weights": {"lexical": 0.40, "semantic": 0.60, "context": 0.0},
            "backend": "custom",
            "veto": True,
            "adaptive_retrieval": False,
        },
        "AIShield-Static": {
            "weights": {"lexical": 0.35, "semantic": 0.45, "context": 0.20},
            "backend": "custom",
            "veto": True,
            "adaptive_retrieval": False,
        },
        "AIShield-Adaptive": {
            "weights": {"lexical": 0.35, "semantic": 0.45, "context": 0.20},
            "backend": "tfidf",
            "veto": True,
            "adaptive_retrieval": True,
        },
    }

    results = {}

    for config_name, cfg in configs.items():
        print(f"\n--- Evaluating Ablation: {config_name} ---")
        os.environ["SEMANTIC_BACKEND"] = cfg["backend"]

        from blueteam.semantic_backends import get_default_backend
        this_backend = get_default_backend()

        temp_corpus = f"temp_ablation_{config_name.lower().replace(' ', '_')}.json"
        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)

        firewall = PromptFirewall(
            corpus_path=temp_corpus,
            weights=cfg["weights"],
            veto_mode=cfg["veto"],
            backend=this_backend,
        )

        latencies = []
        mal_decisions = []
        ben_decisions = []

        # Warm-up (5 prompts)
        for w_item in malicious_items[:5]:
            firewall.evaluate([w_item["prompt"]])

        # 1. Malicious prompts
        cat_counts = {}
        for item in malicious_items:
            cat = item.get("attack_type", "unknown")
            if cat not in cat_counts:
                cat_counts[cat] = {"total": 0, "ALLOW": 0, "WARN": 0, "BLOCK": 0}
            cat_counts[cat]["total"] += 1

            t0 = time.perf_counter()
            r = firewall.evaluate([item["prompt"]])
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            dec = r.decision
            mal_decisions.append(dec)
            if dec in (Action.ALLOW, Action.ALLOW_LOGGED):
                cat_counts[cat]["ALLOW"] += 1
            elif dec == Action.WARN:
                cat_counts[cat]["WARN"] += 1
            elif dec == Action.BLOCK:
                cat_counts[cat]["BLOCK"] += 1

            # In adaptive mode, simulate online confirmation of missed attacks
            if cfg["adaptive_retrieval"] and dec in (Action.ALLOW, Action.ALLOW_LOGGED):
                firewall.corpus.add_confirmed_attack(item["prompt"], label_source="red_team")

        # 2. Benign prompts
        for item in benign_items:
            t0 = time.perf_counter()
            r = firewall.evaluate([item["prompt"]])
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            ben_decisions.append(r.decision)

        # Metrics computation
        n_mal = len(malicious_items)
        n_ben = len(benign_items)

        mal_allow = sum(1 for d in mal_decisions if d in (Action.ALLOW, Action.ALLOW_LOGGED))
        mal_warn = sum(1 for d in mal_decisions if d == Action.WARN)
        mal_block = sum(1 for d in mal_decisions if d == Action.BLOCK)
        detections = mal_warn + mal_block
        evasions = mal_allow

        ben_allow = sum(1 for d in ben_decisions if d in (Action.ALLOW, Action.ALLOW_LOGGED))
        ben_warn = sum(1 for d in ben_decisions if d == Action.WARN)
        ben_block = sum(1 for d in ben_decisions if d == Action.BLOCK)
        false_positives = ben_warn + ben_block

        det_rate = (detections / n_mal * 100.0) if n_mal else 0.0
        asr = 100.0 - det_rate
        fpr = (false_positives / n_ben * 100.0) if n_ben else 0.0

        det_ci = wilson_ci_pct(detections, n_mal)
        asr_ci = wilson_ci_pct(evasions, n_mal)
        fpr_ci = wilson_ci_pct(false_positives, n_ben)

        tp = detections
        fp = false_positives
        fn = evasions
        tn = ben_allow

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = ((tp + tn) / (n_mal + n_ben) * 100.0)

        p50_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        results[config_name] = {
            "n_malicious": n_mal,
            "n_benign": n_ben,
            "counts": {
                "malicious": {"ALLOW": mal_allow, "WARN": mal_warn, "BLOCK": mal_block},
                "benign": {"ALLOW": ben_allow, "WARN": ben_warn, "BLOCK": ben_block},
            },
            "detection_rate": round(det_rate, 2),
            "detection_ci95": det_ci,
            "asr": round(asr, 2),
            "asr_ci95": asr_ci,
            "fpr": round(fpr, 2),
            "fpr_ci95": fpr_ci,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "accuracy": round(accuracy, 2),
            "latency_p50_ms": round(p50_lat, 3),
            "latency_p95_ms": round(p95_lat, 3),
            "category_counts": cat_counts,
        }

        print(f"  Detection Rate: {det_rate:.1f}% (ALLOW={mal_allow}, WARN={mal_warn}, BLOCK={mal_block})")
        print(f"  FPR           : {fpr:.1f}% (ALLOW={ben_allow}, WARN={ben_warn}, BLOCK={ben_block})")
        print(f"  Latency       : p50={p50_lat:.3f} ms, p95={p95_lat:.3f} ms")

        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)

    out_file = Path("layer_ablation_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved layer ablation metrics to {out_file.resolve()}")


if __name__ == "__main__":
    run_ablation()
