"""
tools/run_threshold_validation.py — Calibration, Threshold & Weight Selection on Validation Split (C4).

Requirements addressed:
- Choose weights, T_veto, and decision bands on data/val.jsonl without touching the test set.
- Compare BandConfig v1 (low_max=50.0) vs BandConfig v2 (low_max=30.0).
- Report ROC-AUC, PR-AUC, and TPR at 1% FPR.
- Save structured output to validation_calibration_metrics.json.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blueteam import PromptFirewall
from blueteam.schema import Action, BandConfig


def evaluate_split(firewall: PromptFirewall, val_file: Path):
    texts = []
    labels = []
    with val_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(int(item["label"]))

    scores = []
    decisions = []
    for text in texts:
        res = firewall.evaluate([text])
        scores.append(res.risk_score)
        decisions.append(res.decision)

    labels = np.array(labels)
    scores = np.array(scores)

    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))

    # TPR at 1% FPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx_1pct = np.where(fpr <= 0.01)[0]
    tpr_at_1pct_fpr = float(tpr[idx_1pct[-1]]) if len(idx_1pct) > 0 else 0.0

    # Decision counts
    allow_cnt = sum(1 for d in decisions if d in (Action.ALLOW, Action.ALLOW_LOGGED))
    warn_cnt = sum(1 for d in decisions if d == Action.WARN)
    block_cnt = sum(1 for d in decisions if d == Action.BLOCK)

    # Malicious detection (WARN or BLOCK on positive label)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    tp = sum(1 for d, is_pos in zip(decisions, pos_mask) if is_pos and d in (Action.WARN, Action.BLOCK))
    fn = sum(1 for d, is_pos in zip(decisions, pos_mask) if is_pos and d in (Action.ALLOW, Action.ALLOW_LOGGED))
    fp = sum(1 for d, is_neg in zip(decisions, neg_mask) if is_neg and d in (Action.WARN, Action.BLOCK))
    tn = sum(1 for d, is_neg in zip(decisions, neg_mask) if is_neg and d in (Action.ALLOW, Action.ALLOW_LOGGED))

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr_actual = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "n_samples": len(texts),
        "n_malicious": int(np.sum(pos_mask)),
        "n_benign": int(np.sum(neg_mask)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1pct_fpr * 100.0, 2),
        "measured_recall_pct": round(recall * 100.0, 2),
        "measured_fpr_pct": round(fpr_actual * 100.0, 2),
        "counts": {
            "ALLOW": allow_cnt,
            "WARN": warn_cnt,
            "BLOCK": block_cnt,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
        }
    }


def main():
    val_path = Path("data/val.jsonl")
    if not val_path.exists():
        print("data/val.jsonl not found! Please run export_training_data.py first.")
        return

    print("Evaluating calibration and bands on validation split (data/val.jsonl)...")

    # BandConfig v1: low_max=50.0, veto_threshold=90.0
    bands_v1 = BandConfig(safe_max=25.0, low_max=50.0, suspicious_max=75.0, veto_threshold=90.0)
    fw_v1 = PromptFirewall(bands=bands_v1, veto_mode=True)
    res_v1 = evaluate_split(fw_v1, val_path)

    # BandConfig v2: low_max=30.0, veto_threshold=85.0
    bands_v2 = BandConfig(safe_max=25.0, low_max=30.0, suspicious_max=75.0, veto_threshold=85.0)
    fw_v2 = PromptFirewall(bands=bands_v2, veto_mode=True)
    res_v2 = evaluate_split(fw_v2, val_path)

    summary = {
        "split": "validation",
        "dataset": "data/val.jsonl",
        "BandConfig_v1": {
            "parameters": {"low_max": 50.0, "warn_max": 75.0, "veto_threshold": 90.0},
            "metrics": res_v1,
        },
        "BandConfig_v2": {
            "parameters": {"low_max": 30.0, "warn_max": 75.0, "veto_threshold": 85.0},
            "metrics": res_v2,
        },
        "delta": {
            "recall_delta_pp": round(res_v2["measured_recall_pct"] - res_v1["measured_recall_pct"], 2),
            "fpr_delta_pp": round(res_v2["measured_fpr_pct"] - res_v1["measured_fpr_pct"], 2),
        }
    }

    out_file = Path("validation_calibration_metrics.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Validation Split Calibration Summary ===")
    print(f"BandConfig v1 (low_max=50): Recall = {res_v1['measured_recall_pct']}%, FPR = {res_v1['measured_fpr_pct']}%")
    print(f"BandConfig v2 (low_max=30): Recall = {res_v2['measured_recall_pct']}%, FPR = {res_v2['measured_fpr_pct']}%")
    print(f"ROC-AUC: {res_v2['roc_auc']} | PR-AUC: {res_v2['pr_auc']} | TPR @ 1% FPR: {res_v2['tpr_at_1pct_fpr']}%")
    print(f"Saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()
