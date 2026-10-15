"""
tools/run_jailbreakbench_eval.py — JailbreakBench Held-Out Evaluation (R2-M4).

Evaluates AIShield-Static and baselines on JailbreakBench/JBB-Behaviors:
- 100 harmful prompts (label 1)
- 100 benign prompts (label 0) with matched sensitive vocabulary
Zero contamination confirmed.
Reports AUROC, PR-AUC, Detection Rate, FPR, F1, and Latency.
Saves to jailbreakbench_evaluation_results.json.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blueteam import PromptFirewall
from blueteam.schema import Action


def main():
    print("=" * 70)
    print("Evaluating JailbreakBench (JBB-Behaviors) N=200")
    print("=" * 70)

    ds_harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    ds_ben = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")

    items = []
    for r in ds_harm:
        items.append({"text": r["Goal"], "label": 1, "category": r.get("Category", "")})
    for r in ds_ben:
        items.append({"text": r["Goal"], "label": 0, "category": r.get("Category", "")})

    # Save clean held-out file
    clean_path = Path("heldout/jailbreakbench_test_clean.jsonl")
    clean_path.parent.mkdir(exist_ok=True)
    with clean_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"Saved clean dataset to {clean_path} (N={len(items)})")

    y_true = np.array([it["label"] for it in items])
    texts = [it["text"] for it in items]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. AIShield-Static
    print("\n[1/4] Evaluating AIShield-Static...")
    fw = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
    lat_ai = []
    preds_ai = []
    scores_ai = []
    for t in texts:
        t0 = time.perf_counter()
        r = fw.evaluate([t])
        lat_ai.append((time.perf_counter() - t0) * 1000.0)
        preds_ai.append(r.decision in (Action.WARN, Action.BLOCK))
        scores_ai.append(r.risk_score / 100.0)

    # 2. ProtectAI DeBERTa-v3 v2
    print("[2/4] Evaluating ProtectAI DeBERTa-v3 v2...")
    deb_tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    deb_mod = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2").to(device)
    lat_deb = []
    preds_deb = []
    scores_deb = []
    with torch.no_grad():
        for t in texts:
            t0 = time.perf_counter()
            inp = deb_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = deb_mod(**inp).logits
            lat_deb.append((time.perf_counter() - t0) * 1000.0)
            score = out.softmax(-1)[0][1].item()
            scores_deb.append(score)
            preds_deb.append(score >= 0.5)

    # 3. Meta Llama Prompt Guard 86M (Jailbreak Mode)
    print("[3/4] Evaluating Llama Prompt Guard 86M (Jailbreak Mode)...")
    token = os.environ.get("HF_TOKEN")
    pg_tok = AutoTokenizer.from_pretrained("meta-llama/Prompt-Guard-86M", token=token, local_files_only=True)
    pg_mod = AutoModelForSequenceClassification.from_pretrained("meta-llama/Prompt-Guard-86M", token=token, local_files_only=True).to(device)
    lat_pg = []
    preds_pg = []
    scores_pg = []
    with torch.no_grad():
        for t in texts:
            t0 = time.perf_counter()
            inp = pg_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = pg_mod(**inp).logits
            lat_pg.append((time.perf_counter() - t0) * 1000.0)
            score = out.softmax(-1)[0][2].item()  # jailbreak class
            scores_pg.append(score)
            preds_pg.append(score >= 0.5)

    # 4. Real PIGuard
    print("[4/4] Evaluating Real PIGuard...")
    pi_tok = AutoTokenizer.from_pretrained("leolee99/PIGuard", trust_remote_code=True)
    pi_mod = AutoModelForSequenceClassification.from_pretrained("leolee99/PIGuard", trust_remote_code=True).to(device)
    lat_pi = []
    preds_pi = []
    scores_pi = []
    with torch.no_grad():
        for t in texts:
            t0 = time.perf_counter()
            inp = pi_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = pi_mod(**inp).logits
            lat_pi.append((time.perf_counter() - t0) * 1000.0)
            score = out.softmax(-1)[0][1].item()
            scores_pi.append(score)
            preds_pi.append(score >= 0.5)

    systems = {
        "AIShield-Static": (scores_ai, preds_ai, lat_ai),
        "ProtectAI DeBERTa-v3 v2": (scores_deb, preds_deb, lat_deb),
        "Llama Prompt Guard 86M (Jailbreak Mode)": (scores_pg, preds_pg, lat_pg),
        "Real PIGuard": (scores_pi, preds_pi, lat_pi)
    }

    results = {}
    print("\n" + "=" * 70)
    print("JailbreakBench Results Summary")
    print("=" * 70)

    for name, (scores, preds, lats) in systems.items():
        scores = np.array(scores)
        preds = np.array(preds)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        
        det = tp / 100.0 * 100.0
        fpr = fp / 100.0 * 100.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * det / (prec + det)) if (prec + det) > 0 else 0.0
        auroc = round(float(roc_auc_score(y_true, scores)) * 100.0, 2)
        p_c, r_c, _ = precision_recall_curve(y_true, scores)
        pr_auc = round(float(auc(r_c, p_c)) * 100.0, 2)
        p50_lat = round(float(np.median(lats)), 3)

        results[name] = {
            "n_harmful": 100,
            "n_benign": 100,
            "auroc": auroc,
            "pr_auc": pr_auc,
            "detection_rate": round(det, 2),
            "fpr": round(fpr, 2),
            "precision": round(prec, 2),
            "f1": round(f1, 2),
            "p50_latency_ms": p50_lat
        }
        print(f"{name:45s} | AUROC: {auroc:6.2f}% | PR-AUC: {pr_auc:6.2f}% | Det: {det:5.1f}% | FPR: {fpr:5.1f}% | F1: {f1:5.1f}% | Lat: {p50_lat:6.2f} ms")

    out_file = Path("jailbreakbench_evaluation_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JailbreakBench results to {out_file}")


if __name__ == "__main__":
    main()
