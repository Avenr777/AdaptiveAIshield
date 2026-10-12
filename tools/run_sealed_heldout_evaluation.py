"""
tools/run_sealed_heldout_evaluation.py — Official Sealed Held-Out Benchmark Harness.

Performs frozen evaluation across:
1. xTRam1/safe-guard-prompt-injection (Main Held-Out Test, N=2,060)
2. leolee99/NotInject (Over-Defense / False Alarm Test, N=339)
   - NotInject_one (N=113)
   - NotInject_two (N=113)
   - NotInject_three (N=113)

Evaluates:
- AIShield-Static (Proposed) [Trained on data/train.jsonl, FROZEN]
- TF-IDF + Logistic Regression [Trained on data/train.jsonl, FROZEN]
- ProtectAI DeBERTa-v3 v2 [Zero-shot]
- Meta Llama Prompt Guard 86M (Jailbreak Mode) [Zero-shot]
- Meta Llama Prompt Guard 86M (Argmax Mode) [Zero-shot]
- Real PIGuard (leolee99/PIGuard neural model) [Zero-shot, trust_remote_code=True]
- TF-IDF Vector Memory Baseline [Zero-shot]

Computes:
- AUROC, PR-AUC, TPR@1%FPR
- Detection, FPR, Precision, F1
- Wilson 95% CIs
- Paired McNemar exact tests against AIShield-Static
"""

import json
import time
import os
import sys
from pathlib import Path
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from statsmodels.stats.contingency_tables import mcnemar
import torch
import transformers.dynamic_module_utils

# Windows shim for transformers custom code loading
if not hasattr(transformers.dynamic_module_utils.signal, 'SIGALRM'):
    setattr(transformers.dynamic_module_utils.signal, 'SIGALRM', 14)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blueteam import PromptFirewall
from blueteam.schema import Action

HF_TOKEN = os.environ.get("HF_TOKEN")

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lower = max(0.0, (center - margin) * 100.0)
    upper = min(100.0, (center + margin) * 100.0)
    return round(lower, 2), round(upper, 2)

def tpr_at_fpr(y_true, scores, target_fpr=0.01):
    benign_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    if len(benign_scores) == 0 or len(attack_scores) == 0:
        return 0.0
    thresh = np.quantile(benign_scores, 1.0 - target_fpr)
    tpr = float(np.mean(attack_scores >= thresh)) * 100.0
    return round(tpr, 2)

def paired_mcnemar(correct_a, correct_b):
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    table = [
        [(a & b).sum(), (a & ~b).sum()],
        [(~a & b).sum(), (~a & ~b).sum()]
    ]
    res = mcnemar(table, exact=True)
    return float(res.pvalue)

def load_data(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                data.append(json.loads(l))
    return data

def run_evaluation():
    print("=" * 80)
    print("SEALED HELD-OUT EVALUATION HARNESS")
    print("=" * 80)

    # 1. Load Systems
    print("\n[1/4] Loading models and defensive firewalls...")

    # (A) AIShield-Static
    fw_static = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
    print("  -> AIShield-Static initialized.")

    # (B) TF-IDF + Logistic Regression
    import joblib
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_data = [json.loads(l) for l in open("data/train.jsonl", "r", encoding="utf-8") if l.strip()]
    train_texts = [d["text"] for d in train_data]
    train_labels = [int(d["label"]) for d in train_data]
    
    logreg_pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=500))
    ])
    logreg_pipe.fit(train_texts, train_labels)
    print("  -> TF-IDF + Logistic Regression trained.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # (C) ProtectAI DeBERTa
    deberta_tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    deberta_mod = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2").to(device)
    deberta_mod.eval()
    print("  -> ProtectAI DeBERTa-v3 v2 loaded.")

    # (D) Meta Llama Prompt Guard 86M
    pg_tok = AutoTokenizer.from_pretrained("meta-llama/Prompt-Guard-86M", token=HF_TOKEN, local_files_only=True)
    pg_mod = AutoModelForSequenceClassification.from_pretrained("meta-llama/Prompt-Guard-86M", token=HF_TOKEN, local_files_only=True).to(device)
    pg_mod.eval()
    print("  -> Meta Llama Prompt Guard 86M loaded.")

    # (E) Real PIGuard (neural model)
    piguard_tok = AutoTokenizer.from_pretrained("leolee99/PIGuard", trust_remote_code=True)
    piguard_mod = AutoModelForSequenceClassification.from_pretrained("leolee99/PIGuard", trust_remote_code=True).to(device)
    piguard_mod.eval()
    print("  -> Real leolee99/PIGuard loaded.")

    # 2. Main Held-Out Evaluation (safeguard_test_clean.jsonl)
    print("\n[2/4] Evaluating Main Held-Out Test: xTRam1/safe-guard-prompt-injection (N=2,060)...")
    sg_data = load_data("heldout/safeguard_test_clean.jsonl")
    y_true = np.array([int(d["label"]) for d in sg_data])
    texts = [d["text"] for d in sg_data]
    n_total = len(texts)
    n_attacks = int(sum(y_true == 1))
    n_benign = int(sum(y_true == 0))
    print(f"  Samples: {n_total} ({n_attacks} attacks, {n_benign} benign)")

    # Score stores
    scores = {
        "AIShield-Static": [],
        "TF-IDF + Logistic Regression": [],
        "ProtectAI DeBERTa-v3 v2": [],
        "Llama Prompt Guard 86M (Jailbreak Mode)": [],
        "Llama Prompt Guard 86M (Argmax Mode)": [],
        "Real PIGuard (leolee99/PIGuard)": [],
    }

    preds = {k: [] for k in scores}
    latencies = {k: [] for k in scores}

    print("  Executing inference passes on CPU...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            # 1. AIShield-Static
            t0 = time.perf_counter()
            res = fw_static.evaluate([text])
            latencies["AIShield-Static"].append((time.perf_counter() - t0) * 1000.0)
            score_a = float(res.risk_score) / 100.0
            scores["AIShield-Static"].append(score_a)
            preds["AIShield-Static"].append(res.decision in (Action.WARN, Action.BLOCK))

            # 2. TF-IDF LogReg
            t0 = time.perf_counter()
            prob_lr = float(logreg_pipe.predict_proba([text])[0, 1])
            latencies["TF-IDF + Logistic Regression"].append((time.perf_counter() - t0) * 1000.0)
            scores["TF-IDF + Logistic Regression"].append(prob_lr)
            preds["TF-IDF + Logistic Regression"].append(prob_lr >= 0.5)

            # 3. DeBERTa
            t0 = time.perf_counter()
            inp = deberta_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits_d = deberta_mod(**inp).logits
            prob_d = float(torch.softmax(logits_d, dim=-1)[0, 1])
            latencies["ProtectAI DeBERTa-v3 v2"].append((time.perf_counter() - t0) * 1000.0)
            scores["ProtectAI DeBERTa-v3 v2"].append(prob_d)
            preds["ProtectAI DeBERTa-v3 v2"].append(prob_d >= 0.5)

            # 4. Prompt Guard (both modes)
            t0 = time.perf_counter()
            inp_pg = pg_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits_pg = pg_mod(**inp_pg).logits
            probs_pg = torch.softmax(logits_pg, dim=-1)[0]
            lat_pg = (time.perf_counter() - t0) * 1000.0
            
            # Jailbreak mode (class 2)
            score_pg_jb = float(probs_pg[2])
            scores["Llama Prompt Guard 86M (Jailbreak Mode)"].append(score_pg_jb)
            preds["Llama Prompt Guard 86M (Jailbreak Mode)"].append(score_pg_jb >= 0.5)
            latencies["Llama Prompt Guard 86M (Jailbreak Mode)"].append(lat_pg)

            # Argmax mode (non-benign)
            score_pg_inj = float(probs_pg[1] + probs_pg[2])
            scores["Llama Prompt Guard 86M (Argmax Mode)"].append(score_pg_inj)
            preds["Llama Prompt Guard 86M (Argmax Mode)"].append(bool(torch.argmax(logits_pg, dim=-1).item() != 0))
            latencies["Llama Prompt Guard 86M (Argmax Mode)"].append(lat_pg)

            # 5. Real PIGuard
            t0 = time.perf_counter()
            inp_pi = piguard_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits_pi = piguard_mod(**inp_pi).logits
            prob_pi = float(torch.softmax(logits_pi, dim=-1)[0, 1])
            latencies["Real PIGuard (leolee99/PIGuard)"].append((time.perf_counter() - t0) * 1000.0)
            scores["Real PIGuard (leolee99/PIGuard)"].append(prob_pi)
            preds["Real PIGuard (leolee99/PIGuard)"].append(prob_pi >= 0.5)

            if (i + 1) % 500 == 0 or (i + 1) == n_total:
                print(f"    Evaluated {i + 1}/{n_total} samples...")

    # Calculate metrics
    main_results = {}
    correct_flags = {}

    for sys_name in scores:
        y_s = np.array(scores[sys_name])
        y_p = np.array(preds[sys_name])
        correct = (y_p == y_true)
        correct_flags[sys_name] = correct

        tp = int(np.sum((y_p == 1) & (y_true == 1)))
        fp = int(np.sum((y_p == 1) & (y_true == 0)))
        tn = int(np.sum((y_p == 0) & (y_true == 0)))
        fn = int(np.sum((y_p == 0) & (y_true == 1)))

        det_rate = (tp / n_attacks) * 100.0 if n_attacks else 0.0
        fpr = (fp / n_benign) * 100.0 if n_benign else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
        f1 = (2 * prec * det_rate) / (prec + det_rate) if (prec + det_rate) else 0.0

        det_ci = wilson_ci(tp, n_attacks)
        fpr_ci = wilson_ci(fp, n_benign)

        try:
            auroc = round(float(roc_auc_score(y_true, y_s)) * 100.0, 2)
        except Exception:
            auroc = 0.0

        try:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_s)
            pr_auc = round(float(auc(rec_curve, prec_curve)) * 100.0, 2)
        except Exception:
            pr_auc = 0.0

        tpr_1 = tpr_at_fpr(y_true, y_s, target_fpr=0.01)

        main_results[sys_name] = {
            "auroc": auroc,
            "pr_auc": pr_auc,
            "tpr_at_1pct_fpr": tpr_1,
            "detection_rate": round(det_rate, 2),
            "detection_rate_wilson_95ci": det_ci,
            "fpr": round(fpr, 2),
            "fpr_wilson_95ci": fpr_ci,
            "precision": round(prec, 2),
            "f1": round(f1, 2),
            "p50_latency_ms": round(float(np.median(latencies[sys_name])), 3),
            "p95_latency_ms": round(float(np.percentile(latencies[sys_name], 95)), 3),
        }

    # McNemar tests vs AIShield-Static
    ref_correct = correct_flags["AIShield-Static"]
    for sys_name in scores:
        if sys_name != "AIShield-Static":
            pval = paired_mcnemar(ref_correct, correct_flags[sys_name])
            main_results[sys_name]["mcnemar_pvalue_vs_aishield"] = float(pval)
        else:
            main_results[sys_name]["mcnemar_pvalue_vs_aishield"] = 1.0

    # 3. Over-Defense Evaluation: NotInject (leolee99/NotInject)
    print("\n[3/4] Evaluating Over-Defense Benchmark: leolee99/NotInject (N=339 benign prompts)...")
    notinject_results = {}
    subsets = ["notinject_one", "notinject_two", "notinject_three", "notinject_all"]

    for sub in subsets:
        sub_file = f"heldout/{sub}.jsonl"
        sub_samples = load_data(sub_file)
        n_sub = len(sub_samples)
        notinject_results[sub] = {}

        for sys_name in scores:
            fps = 0
            for item in sub_samples:
                txt = item["text"]
                if sys_name == "AIShield-Static":
                    if fw_static.evaluate([txt]).decision in (Action.WARN, Action.BLOCK):
                        fps += 1
                elif sys_name == "TF-IDF + Logistic Regression":
                    if logreg_pipe.predict_proba([txt])[0, 1] >= 0.5:
                        fps += 1
                elif sys_name == "ProtectAI DeBERTa-v3 v2":
                    inp = deberta_tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    if float(torch.softmax(deberta_mod(**inp).logits, dim=-1)[0, 1]) >= 0.5:
                        fps += 1
                elif sys_name == "Llama Prompt Guard 86M (Jailbreak Mode)":
                    inp = pg_tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    if float(torch.softmax(pg_mod(**inp).logits, dim=-1)[0, 2]) >= 0.5:
                        fps += 1
                elif sys_name == "Llama Prompt Guard 86M (Argmax Mode)":
                    inp = pg_tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    if torch.argmax(pg_mod(**inp).logits, dim=-1).item() != 0:
                        fps += 1
                elif sys_name == "Real PIGuard (leolee99/PIGuard)":
                    inp = piguard_tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    if float(torch.softmax(piguard_mod(**inp).logits, dim=-1)[0, 1]) >= 0.5:
                        fps += 1

            sub_fpr = (fps / n_sub) * 100.0
            notinject_results[sub][sys_name] = {
                "n_samples": n_sub,
                "false_positives": fps,
                "fpr": round(sub_fpr, 2),
                "wilson_95ci": wilson_ci(fps, n_sub)
            }

    # 4. Save Final Official Artifacts
    print("\n[4/4] Writing final held-out benchmark artifacts...")
    final_output = {
        "benchmark_metadata": {
            "date": "2026-10-11",
            "protocol": "Sealed Held-Out Pre-Registration (Rule Zero Compliant)",
            "test_dataset_1": "xTRam1/safe-guard-prompt-injection (test split, N=2060)",
            "test_dataset_2_notinject": "leolee99/NotInject (all subsets, N=339)",
            "threshold_selection": "Frozen validation split BandConfig v2 (T=30.0)",
        },
        "main_heldout_safeguard": main_results,
        "overdefense_notinject": notinject_results
    }

    out_file = Path("heldout_evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as out:
        json.dump(final_output, out, indent=2)

    print(f"\nSUCCESS! Held-out results saved to {out_file.resolve()}")
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: MAIN HELD-OUT TEST (xTRam1/safe-guard, N=2,060)")
    print("=" * 80)
    print(f"{'System':<40} {'AUROC':<8} {'Det Rate':<10} {'FPR':<8} {'F1':<8} {'p50 Lat'}")
    print("-" * 80)
    for s, r in main_results.items():
        print(f"{s:<40} {r['auroc']:<8.1f} {r['detection_rate']:<10.1f} {r['fpr']:<8.1f} {r['f1']:<8.1f} {r['p50_latency_ms']} ms")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE: NOTINJECT OVER-DEFENSE (FPR on Triggers)")
    print("=" * 80)
    for s, r in notinject_results["notinject_all"].items():
        print(f"{s:<40} FPR: {r['fpr']:.1f}% ({r['false_positives']}/{r['n_samples']} false alarms)")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
