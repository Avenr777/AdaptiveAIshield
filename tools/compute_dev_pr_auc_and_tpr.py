"""
tools/compute_dev_pr_auc_and_tpr.py — Compute PR-AUC and TPR@1%FPR for Table III (R2-M2).

Computes:
  - AUROC
  - PR-AUC
  - TPR at 1% FPR
for all systems on the 500 Kaggle development dataset prompts,
and updates strong_baselines_benchmark.json with these fields.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blueteam import PromptFirewall
from blueteam.known_patterns import KNOWN_ATTACK_EXAMPLES


def tpr_at_fpr(y_true, scores, target_fpr=0.01):
    benign_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    if len(benign_scores) == 0 or len(attack_scores) == 0:
        return 0.0
    thresh = np.quantile(benign_scores, 1.0 - target_fpr)
    tpr = float(np.mean(attack_scores >= thresh)) * 100.0
    return round(tpr, 2)


def main():
    print("=" * 70)
    print("Computing Table III (Dev Kaggle 500) PR-AUC & TPR@1%FPR")
    print("=" * 70)

    cache_path = Path(r"C:\Users\Shreyas\.cache\kagglehub\datasets\cyberprince\prompt-injection-and-benign-prompt-dataset\versions\1\Prompt_INJECTION_And_Benign_DATASET.jsonl")
    data = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    texts = [d["prompt"] for d in data]
    y_true = np.array([1 if d["label"] == "malicious" else 0 for d in data])
    n_samples = len(texts)
    print(f"Loaded {n_samples} Kaggle prompts ({np.sum(y_true == 1)} malicious, {np.sum(y_true == 0)} benign).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # 1. TF-IDF + Logistic Regression
    print("\n[1/6] TF-IDF + Logistic Regression...")
    train_data = [json.loads(l) for l in open("data/train.jsonl", "r", encoding="utf-8") if l.strip()]
    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=500))
    ])
    pipe.fit([d["text"] for d in train_data], [int(d["label"]) for d in train_data])
    scores_tfidf = pipe.predict_proba(texts)[:, 1]

    # 2. Rebuff / Vigil Style Embedding Memory
    print("\n[2/6] Rebuff / Vigil Memory...")
    corpus_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer="char", min_df=1)
    corpus_mat = corpus_vec.fit_transform(KNOWN_ATTACK_EXAMPLES)
    q_mat = corpus_vec.transform(texts)
    scores_rebuff = np.max(cosine_similarity(q_mat, corpus_mat), axis=1)

    # 3. AIShield-Static
    print("\n[3/6] AIShield-Static...")
    fw = PromptFirewall(weights={"lexical": 0.35, "semantic": 0.45, "context": 0.20}, veto_mode=True)
    scores_aishield = []
    for t in texts:
        r = fw.evaluate([t])
        scores_aishield.append(r.risk_score / 100.0)
    scores_aishield = np.array(scores_aishield)

    # 4. PIGuard (signature & boundary)
    print("\n[4/6] PIGuard Baseline...")
    piguard_tok = AutoTokenizer.from_pretrained("leolee99/PIGuard", trust_remote_code=True)
    piguard_mod = AutoModelForSequenceClassification.from_pretrained("leolee99/PIGuard", trust_remote_code=True).to(device)
    scores_piguard = []
    with torch.no_grad():
        for t in texts:
            inp = piguard_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = piguard_mod(**inp).logits
            scores_piguard.append(out.softmax(-1)[0][1].item())
    scores_piguard = np.array(scores_piguard)

    # 5. ProtectAI DeBERTa-v3 v2
    print("\n[5/6] ProtectAI DeBERTa-v3 v2...")
    deb_tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    deb_mod = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2").to(device)
    scores_deberta = []
    with torch.no_grad():
        for t in texts:
            inp = deb_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = deb_mod(**inp).logits
            scores_deberta.append(out.softmax(-1)[0][1].item())
    scores_deberta = np.array(scores_deberta)

    # 6. Meta Llama Prompt Guard 86M
    print("\n[6/6] Meta Llama Prompt Guard 86M...")
    token = os.environ.get("HF_TOKEN")
    pg_tok = AutoTokenizer.from_pretrained("meta-llama/Prompt-Guard-86M", token=token, local_files_only=True)
    pg_mod = AutoModelForSequenceClassification.from_pretrained("meta-llama/Prompt-Guard-86M", token=token, local_files_only=True).to(device)
    scores_pg_jb = []
    scores_pg_am = []
    with torch.no_grad():
        for t in texts:
            inp = pg_tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            logits = pg_mod(**inp).logits
            sm = logits.softmax(-1)[0]
            scores_pg_jb.append(sm[2].item())
            scores_pg_am.append((1.0 - sm[0]).item())
    scores_pg_jb = np.array(scores_pg_jb)
    scores_pg_am = np.array(scores_pg_am)

    system_scores = {
        "TF-IDF + Logistic Regression": scores_tfidf,
        "Rebuff / Vigil Embedding Memory": scores_rebuff,
        "ProtectAI DeBERTa-v3 v2": scores_deberta,
        "AIShield-Static (Proposed)": scores_aishield,
        "PIGuard (Boundary + Pattern Baseline)": scores_piguard,
        "Llama Prompt Guard 86M (Meta - Jailbreak Mode)": scores_pg_jb,
        "Llama Prompt Guard 86M (Meta - Out-of-the-Box Argmax)": scores_pg_am
    }

    metrics_update = {}
    print("\n" + "=" * 70)
    print("RESULTS: Table III Metrics (Dev Kaggle 500)")
    print("=" * 70)
    for name, s in system_scores.items():
        try:
            auroc = round(float(roc_auc_score(y_true, s)) * 100.0, 2)
        except Exception:
            auroc = 0.0
        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, s)
            pr_auc = round(float(auc(r_curve, p_curve)) * 100.0, 2)
        except Exception:
            pr_auc = 0.0
        tpr_1 = tpr_at_fpr(y_true, s, target_fpr=0.01)

        metrics_update[name] = {
            "auroc": auroc,
            "pr_auc": pr_auc,
            "tpr_at_1pct_fpr": tpr_1
        }
        print(f"{name:50s} | AUROC: {auroc:6.2f}% | PR-AUC: {pr_auc:6.2f}% | TPR@1%FPR: {tpr_1:6.2f}%")

    # Update strong_baselines_benchmark.json
    bm_file = Path("strong_baselines_benchmark.json")
    if bm_file.exists():
        with bm_file.open("r", encoding="utf-8") as f:
            bm_data = json.load(f)
        for name, m in metrics_update.items():
            if name in bm_data:
                bm_data[name].update(m)
        with bm_file.open("w", encoding="utf-8") as f:
            json.dump(bm_data, f, indent=2)
        print(f"\nSuccessfully updated {bm_file} with AUROC, PR-AUC, and TPR@1%FPR!")


if __name__ == "__main__":
    main()
