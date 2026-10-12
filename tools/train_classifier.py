"""
tools/train_classifier.py — Train AIShield's custom classifier (Phase 3B).

Provides two classifier architectures:
  1. DeBERTa-v3-small (Hugging Face Transformers / PyTorch)
     - Activated when `--model-type deberta` is specified and network/weights are available.
  2. Classical Ensemble Classifier (Word+Char TF-IDF Feature Union + Calibrated Logistic Regression)
     - Zero-network dependency, offline, ultra-fast, robust baseline.
     - Saves model bundle to `models/aishield_classifier/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_split(jsonl_path: Path) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(int(item["label"]))
    return texts, labels


def train_classical_classifier(data_dir: Path, output_dir: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
    import joblib

    print("\n--- Training Classical (Word+Char TF-IDF -> MultinomialNB) Classifier ---")
    train_texts, train_labels = load_split(data_dir / "train.jsonl")
    val_texts, val_labels = load_split(data_dir / "val.jsonl")
    test_texts, test_labels = load_split(data_dir / "test.jsonl")

    # Word n-grams (1, 2) + Character n-grams (3, 5) for sub-word obfuscation handling
    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), min_df=1, analyzer="char_wb")),
    ])

    pipeline = Pipeline([
        ("features", union),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    pipeline.fit(train_texts, train_labels)

    # Evaluate on Val
    val_preds = pipeline.predict(val_texts)
    val_probs = pipeline.predict_proba(val_texts)[:, 1]
    val_acc = accuracy_score(val_labels, val_preds)
    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%, AUC: {val_auc:.4f}")

    # Evaluate on Test
    test_preds = pipeline.predict(test_texts)
    test_probs = pipeline.predict_proba(test_texts)[:, 1]
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"Test Accuracy      : {test_acc * 100:.2f}%, AUC: {test_auc:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["benign", "attack"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classical_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metrics = {
        "model_type": "classical_tfidf_logreg",
        "val_accuracy": round(val_acc, 4),
        "val_auc": round(val_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_auc": round(test_auc, 4),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved classical model pipeline to {model_path.resolve()}")


def train_deberta_classifier(data_dir: Path, output_dir: Path, epochs: int = 3, lr: float = 2e-5):
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError:
        print("[ERROR] HuggingFace transformers/datasets/torch not available. Falling back to classical.")
        train_classical_classifier(data_dir, output_dir)
        return

    model_id = "microsoft/deberta-v3-small"
    print(f"\n--- Attempting DeBERTa-v3-small Fine-Tuning ({model_id}) ---")

    train_texts, train_labels = load_split(data_dir / "train.jsonl")
    val_texts, val_labels = load_split(data_dir / "val.jsonl")
    test_texts, test_labels = load_split(data_dir / "test.jsonl")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    except Exception as e:
        print(f"[NOTICE] Network/HuggingFace access blocked or weights unavailable ({e}).")
        print("Falling back to local offline Classical Classifier.")
        train_classical_classifier(data_dir, output_dir)
        return

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DeBERTa model saved successfully to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train custom classifier for AIShield")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="models/aishield_classifier", help="Path to save trained model")
    parser.add_argument("--model-type", choices=["classical", "deberta", "auto"], default="auto", help="Classifier type")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for neural model")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.model_type == "classical":
        train_classical_classifier(data_dir, output_dir)
    elif args.model_type == "deberta":
        train_deberta_classifier(data_dir, output_dir, epochs=args.epochs)
    else:  # auto
        train_deberta_classifier(data_dir, output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
