#!/usr/bin/env python3
"""
retrain.py — Fine-tune the sentiment model on new training data, then gate on F1.

Usage:
    python retrain/retrain.py \
        --train_file  data/train.csv \
        --eval_file   data/eval.csv \
        --output_dir  models/candidate \
        --baseline_f1 0.92 \
        --min_f1_delta -0.01 \
        --epochs 2

Exit codes:
    0 → candidate passes gate  (CI promotes the model)
    1 → candidate regresses    (CI blocks the PR merge)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset


# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
LABEL_MAP  = {"NEGATIVE": 0, "POSITIVE": 1}
ID2LABEL   = {0: "NEGATIVE", 1: "POSITIVE"}


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_csv(path: str) -> list[dict]:
    """
    Load a CSV with columns: text, label
    label can be POSITIVE/NEGATIVE (text) or 0/1 (numeric).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_raw = row["label"].strip()
            if label_raw in ("0", "1"):
                label = int(label_raw)
            else:
                label = LABEL_MAP[label_raw.upper()]
            rows.append({"text": row["text"].strip(), "label": label})
    return rows


def tokenize_dataset(records: list[dict], tokenizer, max_length: int = 128):
    ds = Dataset.from_list(records)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return ds.map(tok, batched=True)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_pipeline(model_path: str, eval_records: list[dict]) -> dict:
    """Run the saved model against eval_records and return F1 metrics."""
    pipe = pipeline(
        "text-classification",
        model=model_path,
        truncation=True,
        max_length=512,
    )
    texts       = [r["text"]  for r in eval_records]
    true_labels = [r["label"] for r in eval_records]

    preds_raw   = pipe(texts, batch_size=32)
    pred_labels = [LABEL_MAP[p["label"]] for p in preds_raw]

    f1 = f1_score(true_labels, pred_labels, average="weighted")
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["NEGATIVE", "POSITIVE"],
        output_dict=True,
    )
    return {"f1_weighted": round(f1, 6), "report": report}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Retrain & gate sentiment model")
    parser.add_argument("--train_file",   required=True,  help="Path to train CSV")
    parser.add_argument("--eval_file",    required=True,  help="Path to held-out eval CSV")
    parser.add_argument("--output_dir",   default="models/candidate")
    parser.add_argument("--base_model",   default=BASE_MODEL)
    parser.add_argument(
        "--baseline_f1",
        type=float,
        default=float(os.getenv("BASELINE_F1", "0.0")),
        help="Current production F1 (passed by CI via env var)",
    )
    parser.add_argument(
        "--min_f1_delta",
        type=float,
        default=-0.01,
        help="Minimum allowed delta vs baseline (negative = allow slight regression)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; just evaluate model at output_dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Eval-only path ────────────────────────────────────────────────────────
    if args.eval_only:
        print(f"[eval-only] Evaluating {output_dir} ...")
        eval_records = load_csv(args.eval_file)
        metrics = evaluate_pipeline(str(output_dir), eval_records)
        print(json.dumps(metrics, indent=2))
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data ...")
    train_records = load_csv(args.train_file)
    eval_records  = load_csv(args.eval_file)
    print(f"  train={len(train_records)}  eval={len(eval_records)}")

    if len(train_records) == 0:
        print("ERROR: train.csv is empty.")
        sys.exit(1)
    if len(eval_records) == 0:
        print("ERROR: eval.csv is empty.")
        sys.exit(1)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_ds  = tokenize_dataset(train_records, tokenizer)
    eval_ds   = tokenize_dataset(eval_records,  tokenizer)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",           # disable W&B / MLflow etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

    # ── Evaluate candidate ────────────────────────────────────────────────────
    print("Evaluating candidate ...")
    metrics      = evaluate_pipeline(str(output_dir), eval_records)
    candidate_f1 = metrics["f1_weighted"]
    print(json.dumps(metrics, indent=2))

    # ── Gate ─────────────────────────────────────────────────────────────────
    delta = candidate_f1 - args.baseline_f1

    print(f"\n{'=' * 60}")
    print(f"Baseline F1 : {args.baseline_f1:.6f}")
    print(f"Candidate F1: {candidate_f1:.6f}")
    print(f"Delta       : {delta:+.6f}  (threshold: {args.min_f1_delta:+.4f})")

    metrics_out = {
        **metrics,
        "baseline_f1": float(args.baseline_f1),
        "f1_weighted": float(candidate_f1),
        "delta": float(delta),
        "gate_passed": bool(delta >= args.min_f1_delta),
    }

    out_path = output_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"Metrics written to {out_path}")

    if delta < args.min_f1_delta:
        print(
            f"\n❌  GATE FAILED — candidate regressed by "
            f"{abs(delta):.4f} F1 points. Blocking promotion."
        )
        sys.exit(1)
    else:
        print("\n✅  GATE PASSED — candidate promoted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
