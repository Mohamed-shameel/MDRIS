"""
Fine-tune DistilBERT for 3-level flood urgency (docs/MDRIS_CONTEXT.md section 7.3)
===================================================================================
distilbert-base-uncased -> Low / Medium / High on datasets/processed/flood_text.

Fixed settings: max_length 128, batch 16 (8 only after an out-of-memory error),
lr 2e-5, 4 epochs with early stopping on val macro-F1 (patience 1), weight decay
0.01, 10% warmup, fp16, cross-entropy weighted by inverse class frequency,
best model by eval_macro_f1.

Saves model + tokenizer to models/nlp/, the training curve to
models/nlp/figures/training_curve.png and models/nlp/training_info.json.

Usage (run in the background, logging to a file):
    python -u training/train_urgency.py > training/runs/urgency_train.log 2>&1
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import (
    ID_TO_URGENCY,
    NLP_MODEL_DIR,
    PROCESSED_TEXT_DIR,
    RUNS_DIR,
    SEED,
    URGENCY_ID,
    URGENCY_LABELS,
    now_iso,
    require_cuda,
    set_seed,
)

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128


class TextDataset(torch.utils.data.Dataset):
    """Pre-tokenized texts with integer urgency labels."""

    def __init__(self, tokenizer, texts, labels, max_length):
        self.enc = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [int(y) for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse class frequency, normalized so the weights average to 1."""
    counts = np.bincount(labels, minlength=len(URGENCY_LABELS)).astype(float)
    inv = 1.0 / counts
    return inv / inv.sum() * len(URGENCY_LABELS)


def plot_curve(log_history: list[dict], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    evals = [(h["epoch"], h["eval_loss"], h["eval_macro_f1"]) for h in log_history if "eval_loss" in h]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if train:
        ax1.plot(*zip(*train), label="train loss (weighted CE)", color="#2563EB", alpha=0.7)
    if evals:
        ax1.plot([e[0] for e in evals], [e[1] for e in evals], "o-", label="val loss", color="#EA580C")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    if evals:
        ax2.plot([e[0] for e in evals], [e[2] for e in evals], "s--", label="val macro-F1", color="#16A34A")
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0, 1)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="center right", fontsize=8)
    ax1.set_title("DistilBERT urgency fine-tuning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16, choices=[16, 8], help="16, or 8 after an out-of-memory error")
    args = parser.parse_args()

    hardware = require_cuda()
    set_seed(SEED)

    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    train_df = pd.read_csv(PROCESSED_TEXT_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_TEXT_DIR / "val.csv")
    y_train = train_df["urgency"].map(URGENCY_ID).to_numpy()
    y_val = val_df["urgency"].map(URGENCY_ID).to_numpy()
    weights = class_weights(y_train)
    print(f"[{now_iso()}] {hardware} | train={len(train_df)} val={len(val_df)} | class weights "
          f"{dict(zip(URGENCY_LABELS, weights.round(4)))}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(URGENCY_LABELS),
        id2label=ID_TO_URGENCY,
        label2id=URGENCY_ID,
    )
    train_ds = TextDataset(tokenizer, train_df["text"].tolist(), y_train, MAX_LENGTH)
    val_ds = TextDataset(tokenizer, val_df["text"].tolist(), y_val, MAX_LENGTH)
    loss_weights = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits.float(), labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}

    out_dir = RUNS_DIR / "distilbert_urgency"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    NLP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(NLP_MODEL_DIR))
    tokenizer.save_pretrained(str(NLP_MODEL_DIR))
    (NLP_MODEL_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plot_curve(trainer.state.log_history, NLP_MODEL_DIR / "figures" / "training_curve.png")

    evals = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    info = {
        "trained_at": now_iso(),
        "hardware": f"NVIDIA {hardware}" if not hardware.startswith("NVIDIA") else hardware,
        "train_time_minutes": round(minutes, 1),
        "epochs_run": int(round(max(h["epoch"] for h in evals))) if evals else None,
        "best_val_macro_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "val_history": [{"epoch": h["epoch"], "eval_loss": h["eval_loss"], "eval_macro_f1": h["eval_macro_f1"],
                         "eval_accuracy": h["eval_accuracy"]} for h in evals],
        "class_weights": dict(zip(URGENCY_LABELS, [round(float(w), 4) for w in weights])),
        "batch": args.batch,
    }
    (NLP_MODEL_DIR / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[{now_iso()}] Done in {minutes:.1f} min. Best val macro-F1 {trainer.state.best_metric:.4f}. "
          f"Saved to {NLP_MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
