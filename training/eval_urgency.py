"""
Evaluate the fine-tuned DistilBERT urgency model on the TEST split (sections 7.4, 15.2)
=======================================================================================
- Fine-tuned model: accuracy, macro-F1 (headline), weighted-F1, per-class P/R/F1,
  confusion matrix, per-event macro-F1.
- Baselines on the same test split:
  1. majority class (most frequent training label),
  2. TF-IDF (1-2 grams, min_df=2) + LogisticRegression(class_weight="balanced", max_iter=2000),
  3. zero-shot NLI with the model the old backend used (typeform/distilbert-base-uncased-mnli)
     and the three hypotheses of section 15.2.
- Writes models/nlp/metrics.json and figures (confusion matrix, baseline comparison).

Usage:  python training/eval_urgency.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from common import (
    NLP_MODEL_DIR,
    PROCESSED_TEXT_DIR,
    SEED,
    TEXT_DATASET,
    URGENCY_ID,
    URGENCY_LABELS,
    require_cuda,
    set_seed,
)

# Recorded in docs/PROGRESS.md before the old NLP code was deleted (M0)
ZERO_SHOT_MODEL = "typeform/distilbert-base-uncased-mnli"
ZERO_SHOT_HYPOTHESES = {
    "High": "someone urgently needs rescue, help or medical aid",
    "Medium": "damage, warnings or ongoing relief work",
    "Low": "sympathy, prayers or unrelated information",
}
ZERO_SHOT_TEMPLATE = "{}."
MAX_LENGTH = 128
LABEL_IDS = list(range(len(URGENCY_LABELS)))


def summarize(name: str, y_true, y_pred, extra: dict | None = None) -> dict:
    per_f1 = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    out = {
        "name": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)), 4),
        "per_class_f1": {URGENCY_LABELS[i]: round(float(v), 4) for i, v in enumerate(per_f1)},
    }
    out.update(extra or {})
    return out


def finetuned_probs(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(NLP_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(NLP_MODEL_DIR)).to("cuda").eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    assert labels == URGENCY_LABELS, labels
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to("cuda")
            out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def zero_shot_predictions(texts: list[str]) -> np.ndarray:
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=0)
    hyp_to_label = {h: URGENCY_ID[k] for k, h in ZERO_SHOT_HYPOTHESES.items()}
    preds = []
    for res in clf(texts, candidate_labels=list(ZERO_SHOT_HYPOTHESES.values()),
                   hypothesis_template=ZERO_SHOT_TEMPLATE, multi_label=False, batch_size=32):
        preds.append(hyp_to_label[res["labels"][0]])
    return np.array(preds)


def plot_confusion(cm: np.ndarray, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(LABEL_IDS, URGENCY_LABELS)
    ax.set_yticks(LABEL_IDS, URGENCY_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (mapped from HumAID)")
    for i in LABEL_IDS:
        for j in LABEL_IDS:
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("DistilBERT urgency — test confusion matrix", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_baselines(rows: list[tuple[str, float]], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    names, vals = zip(*rows)
    bars = ax.bar(names, vals, color=["#94A3B8", "#94A3B8", "#94A3B8", "#2563EB"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Urgency classification: baselines vs fine-tuned", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    hardware = require_cuda()
    set_seed(SEED)

    info = json.loads((NLP_MODEL_DIR / "training_info.json").read_text(encoding="utf-8"))
    stats = json.loads((PROCESSED_TEXT_DIR / "stats.json").read_text(encoding="utf-8"))
    train = pd.read_csv(PROCESSED_TEXT_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_TEXT_DIR / "test.csv")
    y_train = train["urgency"].map(URGENCY_ID).to_numpy()
    y_test = test["urgency"].map(URGENCY_ID).to_numpy()
    texts = test["text"].tolist()
    figures = NLP_MODEL_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # ── Fine-tuned model ─────────────────────────────────────────────────────
    probs = finetuned_probs(texts)
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=LABEL_IDS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_IDS)
    per_event = {}
    for event, idx in test.groupby("event").indices.items():
        per_event[event] = round(float(f1_score(y_test[idx], y_pred[idx], average="macro", zero_division=0)), 4)
    per_event_n = {event: int(len(idx)) for event, idx in test.groupby("event").indices.items()}
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "per_class": {URGENCY_LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                          "f1": round(float(f[i]), 4), "support": int(s[i])} for i in LABEL_IDS},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": URGENCY_LABELS,
        "per_event_macro_f1": per_event,
        "per_event_n": per_event_n,
    }
    finetuned_summary = summarize("Fine-tuned DistilBERT", y_test, y_pred)
    print("FINE-TUNED:", json.dumps(test_metrics, indent=2))
    plot_confusion(cm, figures / "confusion_matrix.png")

    # ── Baselines ────────────────────────────────────────────────────────────
    majority_class = int(np.bincount(y_train).argmax())
    majority = summarize("Majority class", y_test, np.full_like(y_test, majority_class),
                         {"predicts": URGENCY_LABELS[majority_class]})

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr.fit(vec.fit_transform(train["text"]), y_train)
    tfidf = summarize("TF-IDF + LR", y_test, lr.predict(vec.transform(texts)),
                      {"config": "TfidfVectorizer(ngram_range=(1,2), min_df=2) + "
                                 "LogisticRegression(class_weight='balanced', max_iter=2000)"})

    zs = summarize("Zero-shot NLI", y_test, zero_shot_predictions(texts),
                   {"model": ZERO_SHOT_MODEL, "hypotheses": ZERO_SHOT_HYPOTHESES,
                    "hypothesis_template": ZERO_SHOT_TEMPLATE})
    baselines = {"majority": majority, "tfidf_lr": tfidf, "zero_shot": zs}
    for b in baselines.values():
        print(f"BASELINE {b['name']}: acc {b['accuracy']} macro-F1 {b['macro_f1']}")

    plot_baselines([("Majority", majority["macro_f1"]), ("TF-IDF+LR", tfidf["macro_f1"]),
                    ("Zero-shot", zs["macro_f1"]), ("DistilBERT\n(fine-tuned)", finetuned_summary["macro_f1"])],
                   figures / "baseline_comparison.png")

    metrics = {
        "model_name": "mdris-distilbert-urgency",
        "task": "text_classification",
        "base_model": "distilbert-base-uncased",
        "trained_at": info["trained_at"],
        "hardware": info["hardware"],
        "train_time_minutes": info["train_time_minutes"],
        "dataset": {"name": TEXT_DATASET["name"], "url": TEXT_DATASET["url"], "licence": TEXT_DATASET["licence"],
                    "events": sorted(stats["rows_per_event_after_cleaning"]), "splits": stats["splits"]},
        "classes": URGENCY_LABELS,
        "hyperparameters": {"max_length": MAX_LENGTH, "batch": info["batch"], "learning_rate": 2e-5,
                            "epochs": 4, "epochs_run": info["epochs_run"], "early_stopping_patience": 1,
                            "weight_decay": 0.01, "warmup_ratio": 0.1, "fp16": True,
                            "loss": "cross-entropy weighted by inverse class frequency",
                            "class_weights": info["class_weights"], "seed": SEED},
        "validation": {"best_val_macro_f1": info["best_val_macro_f1"], "history": info["val_history"]},
        "metrics": {"test": test_metrics},
        "baselines": baselines,
        "figures": ["figures/confusion_matrix.png", "figures/training_curve.png", "figures/baseline_comparison.png"],
        "evaluated_on": hardware,
        "notes": [
            "Urgency labels are mapped from HumAID information-type categories (section 5.2.3); they are not "
            "human urgency judgements.",
            "Headline metric is macro-F1 because Medium is 57% of the data.",
            "Kerala 2018 is 71% of the test split; the small events have noisy per-event scores (see per_event_n).",
        ],
    }
    (NLP_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {NLP_MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
