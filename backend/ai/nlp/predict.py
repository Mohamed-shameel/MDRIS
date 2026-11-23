"""
NLP Module — Core Inference
==============================
Runs the fine-tuned DistilBERT urgency classifier (Low / Medium / High) and
builds the NLP response of docs/MDRIS_CONTEXT.md section 9.4.

The NLP score is computed by ai.fusion.fusion_engine.text_score, so the API,
the incident generator and the fusion rule all use the same formula.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from ai import registry
from ai.fusion.constants import DEFAULT_TEXT_SOURCE, TEXT_SOURCE_RELIABILITY, URGENCY_LABELS
from ai.fusion.fusion_engine import text_score

logger = logging.getLogger("mdris.nlp.predict")

MAX_LENGTH = 128  # same as training (section 7.3)

# transformers must not import TensorFlow if it happens to be installed
os.environ.setdefault("USE_TF", "0")


def load_urgency_model(model_dir: Path, device: str):
    """Load the fine-tuned model + tokenizer saved by training/train_urgency.py."""
    if not (Path(model_dir) / "config.json").is_file():
        raise FileNotFoundError(f"no fine-tuned model found in {model_dir} (config.json missing)")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if labels != URGENCY_LABELS:
        raise ValueError(f"model labels {labels} do not match urgency labels {URGENCY_LABELS}")
    model.to(device).eval()
    return model, tokenizer


def validate_source(source: str) -> str:
    if source not in TEXT_SOURCE_RELIABILITY:
        raise ValueError(f"Unknown text source '{source}'. Expected one of {list(TEXT_SOURCE_RELIABILITY)}")
    return source


def classify(text: str) -> Tuple[Dict[str, float], float]:
    """
    Urgency probabilities for one text.

    Raises:
        registry.ModelUnavailable: if the NLP model is not loaded.
    """
    import torch

    state = registry.require_nlp()
    start = time.perf_counter()
    enc = state.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(state.device)
    with torch.no_grad():
        probs = torch.softmax(state.model(**enc).logits[0], dim=-1).tolist()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {label: round(p, 4) for label, p in zip(URGENCY_LABELS, probs)}, elapsed_ms


def predict(text: str, source: str = DEFAULT_TEXT_SOURCE) -> Dict[str, Any]:
    """Full NLP prediction for one text (section 9.4 response dict)."""
    validate_source(source)
    probabilities, elapsed_ms = classify(text.strip())
    s_n, conf_n = text_score(probabilities)
    urgency = max(URGENCY_LABELS, key=lambda k: probabilities[k])
    return {
        "data_origin": "model",
        "model": registry.model_info(registry.NLP),
        "source": source,
        "urgency": urgency,
        "probabilities": probabilities,
        "confidence": round(conf_n, 4),
        "nlp_score": round(s_n, 4),
        "inference_ms": round(elapsed_ms, 1),
    }
