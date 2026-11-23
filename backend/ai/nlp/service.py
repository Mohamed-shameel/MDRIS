"""
NLP Module — Service Layer
============================
Entry point used by the routers: delegates to predict and returns a typed NLPResponse.
"""

import logging

from .predict import predict
from .schemas import NLPResponse

logger = logging.getLogger("mdris.nlp.service")


def analyze_text(text: str, source: str) -> NLPResponse:
    """
    Classify incident text into Low / Medium / High urgency.

    Raises:
        ValueError: unknown source (-> 422).
        registry.ModelUnavailable: model not loaded (-> 503).
    """
    result = predict(text, source)
    logger.info(
        f"NLP analysis | urgency={result['urgency']} conf={result['confidence']:.2f} "
        f"S_n={result['nlp_score']:.3f} time={result['inference_ms']:.1f}ms"
    )
    return NLPResponse(**result)
