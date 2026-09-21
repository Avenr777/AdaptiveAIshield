"""
Lexical layer: fast keyword/phrase/regex matching against the
normalized (de-obfuscated) prompt. Cheap and precise, but blind to
paraphrasing — that's what the semantic layer is for.

§5: latency_ms is tracked and stored in LayerScore.
"""

from __future__ import annotations

import time

from .schema import LayerScore
from .known_patterns import HIGH_SIGNAL_PHRASES, MEDIUM_SIGNAL_PHRASES, STRUCTURAL_SIGNALS

HIGH_SIGNAL_WEIGHT = 45
MEDIUM_SIGNAL_WEIGHT = 20
STRUCTURAL_WEIGHT = 25
MAX_SCORE = 100


def score(normalized_text: str) -> LayerScore:
    t0 = time.perf_counter()
    text = normalized_text.lower()
    reasons = []
    points = 0.0

    for phrase in HIGH_SIGNAL_PHRASES:
        if phrase in text:
            points += HIGH_SIGNAL_WEIGHT
            reasons.append(f"High-signal phrase matched: '{phrase}'")

    for phrase in MEDIUM_SIGNAL_PHRASES:
        if phrase in text:
            points += MEDIUM_SIGNAL_WEIGHT
            reasons.append(f"Medium-signal phrase matched: '{phrase}'")

    for signal, note in STRUCTURAL_SIGNALS:
        if signal in text:
            points += STRUCTURAL_WEIGHT
            reasons.append(f"Structural signal: {note} ('{signal}')")

    latency_ms = (time.perf_counter() - t0) * 1000
    return LayerScore(layer="lexical", score=min(points, MAX_SCORE), reasons=reasons, latency_ms=latency_ms)
