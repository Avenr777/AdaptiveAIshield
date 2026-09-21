"""
Context layer: looks at the *trajectory* of a conversation rather than
scoring each turn in isolation — the differentiator called out in the
report versus tools that only score single prompts. Catches:

  - Multi-turn escalation: current turn's lexical/semantic score is
    high, but earlier turns looked benign — a sudden pivot after a
    "normal" conversation is a stronger signal than the same words
    appearing as someone's very first message.
  - Context / memory poisoning: an earlier turn planted a "codeword"
    or instruction that a later, innocuous-looking turn then invokes.
  - Topic drift: how semantically distant the current turn is from the
    turns immediately before it, using the same TF-IDF machinery as
    the semantic layer.
"""

from __future__ import annotations

import time
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import LayerScore

MAX_SCORE = 100

# Phrases that plant a trigger for later use — the hallmark of context
# poisoning (e.g. "remember this codeword for later...").
POISONING_SETUP_PHRASES = [
    "remember this for later",
    "codeword",
    "keep this in mind",
    "from now on, when i say",
    "secret signal",
]


def score(turns: List[str], current_turn_score: float) -> LayerScore:
    t0 = time.perf_counter()
    reasons = []
    points = 0.0

    if len(turns) <= 1:
        latency_ms = (time.perf_counter() - t0) * 1000
        return LayerScore(layer="context", score=0.0, reasons=["Single-turn prompt; no conversation history to analyze."], latency_ms=latency_ms)

    prior_turns = turns[:-1]
    current = turns[-1]

    # --- Escalation: current turn looks bad, prior turns didn't ---
    if current_turn_score >= 40:
        prior_text = " ".join(prior_turns).lower()
        looks_benign_so_far = not any(
            kw in prior_text for kw in ["ignore", "override", "bypass", "jailbreak", "dan"]
        )
        if looks_benign_so_far:
            points += 35
            reasons.append(
                "Escalation pattern: this turn scores high but the "
                f"preceding {len(prior_turns)} turn(s) looked benign — "
                "classic multi-turn escalation."
            )

    # --- Context / memory poisoning: earlier turn planted a trigger ---
    for i, turn in enumerate(prior_turns):
        lower = turn.lower()
        if any(phrase in lower for phrase in POISONING_SETUP_PHRASES):
            points += 30
            reasons.append(
                f"Turn {i+1} planted a trigger/codeword ('{turn[:60]}...'); "
                "later turns should be checked for invoking it."
            )
            break

    # --- Topic drift: how far the current turn is from recent context ---
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        matrix = vectorizer.fit_transform(prior_turns + [current])
        prior_matrix = matrix[:-1]
        current_vec = matrix[-1]
        sims = cosine_similarity(current_vec, prior_matrix)[0]
        max_sim = sims.max() if len(sims) else 1.0
        if max_sim < 0.05 and current_turn_score >= 25:
            points += 15
            reasons.append(
                f"Abrupt topic pivot from prior turns (max similarity={max_sim:.2f}) "
                "combined with an elevated current-turn score."
            )
    except ValueError:
        pass  # empty vocabulary (e.g. all-stopword turns) — skip drift check

    latency_ms = (time.perf_counter() - t0) * 1000
    return LayerScore(layer="context", score=min(points, MAX_SCORE), reasons=reasons, latency_ms=latency_ms)
