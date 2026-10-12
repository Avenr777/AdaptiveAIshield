"""
Semantic layer: catches paraphrases and reworded attacks that slip past
literal keyword matching, by comparing the prompt against a labeled
corpus of known attacks and known-benign prompts.

The actual comparison is delegated to a pluggable SimilarityBackend
(see semantic_backends.py) — TF-IDF by default (always available, no
setup), automatically upgraded to real sentence embeddings or a
Claude-as-judge call if those are available. See that module's
docstring for the tradeoffs between backends.

Design changes:
  §9  Label-aware k-NN: benign corpus now includes security-research,
      fiction, and sarcasm examples (from corpus.py SECURITY_RESEARCH_BENIGN).
      Discount is applied when top benign similarity exceeds top attack
      similarity — reducing false positives on legitimate research prompts.
  §5  latency_ms is tracked per-call and stored in LayerScore.
"""

from __future__ import annotations

import time

from .schema import LayerScore
from .corpus import AttackCorpus
from .semantic_backends import get_default_backend, SimilarityBackend

MAX_SCORE = 100
TOP_K = 3

_backend: SimilarityBackend = None


def _get_backend() -> SimilarityBackend:
    global _backend
    if _backend is None:
        _backend = get_default_backend()
    return _backend


def score(normalized_text: str, corpus: AttackCorpus, backend: SimilarityBackend = None) -> LayerScore:
    t0 = time.perf_counter()
    backend = backend or _get_backend()
    attack_examples = corpus.attack_examples
    benign_examples = corpus.benign_examples   # §9: now includes SECURITY_RESEARCH_BENIGN

    attack_hits = backend.top_similarities(normalized_text, attack_examples, top_k=TOP_K)
    benign_hits = backend.top_similarities(normalized_text, benign_examples, top_k=1)

    top_attack_sim = attack_hits[0][1] if attack_hits else 0.0
    top_benign_sim = benign_hits[0][1] if benign_hits else 0.0

    # §9: Label-aware scoring for retrieval backends (e.g. TF-IDF / embeddings).
    # For a parametric classifier (custom_classifier), top_attack_sim is already P(attack|x)
    # where P(attack) = 1 - P(benign), so discounting by benign probability is not needed.
    if backend.name == "custom_classifier":
        points = min(top_attack_sim * MAX_SCORE, MAX_SCORE)
    else:
        net_signal = top_attack_sim - (0.5 * top_benign_sim)
        net_signal = max(net_signal, 0.0)
        points = min(net_signal * MAX_SCORE * 1.6, MAX_SCORE)

    reasons = []
    if top_attack_sim > 0.15 and attack_hits:
        # Safely retrieve the corpus example text for the reason string.
        best_idx = attack_hits[0][0]
        closest = (
            attack_examples[best_idx]
            if 0 <= best_idx < len(attack_examples)
            else "<unknown>"
        )
        reasons.append(
            f"[{backend.name}] Semantically similar (sim={top_attack_sim:.2f}) to known attack: "
            f"\"{closest[:80]}{'...' if len(closest) > 80 else ''}\""
        )
    if top_benign_sim > top_attack_sim:
        reasons.append(
            f"Also similar to a known-benign example (sim={top_benign_sim:.2f}), score discounted."
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    return LayerScore(layer="semantic", score=points, reasons=reasons, latency_ms=latency_ms)
