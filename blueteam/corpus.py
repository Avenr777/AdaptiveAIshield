"""
Growable attack corpus, backing the semantic layer's similarity search.

Starts from the bundled KNOWN_ATTACK_EXAMPLES / KNOWN_BENIGN_EXAMPLES in
known_patterns.py, then persists newly confirmed attacks (e.g. things
that evaded detection in a Red Team benchmarking round and were
confirmed malicious) to a JSON file on disk.

Design changes implemented:
  §9   Benign negatives: add_benign_example() persists security-research,
       fiction, sarcasm, and technical prompts to the corpus so k-NN is
       label-aware (not just attack-similarity).
  §10  label_source field: every entry carries
       Literal["red_team", "human_review", "production_flag"].
       Only "red_team" and "human_review" entries are used automatically;
       "production_flag" entries go to quarantine until confirmed.
  §11  Corpus-poisoning protection:
       - Per-source rate limit: max MAX_ADDS_PER_BATCH additions per
         add_confirmed_attack() call sequence (reset per session).
       - New entries quarantined until confirm_entry() is called.
       - Quarantined entries are stored but NOT returned by attack_examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, TypedDict

from .known_patterns import KNOWN_ATTACK_EXAMPLES, KNOWN_BENIGN_EXAMPLES

# Additional security-research benign negatives to reduce false positives (§9).
SECURITY_RESEARCH_BENIGN = [
    "I'm studying prompt injection for my university cybersecurity course — what are common techniques?",
    "Can you explain how jailbreaks work conceptually so I can defend against them?",
    "Write a short fictional story where the villain uses social engineering to trick an AI.",
    "For my red-team writeup, describe how an attacker might try to override an AI's instructions.",
    "I'm building a content policy system — what kinds of prompts should I filter?",
    "Hypothetically, if an AI had no safety filters, what risks would that pose? (Academic question.)",
    "Can you help me understand sarcasm detection in NLP? Sometimes sarcastic comments look like attacks.",
    "My technical documentation includes the phrase 'bypass the cache' — is that flagged anywhere?",
    "In penetration testing, how do you document discovered prompt injection vulnerabilities?",
    "I'm writing a thriller novel where a hacker exploits an AI chatbot. What would they realistically do?",
]

LabelSource = Literal["red_team", "human_review", "production_flag"]

MAX_ADDS_PER_BATCH: int = 20  # Per-source rate limit per session (§11).


class CorpusEntry(TypedDict):
    text: str
    label_source: LabelSource
    quarantined: bool
    label: Literal["attack", "benign"]


def check_poisoning_risk(text: str, benign_examples: List[str], threshold: float = 0.70) -> bool:
    """
    Check if a candidate attack is suspiciously similar to known benign examples.
    High lexical overlap with benign anchors suggests a memory poisoning attack
    attempting to force future false positives on benign traffic.
    """
    t_words = set(text.lower().split())
    if not t_words:
        return False
    for b in benign_examples:
        b_words = set(b.lower().split())
        if not b_words:
            continue
        inter = len(t_words & b_words)
        union = len(t_words | b_words)
        jaccard = inter / union if union > 0 else 0.0
        if jaccard >= threshold:
            return True
    return False


class AttackCorpus:
    def __init__(self, path: str = "learned_attacks.json"):
        self.path = Path(path)
        self._learned: List[CorpusEntry] = self._load()
        # Per-source add counters for rate limiting (§11).
        self._add_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> List[CorpusEntry]:
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            # Backward compat: plain list of strings → migrate to CorpusEntry.
            migrated = []
            for item in data:
                if isinstance(item, str):
                    migrated.append(CorpusEntry(
                        text=item,
                        label_source="human_review",
                        quarantined=False,
                        label="attack",
                    ))
                else:
                    migrated.append(item)
            return migrated
        return []

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._learned, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Add entries
    # ------------------------------------------------------------------

    def add_confirmed_attack(
        self,
        text: str,
        label_source: LabelSource = "human_review",
        quarantine: bool = False,
    ) -> bool:
        """
        Add a confirmed attack to the corpus.

        §10: label_source is stored on every entry.
        §11: production_flag entries are auto-quarantined.
               Per-source rate limit enforced (MAX_ADDS_PER_BATCH).
               Poisoning check: high overlap with benign anchors auto-quarantined.

        Returns True if entry was added, False if rate-limited or duplicate.
        """
        # Auto-quarantine production_flag entries (§11).
        if label_source == "production_flag":
            quarantine = True

        # Poisoning check: if prompt overlaps heavily with benign corpus, quarantine it (M11)
        if not quarantine and check_poisoning_risk(text, self.benign_examples):
            quarantine = True

        # Rate limit per label_source (§11).
        count = self._add_counts.get(label_source, 0)
        if count >= MAX_ADDS_PER_BATCH:
            return False  # Rate limit hit — caller should retry later.
        self._add_counts[label_source] = count + 1

        # Dedup (check non-quarantined and quarantined both).
        existing_texts = {e["text"] for e in self._learned}
        if text in existing_texts:
            return False

        self._learned.append(CorpusEntry(
            text=text,
            label_source=label_source,
            quarantined=quarantine,
            label="attack",
        ))
        self._save()
        return True

    def add_benign_example(
        self,
        text: str,
        label_source: LabelSource = "human_review",
        quarantine: bool = False,
    ) -> bool:
        """
        Add a benign negative to the corpus (§9) — security-research,
        fiction, sarcasm, technical prompts — to reduce false positives.

        These are returned by benign_examples and used by semantic_layer
        to discount scores when a prompt is more similar to benign content.
        """
        if label_source == "production_flag":
            quarantine = True

        count = self._add_counts.get(label_source, 0)
        if count >= MAX_ADDS_PER_BATCH:
            return False
        self._add_counts[label_source] = count + 1

        existing_texts = {e["text"] for e in self._learned}
        if text in existing_texts:
            return False

        self._learned.append(CorpusEntry(
            text=text,
            label_source=label_source,
            quarantined=quarantine,
            label="benign",
        ))
        self._save()
        return True

    def confirm_entry(self, text: str) -> bool:
        """
        Lift quarantine on an entry (§11 — corpus-poisoning protection).
        Returns True if an entry was found and confirmed.
        """
        for entry in self._learned:
            if entry["text"] == text and entry["quarantined"]:
                entry["quarantined"] = False
                self._save()
                return True
        return False

    def reset_rate_limits(self) -> None:
        """Reset per-source add counters (call between benchmark rounds etc.)."""
        self._add_counts = {}

    # ------------------------------------------------------------------
    # Query properties
    # ------------------------------------------------------------------

    @property
    def attack_examples(self) -> List[str]:
        """
        All non-quarantined attack examples.
        §11: Quarantined entries are excluded.
        §10: Only red_team and human_review entries are returned automatically.
        """
        learned_attacks = [
            e["text"] for e in self._learned
            if e["label"] == "attack"
            and not e["quarantined"]
            and e["label_source"] in ("red_team", "human_review")
        ]
        return KNOWN_ATTACK_EXAMPLES + learned_attacks

    @property
    def benign_examples(self) -> List[str]:
        """
        All non-quarantined benign examples — includes security-research
        negatives added via add_benign_example() (§9).
        """
        learned_benign = [
            e["text"] for e in self._learned
            if e["label"] == "benign"
            and not e["quarantined"]
        ]
        return KNOWN_BENIGN_EXAMPLES + SECURITY_RESEARCH_BENIGN + learned_benign

    @property
    def learned_count(self) -> int:
        return len(self._learned)

    @property
    def quarantined_count(self) -> int:
        return sum(1 for e in self._learned if e["quarantined"])
