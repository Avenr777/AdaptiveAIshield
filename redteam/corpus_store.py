"""
Minimal JSON-lines storage for the attack corpus and results.

Good enough for a class project / prototype. Swap this out for the
Postgres + vector-DB setup described in the report once you wire the
Red Team into the real Adaptive Learning loop — the interface
(`save_batch`, `load_all_attacks`, `save_results`) is small on purpose
so that swap is easy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import Attack, AttackResult, GenerationBatch


class CorpusStore:
    def __init__(self, path: str = "attack_corpus.jsonl"):
        self.attacks_path = Path(path)
        self.results_path = Path(path).with_suffix(".results.jsonl")

    def save_batch(self, batch: GenerationBatch) -> None:
        with self.attacks_path.open("a", encoding="utf-8") as f:
            for attack in batch.attacks:
                f.write(json.dumps(attack.to_dict()) + "\n")

    def load_all_attacks(self) -> List[Attack]:
        if not self.attacks_path.exists():
            return []
        attacks = []
        with self.attacks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    attacks.append(Attack.from_dict(json.loads(line)))
        return attacks

    def save_results(self, results: List[AttackResult]) -> None:
        with self.results_path.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")

    def load_all_results(self) -> List[AttackResult]:
        if not self.results_path.exists():
            return []
        results = []
        with self.results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(AttackResult(**json.loads(line)))
        return results
