"""
Real Blue Team adapter — plugs blueteam.PromptFirewall into the same
`evaluate(attacks) -> List[AttackResult]` interface dummy_blue_team.py
used, so run_round.py and generate_three_batches.py work unchanged.

Swap dummy_blue_team for this module (see run_round.py's --real-firewall
flag) once you want benchmarking to run against the actual firewall
instead of the keyword-only stand-in.
"""

from __future__ import annotations

from typing import List

from redteam.schema import Attack, AttackResult
from blueteam import PromptFirewall

_firewall = PromptFirewall(corpus_path="blue_team_learned_attacks.json")


def evaluate(attacks: List[Attack]) -> List[AttackResult]:
    results = []
    for attack in attacks:
        result = _firewall.evaluate(attack.turns)
        results.append(
            AttackResult(
                attack_id=attack.id,
                decision=result.decision.value,
                risk_score=result.risk_score,
                evaded=(result.decision.value == "allow"),
                notes="; ".join(result.reasons[:2]),
            )
        )
    return results
