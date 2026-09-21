"""
A deliberately weak, keyword-only stand-in for the real Blue Team
firewall — just enough so you can run the Red Team pipeline end-to-end
and see the benchmarking loop work before Section 7's actual firewall
exists. Replace `evaluate()` with a call to your real firewall's API.
"""

from typing import List
from redteam.schema import Attack, AttackResult

BLOCK_KEYWORDS = ["ignore", "override", "system prompt", "developer mode", "bypass"]


def evaluate(attacks: List[Attack]) -> List[AttackResult]:
    results = []
    for attack in attacks:
        text = " ".join(attack.turns).lower()
        hit = any(kw in text for kw in BLOCK_KEYWORDS)
        decision = "block" if hit else "allow"
        score = 80.0 if hit else 10.0
        results.append(
            AttackResult(
                attack_id=attack.id,
                decision=decision,
                risk_score=score,
                evaded=(decision == "allow"),
                notes="keyword-only stub — replace with real firewall",
            )
        )
    return results
