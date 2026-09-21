"""
Core data models for the AIShield Blue Team firewall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Literal


class Action(str, Enum):
    ALLOW = "allow"                  # 0-25: Safe
    ALLOW_LOGGED = "allow_logged"    # 26-50: Low risk (allow, log for telemetry)
    WARN = "warn"                    # 51-75: Suspicious (prompt confirmation required)
    BLOCK = "block"                  # 76-100: Malicious (reject outright)


# Keep Decision as an alias to Action for backwards compatibility
Decision = Action


@dataclass
class BandConfig:
    """Configurable threshold bands mapping risk scores (0-100) to actions."""
    safe_max: float = 25.0
    low_max: float = 50.0
    suspicious_max: float = 75.0
    veto_threshold: float = 85.0     # T_veto: any single non-LLM signal >= this forces BLOCK

    def map_score_to_action(self, score: float, veto_triggered: bool = False) -> Action:
        if veto_triggered:
            return Action.BLOCK
        if score <= self.safe_max:
            return Action.ALLOW
        if score <= self.low_max:
            return Action.ALLOW_LOGGED
        if score <= self.suspicious_max:
            return Action.WARN
        return Action.BLOCK


DEFAULT_BANDS = BandConfig()


@dataclass
class LayerScore:
    """One analysis layer's contribution to the overall risk score."""
    layer: str                       # "lexical" | "semantic" | "context" | "judge"
    score: float                     # 0-100
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class UserExplanation:
    """Coarse, privacy-safe explanation displayed to the end user."""
    action: Action
    message: str                     # e.g., "Request permitted" or "Request blocked by safety policy"


@dataclass
class DeveloperExplanation:
    """Full diagnostic payload for security operators / developers."""
    action: Action
    risk_score: float
    layer_scores: List[LayerScore]
    reasons: List[str]
    latencies_ms: Dict[str, float]
    veto_triggered: bool = False
    stage_reached: str = "fast"      # "lexical_early_exit" | "embedding_fast" | "llm_judge"
    judge_evaluation: Optional[Dict] = None


@dataclass
class FirewallResult:
    decision: Action
    risk_score: float                  # 0-100
    layer_scores: List[LayerScore]
    reasons: List[str]                 # Full reasons
    normalized_text: str = ""          # text after de-obfuscation
    user_view: Optional[UserExplanation] = None
    dev_view: Optional[DeveloperExplanation] = None
    total_latency_ms: float = 0.0
    warn_state: Optional[Literal["confirmed", "abandoned"]] = None

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "risk_score": round(self.risk_score, 1),
            "layer_scores": [
                {
                    "layer": ls.layer,
                    "score": round(ls.score, 1),
                    "reasons": ls.reasons,
                    "latency_ms": round(ls.latency_ms, 2),
                }
                for ls in self.layer_scores
            ],
            "reasons": self.reasons,
            "user_explanation": self.user_view.message if self.user_view else "",
            "normalized_text": self.normalized_text,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }
