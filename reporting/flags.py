"""
reporting/flags.py — Shared statistical computation and sample size flagging.

Ensures consistent calculation of:
  - Mean, sample standard deviation (ddof=1)
  - Two-sided 95% Student-t confidence intervals
  - Wilson 95% confidence intervals for proportions (§17)
  - Sample size threshold enforcement (flagging when n < threshold)

§17: 10+ seeds, raw counts, Wilson 95% CIs for ASR, detection, FPR.
     Wilson CI is preferred over t-CI for small-n proportions near 0 or 1.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple


T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}

Z_95 = 1.96  # z* for 95% two-sided Wilson CI


def wilson_ci(successes: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """
    §17: Wilson score interval for a proportion.

    Args:
        successes: Number of successes (e.g. attacks that evaded).
        n:         Total trials.
        z:         z* for desired confidence level (default 1.96 = 95%).

    Returns:
        (lower, upper) as fractions in [0, 1].

    Notes:
        - More accurate than normal approximation (Wald) for p near 0 or 1.
        - Works at n=0 (returns (0.0, 1.0) — uninformative prior).
    """
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


def wilson_ci_pct(successes: int, n: int) -> Tuple[float, float]:
    """Wilson CI returned as percentage points (0-100)."""
    lo, hi = wilson_ci(successes, n)
    return lo * 100.0, hi * 100.0


def compute_stats(values: List[float]) -> Dict[str, float]:
    """
    Computes mean, sample standard deviation (ddof=1), min, max,
    and two-sided 95% Student-t confidence interval half-width.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 3), "std": 0.0, "ci95": 0.0, "min": round(mean, 3), "max": round(mean, 3)}

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    df = n - 1
    t_crit = T_TABLE_95.get(df, 1.96 if df > 120 else 2.042)
    ci95 = t_crit * (std / math.sqrt(n))

    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci95": round(ci95, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def compute_proportion_stats(
    successes_list: List[int],
    totals_list: List[int],
) -> Dict[str, float]:
    """
    §17: Aggregate multi-run proportion statistics using Wilson CIs.

    For each run, compute the proportion (successes/total). Then aggregate:
    - Mean proportion across runs (percentage).
    - Mean Wilson CI half-width across runs (percentage points).
    - Pooled Wilson CI on total successes / total trials.

    Returns a dict suitable for reporting alongside compute_stats().
    """
    if not successes_list or not totals_list:
        return {"mean": 0.0, "std": 0.0, "wilson_lo": 0.0, "wilson_hi": 0.0}

    per_run_pcts = []
    for s, n in zip(successes_list, totals_list):
        per_run_pcts.append((s / n * 100.0) if n > 0 else 0.0)

    mean_pct = sum(per_run_pcts) / len(per_run_pcts)
    n_runs = len(per_run_pcts)
    variance = sum((x - mean_pct) ** 2 for x in per_run_pcts) / max(n_runs - 1, 1)
    std_pct = math.sqrt(variance)

    # Pooled Wilson CI across all runs.
    total_s = sum(successes_list)
    total_n = sum(totals_list)
    lo, hi = wilson_ci_pct(total_s, total_n)

    return {
        "mean": round(mean_pct, 3),
        "std": round(std_pct, 3),
        "wilson_lo": round(lo, 3),
        "wilson_hi": round(hi, 3),
        "total_successes": total_s,
        "total_n": total_n,
    }


def check_sample_size_flag(n: float, threshold: int = 20) -> Tuple[bool, str]:
    """
    Returns (is_low_n, flag_message).
    Enforces minimum sample size threshold so low-n percentages are not
    falsely presented as stable statistics.
    """
    if n < threshold:
        return True, f"# WARNING: mean n={n:.1f} < {threshold}, CI unreliable"
    return False, ""


def format_stat_with_ci(stats: Dict[str, float], digits: int = 1) -> str:
    """Formats stat as 'mean+/-std [+/-ci95]'."""
    m = f"{stats['mean']:.{digits}f}"
    s = f"{stats['std']:.{digits}f}"
    c = f"{stats['ci95']:.{digits}f}"
    return f"{m}+/-{s} [+/-{c}]"


def format_fraction_fpr(fp_count: int, total_n: int, digits: int = 2) -> str:
    """Formats FPR percentage with explicit numerator and denominator."""
    pct = (fp_count / total_n * 100.0) if total_n else 0.0
    return f"{pct:.{digits}f}% ({fp_count}/{total_n})"


def format_wilson(stats: Dict[str, float], digits: int = 1) -> str:
    """Formats Wilson CI proportion as 'mean% [lo–hi%]'."""
    m = f"{stats['mean']:.{digits}f}"
    lo = f"{stats['wilson_lo']:.{digits}f}"
    hi = f"{stats['wilson_hi']:.{digits}f}"
    return f"{m}% [{lo}–{hi}%]"

