"""Exact decimal accuracy thresholds and card-hour estimates."""

from decimal import Decimal, InvalidOperation
from typing import Any


def number(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return result


def accuracy_gate(
    baseline: Any,
    candidate: Any,
    *,
    direction: str,
    max_drop: Any,
    drop_mode: str,
    full_scale: Any = 1,
    metric: str = "score",
) -> dict:
    baseline_value = number(baseline, "baseline")
    candidate_value = number(candidate, "candidate")
    threshold = number(max_drop, "max_drop")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    if threshold < 0:
        raise ValueError("max_drop must be nonnegative")
    zero_baseline_exact = drop_mode == "relative" and baseline_value == 0
    if drop_mode == "full_scale_points":
        scale = number(full_scale, "full_scale")
        if scale <= 0 or threshold > 100:
            raise ValueError("full_scale must be positive and max_drop at most 100 points")
        if not (0 <= baseline_value <= scale and 0 <= candidate_value <= scale):
            raise ValueError("full_scale_points scores must lie between zero and full_scale")
        allowed = scale * threshold / Decimal(100)
    elif drop_mode == "relative":
        if threshold > 1:
            raise ValueError("relative max_drop is a fraction between zero and one")
        allowed = abs(baseline_value) * threshold
    elif drop_mode == "absolute":
        allowed = threshold
    else:
        raise ValueError("drop_mode must be full_scale_points, relative, or absolute")
    degradation = (
        baseline_value - candidate_value
        if direction == "higher"
        else candidate_value - baseline_value
    )
    passed = candidate_value == 0 if zero_baseline_exact else degradation <= allowed
    return {
        "metric": metric,
        "baseline": str(baseline_value),
        "candidate": str(candidate_value),
        "direction": direction,
        "drop_mode": drop_mode,
        "max_drop": str(threshold),
        "allowed_drop": str(allowed),
        "observed_degradation": str(degradation),
        "zero_baseline_requires_exact_match": zero_baseline_exact,
        "passed": passed,
        "official_evaluation": "not_run",
    }


def budget_estimate(
    cards: Any, wall_hours: Any, total_card_hours: Any, used_card_hours: Any = 0
) -> dict:
    count = number(cards, "cards")
    wall = number(wall_hours, "wall_hours")
    total = number(total_card_hours, "total_card_hours")
    used = number(used_card_hours, "used_card_hours")
    if count != count.to_integral_value() or count <= 0:
        raise ValueError("cards must be a positive integer")
    if min(wall, total, used) < 0 or used > total:
        raise ValueError("hours must be nonnegative and used_card_hours cannot exceed total")
    requested = count * wall
    remaining = total - used
    return {
        "cards": int(count),
        "wall_hours": str(wall),
        "estimated_card_hours": str(requested),
        "total_card_hours": str(total),
        "used_card_hours": str(used),
        "remaining_card_hours": str(remaining),
        "remaining_after_run": str(remaining - requested),
        "within_budget": requested <= remaining,
        "billing_basis": "cards multiplied by allocated wall-clock hours; platform billing may differ",
    }
