"""Larsen-style weighted activation inference."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .models import FuzzyRule, InferenceResult, Membership
from .rules import ACTIVE_STATUSES


def infer(memberships: list[Membership], rules: list[FuzzyRule]) -> list[InferenceResult]:
    by_source: dict[str, dict[str, float]] = defaultdict(dict)
    intervals: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for mem in memberships:
        key = mem.variable_key
        by_source[mem.source_id][key] = max(by_source[mem.source_id].get(key, 0.0), mem.membership)
        intervals[(mem.source_id, key)] = (mem.p5, mem.p95)

    results: list[InferenceResult] = []
    for source_id, values in by_source.items():
        for rule in rules:
            if rule.review_status not in ACTIVE_STATUSES:
                continue
            activation = rule.rule_weight
            missing: list[str] = []
            supporting: list[str] = []
            low_activation = rule.rule_weight
            high_activation = rule.rule_weight
            has_interval = False
            for antecedent in rule.antecedents:
                value = values.get(antecedent.variable_key, 0.0)
                if value < antecedent.threshold:
                    missing.append(antecedent.variable_key)
                else:
                    supporting.append(antecedent.variable_key)
                activation *= math.pow(max(value, 1e-12), antecedent.weight)
                p5, p95 = intervals.get((source_id, antecedent.variable_key), (None, None))
                low_activation *= math.pow(max(p5 if p5 is not None else value, 1e-12), antecedent.weight)
                high_activation *= math.pow(max(p95 if p95 is not None else value, 1e-12), antecedent.weight)
                has_interval = has_interval or p5 is not None or p95 is not None
            results.append(
                InferenceResult(
                    source_id=source_id,
                    rule_id=rule.rule_id,
                    consequent_entity=rule.consequent_entity,
                    consequent_type=rule.consequent_type,
                    activation=round(max(0.0, min(1.0, activation)), 6),
                    supporting_variables=supporting,
                    missing_variables=missing,
                    status="fired" if not missing else "below_threshold",
                    p5=round(max(0.0, min(1.0, low_activation)), 6) if has_interval else None,
                    p95=round(max(0.0, min(1.0, high_activation)), 6) if has_interval else None,
                )
            )
    return results


def fired_only(results: Iterable[InferenceResult]) -> list[InferenceResult]:
    return [result for result in results if result.status == "fired"]
