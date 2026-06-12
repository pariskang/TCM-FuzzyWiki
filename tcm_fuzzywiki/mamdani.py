"""Optional Mamdani-style sensitivity inference.

The production Wiki path uses Larsen-style weighted activation.  This module
implements the V5.0 optional sensitivity analysis: fired rule activations truncate
an output fuzzy set, rules for the same source/conclusion are max-aggregated, and
a centroid is calculated on [0, 1].
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .membership import Trapezoid, parse_trapezoid, trapz
from .models import FuzzyRule, InferenceResult, clamp01


def run_mamdani_sensitivity(
    inference_results: list[InferenceResult],
    rules: list[FuzzyRule],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = config.get("mamdani_sensitivity", {})
    if not settings.get("enabled", True):
        return []
    points = int(settings.get("points", 200))
    x = np.linspace(0.0, 1.0, points)
    rule_by_id = {rule.rule_id: rule for rule in rules}
    grouped: dict[tuple[str, str], list[InferenceResult]] = defaultdict(list)
    for result in inference_results:
        if result.status == "fired":
            grouped[(result.source_id, result.consequent_entity)].append(result)

    rows: list[dict[str, Any]] = []
    for (source_id, consequent), results in grouped.items():
        aggregate = np.zeros_like(x)
        contributing_rules: list[str] = []
        for result in results:
            rule = rule_by_id.get(result.rule_id)
            if rule is None:
                continue
            shape = _consequent_shape(consequent, rule.consequent_type, config)
            truncated = np.minimum(shape.values(x), result.activation)
            aggregate = np.maximum(aggregate, truncated)
            contributing_rules.append(result.rule_id)
        denominator = trapz(aggregate, x)
        centroid = 0.0 if denominator <= 0 else float(trapz(x * aggregate, x) / denominator)
        rows.append(
            {
                "source_id": source_id,
                "consequent_entity": consequent,
                "centroid": round(clamp01(centroid), 6),
                "max_membership": round(float(np.max(aggregate)), 6),
                "area": round(float(denominator), 6),
                "rules": contributing_rules,
                "mode": "mamdani_centroid_sensitivity",
            }
        )
    return rows


def _consequent_shape(entity: str, entity_type: str, config: dict[str, Any]) -> Trapezoid:
    shapes = config.get("consequent_fuzzy_sets", {})
    for key in (entity, entity_type, "default"):
        if key in shapes:
            return parse_trapezoid(shapes[key])
    return Trapezoid(0.45, 0.70, 1.0, 1.0)
