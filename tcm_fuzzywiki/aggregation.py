"""Correlation-aware source/tradition/global aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .models import FuzzyRule, InferenceResult, SourceUnit


def aggregate(
    inference_results: list[InferenceResult],
    sources: list[SourceUnit],
    rules: list[FuzzyRule],
    config: dict[str, Any] | None = None,
    gamma: float = 0.5,
    default_delta: float = 1.0,
) -> list[dict[str, Any]]:
    config = config or {}
    gamma = float(config.get("source_rule_discount_gamma", gamma))
    default_delta = float(config.get("default_tradition_delta", default_delta))
    tradition_deltas = config.get("tradition_independence_weights", {})
    source_meta = {source.source_id: source for source in sources}
    rule_quality = {rule.rule_id: max(rule.rule_weight, rule.expert_acceptance_rate, 0.1) for rule in rules}
    fired = [result for result in inference_results if result.status == "fired"]

    by_source_conclusion: dict[tuple[str, str], list[InferenceResult]] = defaultdict(list)
    for result in fired:
        by_source_conclusion[(result.source_id, result.consequent_entity)].append(result)

    source_rows: list[dict[str, Any]] = []
    for (source_id, conclusion), rows in by_source_conclusion.items():
        n = len(rows)
        qualities = [rule_quality.get(row.rule_id, 0.5) for row in rows]
        denom = sum(qualities) or 1.0
        mu_weighted_mean = sum(quality * row.activation for quality, row in zip(qualities, rows)) / denom
        # Quality-scaled noisy-or is the independent-evidence ceiling: the
        # best-quality rule contributes its full activation, lower-quality rules
        # contribute proportionally less, so n == 1 keeps mu == activation.
        max_quality = max(qualities) or 1.0
        complement = 1.0
        for quality, row in zip(qualities, rows):
            complement *= 1.0 - min(1.0, max(0.0, (quality / max_quality) * row.activation))
        mu_noisy_or = 1.0 - complement
        # Correlation discount: gamma = 0 treats co-fired rules as independent
        # evidence (noisy-or); larger gamma treats them as increasingly redundant
        # and pulls mu back toward the quality-weighted mean, the fully-correlated
        # floor.  Both ends coincide for a single rule, so the discount only
        # affects genuinely multi-rule evidence.
        discount_factor = 1.0 / math.pow(n, gamma)
        mu = max(0.0, min(1.0, mu_weighted_mean + discount_factor * (mu_noisy_or - mu_weighted_mean)))
        source = source_meta[source_id]
        source_rows.append(
            {
                "aggregation_level": "source",
                "source_id": source_id,
                "tradition_id": source.tradition_id,
                "consequent_entity": conclusion,
                "mu": round(mu, 6),
                "mu_weighted_mean": round(mu_weighted_mean, 6),
                "mu_noisy_or": round(mu_noisy_or, 6),
                "evidence_count": n,
                "quality_weight": round(source.evidence_quality.score, 6),
                "rule_discount_gamma": gamma,
                "rule_discount_factor": round(discount_factor, 6),
                "effective_rule_weight_sum": round(sum(qualities) * discount_factor, 6),
            }
        )

    by_tradition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_tradition[(row["tradition_id"], row["consequent_entity"])].append(row)

    tradition_rows: list[dict[str, Any]] = []
    for (tradition_id, conclusion), rows in by_tradition.items():
        denom = sum(row["quality_weight"] for row in rows) or 1.0
        mu = sum(row["quality_weight"] * row["mu"] for row in rows) / denom
        tradition_rows.append(
            {
                "aggregation_level": "tradition",
                "tradition_id": tradition_id,
                "consequent_entity": conclusion,
                "mu": round(mu, 6),
                "evidence_count": sum(row["evidence_count"] for row in rows),
                "source_count": len(rows),
            }
        )

    by_conclusion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tradition_rows:
        by_conclusion[row["consequent_entity"]].append(row)

    global_rows: list[dict[str, Any]] = []
    for conclusion, rows in by_conclusion.items():
        product = 1.0
        deltas: list[float] = []
        for row in rows:
            delta = float(tradition_deltas.get(row.get("tradition_id"), default_delta))
            deltas.append(delta)
            product *= 1.0 - delta * row["mu"]
        global_rows.append(
            {
                "aggregation_level": "global",
                "consequent_entity": conclusion,
                "mu": round(max(0.0, min(1.0, 1.0 - product)), 6),
                "evidence_count": sum(row["evidence_count"] for row in rows),
                "tradition_count": len(rows),
                "delta_values": deltas,
            }
        )

    return source_rows + tradition_rows + global_rows
