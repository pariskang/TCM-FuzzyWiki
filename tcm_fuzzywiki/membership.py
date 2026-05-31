"""Fuzzy-set overlap-integral membership computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import Membership, Observation, clamp01


@dataclass(slots=True)
class Trapezoid:
    a: float
    b: float
    c: float
    d: float

    def values(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        if self.b > self.a:
            rising = (x >= self.a) & (x < self.b)
            y[rising] = (x[rising] - self.a) / (self.b - self.a)
        y[(x >= self.b) & (x <= self.c)] = 1.0
        if self.d > self.c:
            falling = (x > self.c) & (x <= self.d)
            y[falling] = (self.d - x[falling]) / (self.d - self.c)
        return np.clip(y, 0.0, 1.0)


def parse_trapezoid(data: Any) -> Trapezoid:
    if isinstance(data, dict):
        points = data.get("points") or data.get("trapezoid")
    else:
        points = data
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError(f"Trapezoid requires four points: {data}")
    return Trapezoid(*(float(v) for v in points))


def overlap_integral(linguistic: Trapezoid, target: Trapezoid, points: int = 200) -> float:
    low = min(linguistic.a, target.a)
    high = max(linguistic.d, target.d)
    x = np.linspace(low, high, points)
    linguistic_values = linguistic.values(x)
    target_values = target.values(x)
    denominator = np.trapezoid(linguistic_values, x)
    if denominator <= 0:
        return 0.0
    numerator = np.trapezoid(linguistic_values * target_values, x)
    return clamp01(numerator / denominator)


class MembershipCalculator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.linguistic_values: dict[str, Any] = config.get("linguistic_values", {})
        self.fuzzy_sets: dict[str, Any] = config.get("fuzzy_sets", {})
        self.mode = config.get("membership_calculation", {}).get("default_mode", "overlap_integral")
        self.points = int(config.get("membership_calculation", {}).get("numerical_integration", {}).get("points", 200))
        uncertainty = config.get("uncertainty_propagation", {})
        self.icc_threshold = float(uncertainty.get("icc_threshold", 0.75))
        self.n_samples = int(uncertainty.get("n_samples", 200))
        self.random_seed = int(uncertainty.get("random_seed", 20260531))

    def compute(self, observations: list[Observation]) -> list[Membership]:
        memberships: list[Membership] = []
        next_id = 1
        rng = np.random.default_rng(self.random_seed)
        for obs in observations:
            entry = self.linguistic_values.get(obs.feature_value) or self.linguistic_values.get(obs.standard_observation)
            if not isinstance(entry, dict):
                continue
            maps_to = entry.get("maps_to", {})
            for variable, mapping in maps_to.items():
                fuzzy_set = str(mapping.get("fuzzy_set", "high"))
                icc = mapping.get("icc")
                icc_value = None if icc is None else float(icc)
                membership = self._compute_value(entry, variable, fuzzy_set, mapping)
                p5 = p95 = width = None
                if icc_value is not None and icc_value < self.icc_threshold:
                    samples = rng.normal(loc=membership, scale=max(0.03, (1.0 - icc_value) * 0.12), size=self.n_samples)
                    samples = np.clip(samples, 0.0, 1.0)
                    membership = float(np.mean(samples))
                    p5 = float(np.percentile(samples, 5))
                    p95 = float(np.percentile(samples, 95))
                    width = p95 - p5
                memberships.append(
                    Membership(
                        membership_id=f"MEM_{next_id:06d}",
                        observation_id=obs.observation_id,
                        source_id=obs.source_id,
                        standard_observation=obs.standard_observation,
                        variable=str(variable),
                        fuzzy_set=fuzzy_set,
                        membership=round(membership, 6),
                        calculation_mode=self.mode,
                        status=str(mapping.get("status", entry.get("status", "bootstrap_prior"))),
                        icc=icc_value,
                        p5=None if p5 is None else round(p5, 6),
                        p95=None if p95 is None else round(p95, 6),
                        uncertainty_width=None if width is None else round(width, 6),
                    )
                )
                next_id += 1
        return memberships

    def _compute_value(self, entry: dict[str, Any], variable: str, fuzzy_set: str, mapping: dict[str, Any]) -> float:
        if self.mode == "overlap_integral" and "linguistic_set" in mapping:
            target_config = self.fuzzy_sets.get(variable, {}).get(fuzzy_set)
            if target_config is not None:
                return overlap_integral(parse_trapezoid(mapping["linguistic_set"]), parse_trapezoid(target_config), self.points)
        return clamp01(mapping.get("prior_membership", 0.0))


def coverage(observations: list[Observation], memberships: list[Membership]) -> float:
    if not observations:
        return 1.0
    mapped_obs_ids = {m.observation_id for m in memberships}
    return len(mapped_obs_ids) / len(observations)


def unmapped_log(observations: list[Observation], threshold: float = 0.85) -> str:
    lines: list[str] = []
    for obs in observations:
        if obs.mapping_status != "mapped" or not obs.standard_observation:
            status = "needs_review" if obs.extraction_confidence >= threshold else "low_confidence_unmapped"
            lines.append(
                f"source_id={obs.source_id}\tfeature={obs.feature}\tfeature_value={obs.feature_value}"
                f"\textraction_confidence={obs.extraction_confidence:.2f}\tstatus={status}"
            )
    return "\n".join(lines) + ("\n" if lines else "")
