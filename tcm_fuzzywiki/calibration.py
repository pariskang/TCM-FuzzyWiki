"""Expert calibration workflow for bootstrap linguistic values.

The V5.0 method starts from bootstrap priors and then calibrates memberships with
expert scores.  This module closes that loop by reading expert membership scores,
computing median calibrated memberships and an ICC-like one-way reliability
estimate, and writing an updated YAML config plus a calibration report.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import write_csv
from .models import clamp01

REQUIRED_COLUMNS = {"term", "variable", "fuzzy_set", "expert_id", "score"}


def calibrate_config_from_experts(
    config: dict[str, Any],
    expert_scores_csv: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return an updated config and row-level calibration report.

    Expected CSV columns:
    - term: linguistic value key, e.g. 冷痛
    - variable: fuzzy variable, e.g. cold_property
    - fuzzy_set: fuzzy set label, e.g. high
    - expert_id: expert/rater identifier
    - score: expert membership score in [0, 1]
    """

    frame = pd.read_csv(expert_scores_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Expert calibration CSV missing columns: {sorted(missing)}")

    calibrated = deepcopy(config)
    linguistic_values = calibrated.setdefault("linguistic_values", {})
    report: list[dict[str, Any]] = []

    grouped = frame.groupby(["term", "variable", "fuzzy_set"], dropna=False)
    for (term, variable, fuzzy_set), group in grouped:
        scores = [clamp01(score) for score in group["score"].tolist()]
        median = float(np.median(scores))
        mean = float(np.mean(scores))
        p5 = float(np.percentile(scores, 5))
        p95 = float(np.percentile(scores, 95))
        icc = _one_way_icc(group)
        status = "expert_calibrated" if icc is None or icc >= 0.75 else "expert_calibrated_low_icc"

        term_entry = linguistic_values.setdefault(str(term), {"feature": "expert_calibrated", "maps_to": {}})
        maps_to = term_entry.setdefault("maps_to", {})
        mapping = maps_to.setdefault(str(variable), {})
        mapping["fuzzy_set"] = str(fuzzy_set)
        mapping["prior_membership"] = round(median, 6)
        mapping["calibrated_membership"] = round(median, 6)
        mapping["expert_mean"] = round(mean, 6)
        mapping["expert_p5"] = round(p5, 6)
        mapping["expert_p95"] = round(p95, 6)
        mapping["status"] = status
        mapping["icc"] = None if icc is None else round(icc, 6)
        mapping["review_status"] = "expert_reviewed"
        mapping["expert_count"] = int(group["expert_id"].nunique())
        mapping["score_count"] = int(len(group))

        report.append(
            {
                "term": term,
                "variable": variable,
                "fuzzy_set": fuzzy_set,
                "calibrated_membership": round(median, 6),
                "expert_mean": round(mean, 6),
                "expert_p5": round(p5, 6),
                "expert_p95": round(p95, 6),
                "icc": None if icc is None else round(icc, 6),
                "status": status,
                "expert_count": int(group["expert_id"].nunique()),
                "score_count": int(len(group)),
            }
        )
    return calibrated, report


def write_calibrated_config(
    config: dict[str, Any],
    report: list[dict[str, Any]],
    output_config: str | Path,
    report_csv: str | Path | None = None,
) -> None:
    output_config = Path(output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if report_csv:
        write_csv(report_csv, report)


def _one_way_icc(group: pd.DataFrame) -> float | None:
    """Compute ICC(1,1)-style reliability for term/variable expert scores.

    A single item with one score per expert has no within-item variance estimate;
    in that common calibration-review case, return a conservative agreement proxy
    derived from score dispersion so low agreement still propagates downstream.
    """

    experts = sorted(group["expert_id"].astype(str).unique())
    if len(experts) < 2:
        return None
    scores = np.array([clamp01(value) for value in group["score"].tolist()], dtype=float)
    if len(scores) <= len(experts):
        dispersion = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        return clamp01(1.0 - dispersion / 0.5)

    pivot = group.pivot_table(index=group.index, columns="expert_id", values="score", aggfunc="mean")
    values = pivot.dropna(axis=0, how="any").to_numpy(dtype=float)
    n, k = values.shape if values.size else (0, 0)
    if n < 2 or k < 2:
        dispersion = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        return clamp01(1.0 - dispersion / 0.5)
    row_means = values.mean(axis=1)
    grand_mean = values.mean()
    ms_between = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    ms_within = np.sum((values - row_means[:, None]) ** 2) / (n * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    if denominator <= 0:
        return 1.0
    return clamp01((ms_between - ms_within) / denominator)
