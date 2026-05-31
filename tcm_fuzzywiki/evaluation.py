"""Evaluation metrics for TCM-FuzzyWiki V5.0.

Metrics that require expert gold standards return explicit template/status rows
when gold labels are not supplied.  This keeps the pipeline complete and honest:
formulas are implemented, but missing external labels are not fabricated.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .models import InferenceResult, Membership


def evaluate_pipeline(
    memberships: list[Membership],
    inference_results: list[InferenceResult],
    entities: list[dict[str, Any]],
    gold_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold_path = Path(gold_dir) if gold_dir else None
    rows: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []

    rows.extend(_fcr(memberships, gold_path))
    rows.extend(_fia_local(inference_results, gold_path))
    rows.extend(_mic(inference_results, gold_path))
    rows.extend(_smb(entities, gold_path))
    rows.extend(_crp(gold_path))
    rows.extend(_fia_chain(gold_path))
    rows.append(_intrinsic_multi_interpretation_proxy(inference_results))

    templates.extend(_gold_templates())
    return rows, templates


def _fcr(memberships: list[Membership], gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "expert_memberships.csv")
    if gold is None:
        return [_needs_gold("FCR", "expert_memberships.csv", "1 - mean(|μ_model - μ_expert|)")]
    model = {
        (m.source_id, m.standard_observation, m.variable, m.fuzzy_set): m.membership
        for m in memberships
    }
    diffs: list[float] = []
    for row in gold.to_dict("records"):
        key = (str(row["source_id"]), str(row["standard_observation"]), str(row["variable"]), str(row["fuzzy_set"]))
        if key in model:
            diffs.append(abs(model[key] - float(row["expert_membership"])))
    return [_score("FCR", 1.0 - sum(diffs) / len(diffs), len(diffs))] if diffs else [_needs_gold("FCR", "expert_memberships.csv", "No matching gold rows")]


def _fia_local(inference_results: list[InferenceResult], gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "expert_inference.csv")
    if gold is None:
        return [_needs_gold("FIA-local", "expert_inference.csv", "1 - mean(|α_model - μ_expert|)")]
    model = {(r.source_id, r.consequent_entity): r.activation for r in inference_results if r.status == "fired"}
    diffs: list[float] = []
    for row in gold.to_dict("records"):
        key = (str(row["source_id"]), str(row["consequent_entity"]))
        if key in model:
            diffs.append(abs(model[key] - float(row["expert_membership"])))
    return [_score("FIA-local", 1.0 - sum(diffs) / len(diffs), len(diffs))] if diffs else [_needs_gold("FIA-local", "expert_inference.csv", "No matching gold rows")]


def _mic(inference_results: list[InferenceResult], gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "expected_interpretations.csv")
    if gold is None:
        return [_needs_gold("MIC", "expected_interpretations.csv", "mean(|model ∩ expected| / |expected|)")]
    by_source: dict[str, set[str]] = defaultdict(set)
    for result in inference_results:
        if result.status == "fired":
            by_source[result.source_id].add(result.consequent_entity)
    scores: list[float] = []
    for row in gold.to_dict("records"):
        expected = _split_set(row.get("expected_consequents", ""))
        if expected:
            scores.append(len(by_source[str(row["source_id"])] & expected) / len(expected))
    return [_score("MIC", sum(scores) / len(scores), len(scores))] if scores else [_needs_gold("MIC", "expected_interpretations.csv", "No expected interpretations")]


def _smb(entities: list[dict[str, Any]], gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "modern_mappings.csv")
    if gold is None:
        mapped = [entity for entity in entities if str(entity.get("modern_mapping", "")).strip()]
        return [_score("SMB-coverage-proxy", len(mapped) / max(1, len(entities)), len(entities), status="proxy")]
    model = {str(entity.get("entity_name")): _split_set(entity.get("modern_mapping", "")) for entity in entities}
    scores: list[float] = []
    for row in gold.to_dict("records"):
        expected = _split_set(row.get("expected_modern_mapping", ""))
        predicted = model.get(str(row["entity_name"]), set())
        if expected:
            scores.append(len(predicted & expected) / len(expected))
    return [_score("SMB", sum(scores) / len(scores), len(scores))] if scores else [_needs_gold("SMB", "modern_mappings.csv", "No expected mappings")]


def _crp(gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "conditional_relations.csv")
    if gold is None:
        return [_needs_gold("CRP", "conditional_relations.csv", "Requires condition-specific expert relation strengths")]
    if "model_membership" not in gold or "expert_membership" not in gold:
        return [_needs_gold("CRP", "conditional_relations.csv", "Columns model_membership/expert_membership are required")]
    diffs = [abs(float(row["model_membership"]) - float(row["expert_membership"])) for row in gold.to_dict("records")]
    return [_score("CRP", 1.0 - sum(diffs) / len(diffs), len(diffs))] if diffs else [_needs_gold("CRP", "conditional_relations.csv", "No rows")]


def _fia_chain(gold_path: Path | None) -> list[dict[str, Any]]:
    gold = _read_gold(gold_path, "chain_paths.csv")
    if gold is None:
        return [_needs_gold("FIA-chain", "chain_paths.csv", "Requires expert/model path overlap rows")]
    if "path_overlap" in gold:
        values = [float(value) for value in gold["path_overlap"].dropna().tolist()]
        return [_score("FIA-chain", sum(values) / len(values), len(values))] if values else [_needs_gold("FIA-chain", "chain_paths.csv", "No path_overlap values")]
    return [_needs_gold("FIA-chain", "chain_paths.csv", "Column path_overlap is required")]


def _intrinsic_multi_interpretation_proxy(inference_results: list[InferenceResult]) -> dict[str, Any]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for result in inference_results:
        if result.status == "fired":
            by_source[result.source_id].add(result.consequent_entity)
    counts = [len(values) for values in by_source.values()]
    value = sum(counts) / len(counts) if counts else 0.0
    return {
        "metric": "multi_interpretation_count_proxy",
        "value": round(value, 6),
        "n": len(counts),
        "status": "proxy",
        "formula": "mean fired consequents per source",
        "required_gold_file": "",
    }


def _read_gold(gold_path: Path | None, name: str) -> pd.DataFrame | None:
    if gold_path is None:
        return None
    path = gold_path / name
    return pd.read_csv(path) if path.exists() else None


def _needs_gold(metric: str, file_name: str, formula: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "value": "",
        "n": 0,
        "status": "needs_gold_standard",
        "formula": formula,
        "required_gold_file": file_name,
    }


def _score(metric: str, value: float, n: int, status: str = "computed") -> dict[str, Any]:
    return {
        "metric": metric,
        "value": round(max(0.0, min(1.0, value)), 6),
        "n": n,
        "status": status,
        "formula": "see metric definition",
        "required_gold_file": "",
    }


def _split_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    if text.startswith("["):
        return {str(item).strip() for item in json.loads(text) if str(item).strip()}
    for sep in [";", "；", ",", "，", "|"]:
        text = text.replace(sep, ";")
    return {part.strip() for part in text.split(";") if part.strip()}


def _gold_templates() -> list[dict[str, Any]]:
    return [
        {"file_name": "expert_memberships.csv", "columns": "source_id,standard_observation,variable,fuzzy_set,expert_membership"},
        {"file_name": "expert_inference.csv", "columns": "source_id,consequent_entity,expert_membership"},
        {"file_name": "expected_interpretations.csv", "columns": "source_id,expected_consequents"},
        {"file_name": "modern_mappings.csv", "columns": "entity_name,expected_modern_mapping"},
        {"file_name": "conditional_relations.csv", "columns": "condition,model_membership,expert_membership"},
        {"file_name": "chain_paths.csv", "columns": "case_id,model_path,expert_path,path_overlap"},
    ]
