"""Rule loading and expert-review promotion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .models import FuzzyRule, RuleAntecedent, clamp01

ACTIVE_STATUSES = {"active_rule", "expert_verified", "expert_verified_rule"}


def load_rules(config: dict[str, Any], rules_csv: str | Path | None = None) -> list[FuzzyRule]:
    """Load seed rules and optional expert-reviewed rules.

    Rows from an expert review CSV are intentionally permissive: pending rows or
    incomplete rows are skipped instead of crashing the full build.  A reviewed
    row becomes a rule when it includes parseable antecedents and a consequent.
    """

    rules: list[FuzzyRule] = []
    for row in config.get("seed_rules", []):
        rule = _maybe_rule_from_mapping(row)
        if rule is not None:
            rules.append(rule)
    if rules_csv and Path(rules_csv).exists():
        frame = pd.read_csv(rules_csv)
        for _, row in frame.iterrows():
            rule = _maybe_rule_from_mapping(row.to_dict())
            if rule is not None:
                rules.append(rule)
    return rules


def _maybe_rule_from_mapping(row: dict[str, Any]) -> FuzzyRule | None:
    decision = str(row.get("expert_decision", "accepted")).strip().lower()
    if decision in {"pending", "reject", "rejected", "否", "不通过"}:
        return None
    antecedents = _parse_antecedents(row.get("antecedents", row.get("antecedents_json", [])))
    if not antecedents:
        return None
    consequent = _parse_consequent(row)
    consequent_entity = _first_nonblank(row.get("consequent_entity"), row.get("suggested_consequent"), consequent.get("entity"))
    if not consequent_entity:
        return None
    return FuzzyRule(
        rule_id=_first_nonblank(row.get("rule_id"), _derived_rule_id(row)) or "RULE_UNSPECIFIED",
        rule_name=_first_nonblank(row.get("rule_name"), row.get("pattern_id"), consequent_entity) or "未命名规则",
        rule_origin=_first_nonblank(row.get("rule_origin"), "cooccurrence_induced") or "cooccurrence_induced",
        pattern_id=_first_nonblank(row.get("pattern_id"), "") or "",
        antecedents=antecedents,
        consequent_entity=consequent_entity,
        consequent_type=_first_nonblank(row.get("consequent_type"), consequent.get("entity_type"), "syndrome") or "syndrome",
        rule_weight=clamp01(_first_nonblank(row.get("rule_weight"), row.get("suggested_rule_weight"), 1.0), 1.0),
        support=_safe_float(row.get("support")),
        confidence=_safe_float(row.get("confidence")),
        lift=_safe_float(row.get("lift")),
        pmi=_safe_float(row.get("pmi")),
        source_count=int(_safe_float(row.get("source_count"))),
        source_ids=_list(row.get("source_ids", [])),
        book_count=int(_safe_float(row.get("book_count"))),
        tradition_count=int(_safe_float(row.get("tradition_count"))),
        tradition_ids=_list(row.get("tradition_ids", [])),
        source_diversity=_safe_float(row.get("source_diversity")),
        tradition_entropy=_safe_float(row.get("tradition_entropy")),
        applicable_context=_first_nonblank(row.get("applicable_context"), "") or "",
        conflict_with=_first_nonblank(row.get("conflict_with"), "") or "",
        expert_acceptance_rate=_safe_float(row.get("expert_acceptance_rate")),
        expert_disagreement_note=_first_nonblank(row.get("expert_disagreement_note"), row.get("conflict_note"), "") or "",
        review_status=_first_nonblank(row.get("review_status"), "active_rule") or "active_rule",
        created_from=_first_nonblank(row.get("created_from"), row.get("pattern_id"), "") or "",
        last_updated=_first_nonblank(row.get("last_updated"), "") or "",
    )


def _parse_antecedents(value: Any) -> list[RuleAntecedent]:
    if _is_blank(value):
        return []
    raw = value
    if isinstance(raw, str):
        raw = yaml.safe_load(raw) if raw.strip() else []
    if not isinstance(raw, list):
        return []
    antecedents: list[RuleAntecedent] = []
    for item in raw:
        if not isinstance(item, dict) or _is_blank(item.get("variable")):
            continue
        antecedents.append(
            RuleAntecedent(
                variable=str(item["variable"]),
                fuzzy_set=str(item.get("fuzzy_set", "high")),
                threshold=_safe_float(item.get("threshold")),
                weight=_safe_float(item.get("weight"), 1.0),
            )
        )
    return antecedents


def _parse_consequent(row: dict[str, Any]) -> dict[str, Any]:
    consequent = row.get("consequent", {})
    if _is_blank(consequent):
        return {}
    if isinstance(consequent, str):
        text = consequent.strip()
        if text.startswith("{"):
            parsed = yaml.safe_load(text)
            return parsed if isinstance(parsed, dict) else {}
        return {"entity": text}
    return consequent if isinstance(consequent, dict) else {}


def _derived_rule_id(row: dict[str, Any]) -> str:
    pattern_id = _first_nonblank(row.get("pattern_id"), "")
    return f"RULE_FROM_{pattern_id}" if pattern_id else ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    if _is_blank(value):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _first_nonblank(*values: Any) -> Any:
    for value in values:
        if not _is_blank(value):
            return value
    return None


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if _is_blank(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return [str(item) for item in json.loads(text)]
        return [part.strip() for part in text.replace("；", ";").split(";") if part.strip()]
    return []
