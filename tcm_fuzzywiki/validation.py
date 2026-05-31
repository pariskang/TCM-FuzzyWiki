"""Validation and readiness checks for TCM-FuzzyWiki V5.0.

This module answers the practical question "is the build perfect enough for the
next research step?"  It does not replace expert review, but it makes missing
configuration, weak metadata, low coverage, absent gold standards, and pending
expert rules visible as machine-readable findings.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .models import FuzzyRule, Membership, Observation, SourceUnit

REQUIRED_CONFIG_SECTIONS = (
    "membership_calculation",
    "mapping_policy",
    "uncertainty_propagation",
    "candidate_pattern_filter",
    "fuzzy_sets",
    "observation_mapping",
    "linguistic_values",
    "seed_rules",
    "entities",
)

REQUIRED_METADATA_FIELDS = (
    "book_name",
    "chapter_title",
    "dynasty",
    "author",
    "text_type",
    "topic_hint",
    "tradition_id",
    "text_family",
    "citation_family",
)


def validate_build(
    config: dict[str, Any],
    sources: list[SourceUnit],
    observations: list[Observation],
    memberships: list[Membership],
    rules: list[FuzzyRule],
    evaluation_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    findings.extend(validate_config(config))
    findings.extend(validate_sources(sources))
    findings.extend(validate_observation_coverage(observations, memberships, summary))
    findings.extend(validate_rules(rules))
    findings.extend(validate_evaluation_readiness(evaluation_rows))
    findings.append(_overall_readiness(findings))
    return findings


def validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in config:
            findings.append(_finding("config", section, "error", "missing_config_section", f"Missing required config section: {section}"))
    fuzzy_sets = config.get("fuzzy_sets", {})
    for rule in config.get("seed_rules", []):
        for ant in rule.get("antecedents", []):
            variable = ant.get("variable")
            fuzzy_set = ant.get("fuzzy_set")
            if variable not in fuzzy_sets or fuzzy_set not in fuzzy_sets.get(variable, {}):
                findings.append(
                    _finding(
                        "config",
                        str(rule.get("rule_id", "unknown_rule")),
                        "error",
                        "rule_references_missing_fuzzy_set",
                        f"Rule antecedent {variable}.{fuzzy_set} has no configured fuzzy set.",
                    )
                )
    return findings or [_finding("config", "all", "info", "config_valid", "Required config sections and seed-rule fuzzy-set references are present.")]


def validate_sources(sources: list[SourceUnit]) -> list[dict[str, Any]]:
    if not sources:
        return [_finding("source", "all", "error", "no_sources", "No source evidence units were loaded.")]
    findings: list[dict[str, Any]] = []
    ids = [source.source_id for source in sources]
    duplicates = [source_id for source_id, count in Counter(ids).items() if count > 1]
    for source_id in duplicates:
        findings.append(_finding("source", source_id, "error", "duplicate_source_id", "Duplicate source_id found."))
    for source in sources:
        for field in REQUIRED_METADATA_FIELDS:
            value = getattr(source, field)
            if value in ("", "uncertain", ["uncertain"]):
                findings.append(
                    _finding(
                        "source",
                        source.source_id,
                        "warning",
                        "uncertain_metadata",
                        f"Metadata field {field} is uncertain and should be reviewed.",
                    )
                )
        if not source.original_text and not source.text_punctuated and not source.text_modern:
            findings.append(_finding("source", source.source_id, "error", "missing_text", "Source has no usable text."))
    return findings or [_finding("source", "all", "info", "source_metadata_ready", "Source identifiers, text, and required metadata are present.")]


def validate_observation_coverage(
    observations: list[Observation], memberships: list[Membership], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not observations:
        return [_finding("observation", "all", "error", "no_observations", "No observations were extracted.")]
    mapped = len({membership.observation_id for membership in memberships})
    coverage = mapped / len(observations)
    threshold = 0.70
    if coverage < threshold:
        findings.append(
            _finding(
                "coverage",
                "membership",
                "warning",
                "low_membership_coverage",
                f"Membership coverage {coverage:.3f} is below exploratory threshold {threshold:.2f}.",
            )
        )
    unmapped_high_conf = [obs for obs in observations if obs.mapping_status != "mapped" and obs.extraction_confidence >= 0.85]
    for obs in unmapped_high_conf[:20]:
        findings.append(
            _finding(
                "observation",
                obs.observation_id,
                "warning",
                "high_confidence_unmapped_observation",
                f"{obs.feature}:{obs.feature_value} from {obs.source_id} needs lexicon review.",
            )
        )
    if not findings:
        findings.append(_finding("coverage", "membership", "info", "coverage_ready", f"Membership coverage is {coverage:.3f}."))
    return findings


def validate_rules(rules: list[FuzzyRule]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not rules:
        return [_finding("rule", "all", "error", "no_rules", "No active or seed rules are loaded.")]
    for rule in rules:
        if not rule.antecedents:
            findings.append(_finding("rule", rule.rule_id, "error", "rule_without_antecedents", "Rule has no antecedents."))
        if not rule.consequent_entity:
            findings.append(_finding("rule", rule.rule_id, "error", "rule_without_consequent", "Rule has no consequent entity."))
        if rule.review_status not in {"active_rule", "expert_verified", "expert_verified_rule"}:
            findings.append(_finding("rule", rule.rule_id, "warning", "inactive_rule_status", f"Rule status is {rule.review_status}."))
    return findings or [_finding("rule", "all", "info", "rules_ready", "Rules have antecedents and consequents.")]


def validate_evaluation_readiness(evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row in evaluation_rows:
        if row.get("status") == "needs_gold_standard":
            findings.append(
                _finding(
                    "evaluation",
                    str(row.get("metric", "unknown_metric")),
                    "warning",
                    "missing_gold_standard",
                    f"Provide {row.get('required_gold_file')} to compute this metric formally.",
                )
            )
    return findings or [_finding("evaluation", "all", "info", "evaluation_ready", "All configured formal metrics have required gold data or proxy status.")]


def readiness_markdown(findings: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"| {item['severity']} | {item['scope']} | {item['item_id']} | {item['code']} | {item['message']} |"
        for item in findings
    )
    counts = Counter(item["severity"] for item in findings)
    return f"""# 完备性与质量验证报告

## 结论

- Errors：{counts.get('error', 0)}
- Warnings：{counts.get('warning', 0)}
- Info：{counts.get('info', 0)}

如果存在 `error`，当前构建不应进入正式分析；如果存在 `warning`，可用于探索性分析，但需要在专家校准或论文实验前补齐。

## 明细

| Severity | Scope | Item | Code | Message |
|---|---|---|---|---|
{rows}
"""


def _overall_readiness(findings: list[dict[str, Any]]) -> dict[str, Any]:
    severities = {finding["severity"] for finding in findings}
    if "error" in severities:
        severity = "error"
        code = "not_ready"
        message = "Build has blocking validation errors."
    elif "warning" in severities:
        severity = "warning"
        code = "exploratory_ready"
        message = "Build is usable for exploration but not perfect/formal without review."
    else:
        severity = "info"
        code = "formal_ready"
        message = "No validation blockers or warnings were detected."
    return _finding("readiness", "overall", severity, code, message)


def _finding(scope: str, item_id: str, severity: str, code: str, message: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "item_id": item_id,
        "severity": severity,
        "code": code,
        "message": message,
    }
