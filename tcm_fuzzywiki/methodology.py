"""Strict methodology compliance matrix for TCM-FuzzyWiki V5.0.

This module is intentionally conservative: it separates software capabilities that
are implemented by the pipeline from research inputs that must remain external
(expert calibration scores, reviewed rules, and gold standards).  The resulting
CSV/Markdown audit page prevents the build from claiming that bootstrap data are
already formal expert validation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

REQUIREMENT_MATRIX: tuple[dict[str, str], ...] = (
    {
        "stage": "A-B",
        "requirement": "XLSX/CSV row to Source Evidence Unit with metadata",
        "software_status": "implemented",
        "artifact": "source_units.csv; source_metadata.csv; wiki/sources/*.md",
        "evidence": "read_chapters preserves chapter, school, region, tradition, text family, citation family, and evidence-quality fields.",
        "external_dependency": "Curated source metadata quality still depends on the input spreadsheet.",
    },
    {
        "stage": "C",
        "requirement": "Observation-first extraction; LLM must not output syndrome conclusions",
        "software_status": "implemented",
        "artifact": "observations.csv; tcm_fuzzywiki/llmlite.py; tcm_fuzzywiki/extraction.py",
        "evidence": "llmlite ChatModel and Azure adapter are available; prompts restrict the LLM to observable facts; deterministic extractor supports offline tests.",
        "external_dependency": "LLM extraction quality requires manual/expert review on real corpora.",
    },
    {
        "stage": "D-E",
        "requirement": "Ontology lexicon and bootstrap prior export for expert curation",
        "software_status": "implemented",
        "artifact": "ontology_lexicon.csv; entities.csv; observation_mapping.csv; fuzzy_sets.csv",
        "evidence": "Entities, observation mappings, and linguistic values are flattened into auditable CSV tables.",
        "external_dependency": "Long-term ontology alignment to external TCM terminologies remains a curation task.",
    },
    {
        "stage": "F",
        "requirement": "Overlap-integral fuzzy membership and low-ICC uncertainty propagation",
        "software_status": "implemented",
        "artifact": "memberships.csv; expert_calibration_template.csv",
        "evidence": "Trapezoidal overlap integral, ICC-aware Monte Carlo intervals, p5/p95, and calibration rows are generated.",
        "external_dependency": "Formal calibrated memberships require expert scores and ICC estimates.",
    },
    {
        "stage": "G-H",
        "requirement": "Scalable observation co-occurrence mining and candidate patterns",
        "software_status": "implemented",
        "artifact": "observation_itemsets.csv; cooccurrence_stats.csv; candidate_patterns.csv; wiki/patterns/*.md",
        "evidence": "Apriori-style support pruning with integer tidsets, lift/PMI/Jaccard/Fisher metadata, small-N warnings, and page caps are implemented.",
        "external_dependency": "Candidate patterns are hypotheses, not rules, until expert-reviewed.",
    },
    {
        "stage": "I",
        "requirement": "Expert review lifecycle from candidate pattern to active rule",
        "software_status": "implemented_file_workflow",
        "artifact": "expert_rule_review.csv; rule_lifecycle.csv; optional reviewed rules.csv",
        "evidence": "The pipeline emits review templates and can load expert-accepted rules from CSV without changing code.",
        "external_dependency": "Actual expert decisions, consequents, weights, and conflict notes must be supplied by reviewers.",
    },
    {
        "stage": "J",
        "requirement": "Larsen-style weighted activation inference with fuzzy-variable antecedents",
        "software_status": "implemented",
        "artifact": "rules.csv; inference_results.csv; wiki/rules/*.md; wiki/sources/*.md",
        "evidence": "Rule activation uses rule_weight × product(mu_i ** weight_i), and missing/thresholded antecedents are recorded.",
        "external_dependency": "Rule validity depends on seed rules and/or expert-reviewed induced rules.",
    },
    {
        "stage": "J-optional",
        "requirement": "Optional Mamdani sensitivity analysis",
        "software_status": "implemented_optional",
        "artifact": "mamdani_results.csv; wiki/audit/mamdani_sensitivity.md",
        "evidence": "Consequent fuzzy sets are truncated, max-aggregated, and centroid-defuzzified for sensitivity checks.",
        "external_dependency": "This is an optional validation module, not the default large-scale inference claim.",
    },
    {
        "stage": "K",
        "requirement": "Correlation-discounted source/tradition/global hierarchical aggregation",
        "software_status": "implemented",
        "artifact": "aggregation_results.csv; source_stratification.csv; wiki/traditions/*.md; wiki/synthesis/global_syndrome_spectrum.md",
        "evidence": "Source-level discounted weighted means, tradition weighted means, and cross-tradition discounted noisy-or are generated.",
        "external_dependency": "Independence discount values should be reviewed for each corpus/tradition design.",
    },
    {
        "stage": "L",
        "requirement": "Auditable Markdown Wiki generation",
        "software_status": "implemented",
        "artifact": "wiki/index.md; wiki/sources; wiki/observations; wiki/rules; wiki/patterns; wiki/audit",
        "evidence": "Pages expose original text, observations, memberships, inference, aggregation, pattern metadata, and formal-boundary statements.",
        "external_dependency": "Interpretation remains bounded by corpus quality and expert-reviewed inputs.",
    },
    {
        "stage": "Evaluation",
        "requirement": "FCR/CRP/MIC/SMB/FIA metric formulas and gold-standard templates",
        "software_status": "implemented_requires_gold",
        "artifact": "evaluation_results.csv; evaluation_gold_templates.csv; wiki/audit/evaluation_metrics.md",
        "evidence": "Metric formulas run when gold CSVs exist; otherwise explicit needs_gold_standard rows and templates are emitted.",
        "external_dependency": "Formal metric values require expert gold-standard files.",
    },
)


def methodology_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in REQUIREMENT_MATRIX:
        enriched = dict(row)
        enriched["source_count"] = summary.get("source_count", 0)
        enriched["observation_count"] = summary.get("observation_count", 0)
        enriched["candidate_pattern_count"] = summary.get("candidate_pattern_count", 0)
        enriched["rule_count"] = summary.get("rule_count", 0)
        enriched["completion_verdict"] = summary.get("completion_verdict", "")
        rows.append(enriched)
    return rows


def methodology_markdown(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["software_status"] for row in rows)
    count_rows = "\n".join(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    table_rows = "\n".join(
        "| {stage} | {requirement} | {software_status} | {artifact} | {external_dependency} |".format(**row)
        for row in rows
    )
    return f"""# V5.0 方法学合规矩阵

## 状态计数

| Software status | Count |
|---|---:|
{count_rows}

## 逐项要求

| Stage | Requirement | Software status | Artifact | External dependency / boundary |
|---|---|---|---|---|
{table_rows}

## 审计结论

本页区分“代码已实现的可复算文件化流程”和“必须由专家、本体维护或 gold standard 提供的外部研究输入”。因此，即使软件链路完整运行，也不会把 bootstrap prior、未审核 candidate pattern 或缺失 gold 的指标误称为已经完成的正式专家验证。
"""
