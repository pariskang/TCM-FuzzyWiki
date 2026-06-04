"""Implementation audit helpers for TCM-FuzzyWiki V5.0.

The V5.0 proposal is intentionally broad.  This module keeps the produced Wiki
honest by recording which capabilities are implemented in the runnable pipeline,
which are bootstrap/MVP approximations, and which require external expert data or
future system integrations.
"""

from __future__ import annotations

from typing import Any


CAPABILITY_AUDIT: tuple[dict[str, str], ...] = (
    {
        "capability": "XLSX/CSV chapter import",
        "status": "implemented",
        "evidence": "read_chapters supports .xlsx/.xls/.csv and preserves source/tradition metadata.",
    },
    {
        "capability": "Source Evidence Unit",
        "status": "implemented",
        "evidence": "Each chapter row becomes a SourceUnit with evidence quality fields.",
    },
    {
        "capability": "Observation-first LLM extraction",
        "status": "implemented",
        "evidence": "LLM prompt forbids syndrome/pathomechanism conclusions; deterministic extractor is available for bootstrap.",
    },
    {
        "capability": "Ontology lexicon bootstrap",
        "status": "implemented",
        "evidence": "entities, observation mappings, and linguistic-value mappings are exported as ontology_lexicon.csv for expert curation and audit.",
    },
    {
        "capability": "Bootstrap prior and expert calibration workflow",
        "status": "implemented",
        "evidence": "bootstrap prior config, ICC fields, low-ICC intervals, and expert_calibration_template.csv are generated; expert scores remain external data.",
    },
    {
        "capability": "Overlap-integral membership",
        "status": "implemented",
        "evidence": "Trapezoidal fuzzy-set overlap is used when linguistic_set and target fuzzy_set are configured.",
    },
    {
        "capability": "Low-ICC uncertainty propagation",
        "status": "implemented_mvp",
        "evidence": "Monte Carlo p5/p95 intervals are generated for low-ICC mappings; formal experiments can raise n_samples.",
    },
    {
        "capability": "Observation co-occurrence mining",
        "status": "implemented",
        "evidence": "Itemsets and candidate patterns include support, confidence, lift, PMI, Jaccard, source diversity, and tradition entropy.",
    },
    {
        "capability": "Candidate pattern to expert rule lifecycle",
        "status": "implemented",
        "evidence": "Expert review templates, rule_lifecycle.csv, and optional reviewed rules.csv loading are supported; external UI is not required for the file-based workflow.",
    },
    {
        "capability": "Larsen-style weighted activation inference",
        "status": "implemented",
        "evidence": "Rule activation uses rule_weight multiplied by weighted antecedent memberships.",
    },
    {
        "capability": "Full Mamdani inference",
        "status": "implemented",
        "evidence": "Optional centroid sensitivity analysis truncates consequent fuzzy sets and max-aggregates fired rules.",
    },
    {
        "capability": "Source/tradition/global hierarchical aggregation",
        "status": "implemented",
        "evidence": "Source and tradition weighted means plus cross-tradition discounted noisy-or are implemented with configurable correlation discounts.",
    },
    {
        "capability": "Fuzzy relation network export",
        "status": "implemented",
        "evidence": "Observation→variable, variable→rule, rule→syndrome, and aggregation edges are exported as CSV tables.",
    },
    {
        "capability": "Markdown Wiki auditable display",
        "status": "implemented",
        "evidence": "Source, observation, rule, tradition, synthesis, and audit pages are generated.",
    },
    {
        "capability": "Evaluation metrics FCR/CRP/MIC/SMB/FIA",
        "status": "implemented",
        "evidence": "Metric formulas and gold-standard CSV templates are implemented; rows requiring absent expert labels are marked needs_gold_standard.",
    },
)


def capability_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in CAPABILITY_AUDIT:
        enriched = dict(row)
        enriched["source_count"] = summary.get("source_count", 0)
        enriched["observation_count"] = summary.get("observation_count", 0)
        enriched["coverage"] = summary.get("coverage", 0.0)
        rows.append(enriched)
    return rows


def capability_markdown(summary: dict[str, Any]) -> str:
    table = "\n".join(
        f"| {row['capability']} | {row['status']} | {row['evidence']} |" for row in CAPABILITY_AUDIT
    )
    return f"""# TCM-FuzzyWiki V5.0 实现审计

## 运行摘要

| 指标 | 数值 |
|---|---:|
| Source units | {summary.get('source_count', 0)} |
| Observations | {summary.get('observation_count', 0)} |
| Memberships | {summary.get('membership_count', 0)} |
| Coverage | {summary.get('coverage', 0.0):.3f} |
| Candidate patterns | {summary.get('candidate_pattern_count', 0)} |
| Rules | {summary.get('rule_count', 0)} |
| Wiki pages | {summary.get('wiki_page_count', 0)} |

## 能力状态

| 能力 | 状态 | 说明 |
|---|---|---|
{table}

## 结论

当前代码已经实现 V5.0 的计算链路、可选 Mamdani 敏感性分析、指标公式与审计模板。凡是依赖真实专家评分、外部本体长期维护、UI 审核系统或金标准评测集的数据输入，仍会在输出中显式标注为 `needs_gold_standard` 或 `implemented_mvp`，避免把 bootstrap 近似误写成已完成实验结论。
"""
