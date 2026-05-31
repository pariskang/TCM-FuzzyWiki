"""Markdown Wiki generation for auditable fuzzy inference results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import write_text
from .models import FuzzyRule, InferenceResult, Membership, Observation, SourceUnit

DISCLAIMER = "本页 μ 值为基于古籍文本、语言变量映射、模糊规则和专家校准得到的形式化近似结果，不等同于唯一临床诊断。"


def generate_wiki(
    out_dir: str | Path,
    sources: list[SourceUnit],
    observations: list[Observation],
    memberships: list[Membership],
    inference_results: list[InferenceResult],
    rules: list[FuzzyRule],
    aggregations: list[dict[str, Any]],
    mamdani_results: list[dict[str, Any]] | None = None,
    evaluation_rows: list[dict[str, Any]] | None = None,
    patterns: list[Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    out = Path(out_dir)
    pages: list[dict[str, str]] = []
    for subdir in ["sources", "observations", "entities", "syndromes", "mechanisms", "methods", "formulas", "herbs", "traditions", "patterns", "rules", "synthesis", "audit"]:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    by_source_obs = _group(observations, lambda obs: obs.source_id)
    by_source_mem = _group(memberships, lambda mem: mem.source_id)
    by_source_inf = _group(inference_results, lambda res: res.source_id)

    for source in sources:
        path = out / "sources" / f"{source.source_id}.md"
        text = _source_page(source, by_source_obs.get(source.source_id, []), by_source_mem.get(source.source_id, []), by_source_inf.get(source.source_id, []))
        write_text(path, text)
        pages.append({"page_type": "source", "page_path": str(path)})

    for rule in rules:
        path = out / "rules" / f"{rule.rule_id}.md"
        write_text(path, _rule_page(rule))
        pages.append({"page_type": "rule", "page_path": str(path)})

    for pattern in patterns or []:
        path = out / "patterns" / f"{pattern.pattern_id}.md"
        write_text(path, _pattern_page(pattern))
        pages.append({"page_type": "pattern", "page_path": str(path)})

    entity_lookup = _entity_lookup(entities or [])
    for entity in entities or []:
        folder = _entity_folder(entity)
        safe_name = _safe_filename(str(entity.get("entity_id") or entity.get("entity_name") or "entity"))
        path = out / folder / f"{safe_name}.md"
        write_text(path, _entity_page(entity, aggregations, rules))
        pages.append({"page_type": folder.rstrip('s'), "page_path": str(path)})

    for conclusion in sorted({result.consequent_entity for result in inference_results} | {row.get("consequent_entity", "") for row in aggregations if row.get("consequent_entity")}):
        if not conclusion:
            continue
        safe_name = _safe_filename(conclusion)
        path = out / "syndromes" / f"{safe_name}.md"
        write_text(path, _syndrome_page(conclusion, entity_lookup.get(conclusion), inference_results, aggregations, rules))
        pages.append({"page_type": "syndrome", "page_path": str(path)})

    for observation_key, rows in _group(observations, lambda obs: obs.standard_observation or obs.feature_value).items():
        safe_name = _safe_filename(observation_key)
        path = out / "observations" / f"{safe_name}.md"
        related = [mem for mem in memberships if mem.standard_observation == observation_key]
        write_text(path, _observation_page(observation_key, rows, related))
        pages.append({"page_type": "observation", "page_path": str(path)})

    for tradition_id, rows in _group(sources, lambda src: src.tradition_id).items():
        path = out / "traditions" / f"{tradition_id}.md"
        write_text(path, _tradition_page(tradition_id, rows, aggregations))
        pages.append({"page_type": "tradition", "page_path": str(path)})

    synthesis_path = out / "synthesis" / "global_syndrome_spectrum.md"
    write_text(synthesis_path, _synthesis_page(aggregations))
    pages.append({"page_type": "synthesis", "page_path": str(synthesis_path)})

    index = _index_page(sources, rules, aggregations)
    write_text(out / "index.md", index)
    pages.append({"page_type": "index", "page_path": str(out / "index.md")})
    mamdani_path = out / "audit" / "mamdani_sensitivity.md"
    write_text(mamdani_path, _mamdani_page(mamdani_results or []))
    pages.append({"page_type": "audit", "page_path": str(mamdani_path)})

    evaluation_path = out / "audit" / "evaluation_metrics.md"
    write_text(evaluation_path, _evaluation_page(evaluation_rows or []))
    pages.append({"page_type": "audit", "page_path": str(evaluation_path)})

    write_text(out / "audit" / "formal_boundary.md", f"# 形式化边界\n\n{DISCLAIMER}\n")
    pages.append({"page_type": "audit", "page_path": str(out / "audit" / "formal_boundary.md")})
    return pages


def _group(items: list[Any], key_fn: Any) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        grouped[str(key_fn(item))].append(item)
    return dict(grouped)


def _source_page(source: SourceUnit, observations: list[Observation], memberships: list[Membership], inference_results: list[InferenceResult]) -> str:
    fired = [row for row in inference_results if row.status == "fired"]
    obs_rows = "\n".join(
        f"| {obs.observation_id} | {obs.feature} | {obs.standard_observation} | {obs.extraction_confidence:.2f} | {obs.evidence_text} |"
        for obs in observations
    ) or "| - | - | - | - | - |"
    mem_rows = "\n".join(
        f"| {mem.standard_observation} | {mem.variable}.{mem.fuzzy_set} | {mem.membership:.2f} | {mem.calculation_mode} |"
        for mem in memberships
    ) or "| - | - | - | - |"
    inf_rows = "\n".join(
        f"| {res.consequent_entity} | {res.activation:.2f} | {res.p5 if res.p5 is not None else '-'} | {res.p95 if res.p95 is not None else '-'} | {res.rule_id} |"
        for res in fired
    ) or "| - | - | - | - | - |"
    return f"""# 《{source.book_name}》{source.volume_name}：{source.chapter_title}

## 基本信息
- Source ID：{source.source_id}
- 书名：{source.book_name}
- 类型：{source.text_type}
- 学术传统：{source.tradition_id}
- 学派：{', '.join(source.school_tag)}
- 地域：{', '.join(source.region_tag)}
- 主题：{source.topic_hint}

## 原文
{source.original_text}

## Observation
| Observation | 特征 | 标准值 | 置信度 | 原文证据 |
|---|---|---|---:|---|
{obs_rows}

## Fuzzy Membership
| Observation | Fuzzy variable | μ | 计算方式 |
|---|---|---:|---|
{mem_rows}

## Inference
| 结论 | μ | p5 | p95 | 规则 |
|---|---:|---:|---:|---|
{inf_rows}

## 形式化说明
{DISCLAIMER}
"""


def _rule_page(rule: FuzzyRule) -> str:
    antecedents = "\n".join(
        f"| {ant.variable} | {ant.fuzzy_set} | {ant.threshold:.2f} | {ant.weight:.2f} |" for ant in rule.antecedents
    )
    return f"""# {rule.rule_id}：{rule.rule_name}

## 规则来源
- 来源类型：{rule.rule_origin}
- Pattern ID：{rule.pattern_id}
- 专家审核状态：{rule.review_status}

## Antecedents
| Variable | Fuzzy set | Threshold | Weight |
|---|---|---:|---:|
{antecedents}

## Consequent
{rule.consequent_entity}

## 统计依据
| 指标 | 数值 |
|---|---:|
| Support | {rule.support:.3f} |
| Confidence | {rule.confidence:.3f} |
| Lift | {rule.lift:.3f} |
| PMI | {rule.pmi:.3f} |
| Source count | {rule.source_count} |
| Tradition count | {rule.tradition_count} |

## 专家意见
{rule.expert_disagreement_note or rule.applicable_context or '待补充。'}
"""


def _safe_filename(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _entity_lookup(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entity in entities:
        name = str(entity.get("entity_name", ""))
        normalized = str(entity.get("normalized_name", ""))
        if name:
            lookup[name] = entity
        if normalized:
            lookup[normalized] = entity
    return lookup


def _entity_folder(entity: dict[str, Any]) -> str:
    entity_type = str(entity.get("entity_type", ""))
    if "方" in entity_type:
        return "formulas"
    if "药" in entity_type:
        return "herbs"
    if "证" in entity_type:
        return "syndromes"
    if "病机" in entity_type:
        return "mechanisms"
    if "治法" in entity_type or "方法" in entity_type:
        return "methods"
    return "entities"


def _entity_page(entity: dict[str, Any], aggregations: list[dict[str, Any]], rules: list[FuzzyRule]) -> str:
    name = str(entity.get("entity_name", entity.get("normalized_name", "未命名实体")))
    related_agg = [row for row in aggregations if row.get("consequent_entity") == name]
    agg_rows = "\n".join(
        f"| {row.get('aggregation_level', '')} | {row.get('mu', '')} | {row.get('evidence_count', '')} | {row.get('tradition_count', row.get('source_count', ''))} |"
        for row in related_agg
    ) or "| - | - | - | - |"
    related_rules = [rule for rule in rules if rule.consequent_entity == name]
    rule_rows = "\n".join(f"| {rule.rule_id} | {rule.rule_name} | {rule.rule_weight:.2f} | {rule.review_status} |" for rule in related_rules) or "| - | - | - | - |"
    return f"""# {name}

## 基本信息
- Entity ID：{entity.get('entity_id', '')}
- 类型：{entity.get('entity_type', '')}
- 规范名：{entity.get('normalized_name', '')}
- 同义词：{entity.get('synonyms', '')}
- 古籍术语：{entity.get('ancient_terms', '')}
- 现代映射：{entity.get('modern_mapping', '')}
- 上位类：{entity.get('parent_category', '')}
- 审核状态：{entity.get('review_status', '')}

## 定义
{entity.get('definition', '')}

## 聚合证据
| Level | μ | Evidence count | Scope count |
|---|---:|---:|---:|
{agg_rows}

## 相关规则
| Rule | Name | Weight | Status |
|---|---|---:|---|
{rule_rows}

## 形式化边界
{DISCLAIMER}
"""


def _syndrome_page(
    syndrome: str,
    entity: dict[str, Any] | None,
    inference_results: list[InferenceResult],
    aggregations: list[dict[str, Any]],
    rules: list[FuzzyRule],
) -> str:
    related_results = [row for row in inference_results if row.consequent_entity == syndrome and row.status == "fired"]
    result_rows = "\n".join(
        f"| {row.source_id} | {row.activation:.2f} | {row.rule_id} | {', '.join(row.supporting_variables)} |"
        for row in related_results
    ) or "| - | - | - | - |"
    related_agg = [row for row in aggregations if row.get("consequent_entity") == syndrome]
    agg_rows = "\n".join(
        f"| {row.get('aggregation_level', '')} | {row.get('mu', '')} | {row.get('evidence_count', '')} |"
        for row in related_agg
    ) or "| - | - | - |"
    related_rules = [rule for rule in rules if rule.consequent_entity == syndrome]
    rule_rows = "\n".join(f"- [[{rule.rule_id}]]：{rule.rule_name}" for rule in related_rules) or "- 暂无"
    definition = entity.get('definition', '') if entity else ''
    modern = entity.get('modern_mapping', '') if entity else ''
    return f"""# {syndrome}

## 定义与映射
- 定义：{definition}
- 现代映射：{modern}

## 激活来源
| Source | Activation | Rule | Supporting variables |
|---|---:|---|---|
{result_rows}

## 分层聚合
| Level | μ | Evidence count |
|---|---:|---:|
{agg_rows}

## 相关规则
{rule_rows}

## 形式化边界
{DISCLAIMER}
"""


def _pattern_page(pattern: Any) -> str:
    obs_rows = "\n".join(f"- `{item}`" for item in pattern.observations)
    evidence_rows = "\n".join(f"- {item}" for item in pattern.representative_evidence) or "- 待补充"
    interpretation_rows = "、".join(pattern.possible_interpretation) or "待专家判断"
    return f"""# {pattern.pattern_id}

## Observation 组合
{obs_rows}

## 统计指标
| 指标 | 数值 |
|---|---:|
| Support | {pattern.support:.3f} |
| Confidence | {pattern.confidence:.3f} |
| Lift | {pattern.lift:.3f} |
| PMI | {pattern.pmi:.3f} |
| Jaccard | {pattern.jaccard:.3f} |
| Source count | {pattern.source_count} |
| Book count | {pattern.book_count} |
| Tradition count | {pattern.tradition_count} |
| Source diversity | {pattern.source_diversity:.3f} |
| Tradition entropy | {pattern.tradition_entropy:.3f} |

## 来源范围
- Sources：{', '.join(pattern.source_ids)}
- Books：{', '.join(pattern.book_names)}
- Traditions：{', '.join(pattern.tradition_ids)}

## 代表证据
{evidence_rows}

## 系统建议解释
{interpretation_rows}

## 专家审核状态
{pattern.status}
"""


def _observation_page(observation_key: str, observations: list[Observation], memberships: list[Membership]) -> str:
    mem_rows = "\n".join(
        sorted({f"| {mem.variable}.{mem.fuzzy_set} | {mem.membership:.2f} | {mem.status} |" for mem in memberships})
    ) or "| - | - | - |"
    return f"""# {observation_key}

## 类型
Observation

## 映射变量
| Fuzzy variable | μ | 状态 |
|---|---:|---|
{mem_rows}

## 证据计数
- 出现次数：{len(observations)}
- 来源章节：{len({obs.source_id for obs in observations})}
"""


def _tradition_page(tradition_id: str, sources: list[SourceUnit], aggregations: list[dict[str, Any]]) -> str:
    rows = [row for row in aggregations if row.get("aggregation_level") == "tradition" and row.get("tradition_id") == tradition_id]
    agg_rows = "\n".join(f"| {row['consequent_entity']} | {row['mu']:.2f} | {row['evidence_count']} |" for row in rows) or "| - | - | - |"
    books = "\n".join(f"- 《{source.book_name}》：{source.chapter_title}" for source in sources)
    return f"""# {tradition_id}

## 代表文献
{books}

## 证候倾向
| 证候 | 综合 μ | 证据数 |
|---|---:|---:|
{agg_rows}

## 形式化边界
{DISCLAIMER}
"""


def _mamdani_page(rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        f"| {row['source_id']} | {row['consequent_entity']} | {row['centroid']:.2f} | {row['max_membership']:.2f} | {row['area']:.3f} | {', '.join(row.get('rules', []))} |"
        for row in rows
    ) or "| - | - | - | - | - | - |"
    return f"""# Mamdani 敏感性分析

## 说明
本页为可选验证模块：使用 Larsen-style 推理得到的 fired rule activation 截断 consequent fuzzy set，多规则 max 聚合后计算质心。大规模 Wiki 默认仍以 Larsen-style activation 作为主结果。

| Source | Consequent | Centroid | Max μ | Area | Rules |
|---|---|---:|---:|---:|---|
{table}

## 形式化边界
{DISCLAIMER}
"""


def _evaluation_page(rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        f"| {row['metric']} | {row.get('value', '')} | {row.get('n', 0)} | {row.get('status', '')} | {row.get('required_gold_file', '')} |"
        for row in rows
    ) or "| - | - | - | - | - |"
    return f"""# 评估指标

## 指标结果
| Metric | Value | N | Status | Required gold file |
|---|---:|---:|---|---|
{table}

## 说明
FCR、CRP、MIC、SMB、FIA-local 与 FIA-chain 的公式已实现。若未提供专家金标准 CSV，相关指标会标记为 `needs_gold_standard`；系统只输出可复算模板与 proxy，不伪造专家评估结果。
"""


def _synthesis_page(aggregations: list[dict[str, Any]]) -> str:
    global_rows = sorted(
        [row for row in aggregations if row.get("aggregation_level") == "global"],
        key=lambda row: row.get("mu", 0.0),
        reverse=True,
    )
    table = "\n".join(
        f"| {row['consequent_entity']} | {row['mu']:.2f} | {row['evidence_count']} | {row['tradition_count']} |"
        for row in global_rows
    ) or "| - | - | - | - |"
    return f"""# 全局证候谱系

## 综合结论
当前资料显示，系统保留多证候并列解释，不将章节或主题强行归为单一证候。

## 全局证候分布
| 证候 | μ_global | 证据数 | 传统数 |
|---|---:|---:|---:|
{table}

## 形式化边界
{DISCLAIMER}
"""


def _index_page(sources: list[SourceUnit], rules: list[FuzzyRule], aggregations: list[dict[str, Any]]) -> str:
    global_rows = [row for row in aggregations if row.get("aggregation_level") == "global"]
    table = "\n".join(f"| {row['consequent_entity']} | {row['mu']:.2f} | {row['evidence_count']} | {row['tradition_count']} |" for row in global_rows) or "| - | - | - | - |"
    return f"""# TCM-FuzzyWiki V5.0

## 概览
- 章节级证据单元：{len(sources)}
- 活动规则：{len(rules)}

## 全局证候分布
| 证候 | μ_global | 证据数 | 传统数 |
|---|---:|---:|---:|
{table}

## 方法学
XLSX 古籍章节 → Source Evidence Unit → Observation → Bootstrap Prior → Overlap Integral → 共现模式 → 专家规则 → Larsen-style 推理 → 分层聚合 → Markdown Wiki。

> {DISCLAIMER}
"""
