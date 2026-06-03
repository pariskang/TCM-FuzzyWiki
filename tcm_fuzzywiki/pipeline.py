"""End-to-end orchestration for TCM-FuzzyWiki V5.0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .aggregation import aggregate
from .assessment import assess_output
from .audit import capability_markdown, capability_rows
from .cooccurrence import itemsets_by_source, mine_candidate_patterns
from .config import load_yaml
from .evaluation import evaluate_pipeline
from .extraction import LLMObservationExtractor, ObservationNormalizer, RuleBasedObservationExtractor
from .inference import infer
from .io import read_chapters, write_csv, write_json, write_text
from .llmlite import AzureChatGPTConfig, AzureChatGPTLLM
from .mamdani import run_mamdani_sensitivity
from .membership import MembershipCalculator, coverage, unmapped_log
from .network import build_relation_network
from .provenance import build_manifest, manifest_markdown
from .rules import load_rules
from .validation import readiness_markdown, validate_build
from .wiki import generate_wiki


def run_pipeline(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    use_azure_llm: bool = False,
    rules_csv: str | Path | None = None,
    gold_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    out = Path(output_dir)
    data_dir = out / "data"
    wiki_dir = out / "wiki"

    sources = read_chapters(input_path)
    extractor = LLMObservationExtractor(AzureChatGPTLLM(AzureChatGPTConfig.from_env())) if use_azure_llm else RuleBasedObservationExtractor()
    observations = extractor.extract(sources)
    observations = ObservationNormalizer(config.get("observation_mapping", {})).normalize(observations)
    memberships = MembershipCalculator(config).compute(observations)
    patterns = mine_candidate_patterns(observations, sources, config)
    rules = load_rules(config, rules_csv)
    inference_results = infer(memberships, rules)
    mamdani_results = run_mamdani_sensitivity(inference_results, rules, config)
    aggregations = aggregate(inference_results, sources, rules, config.get("aggregation", {}))
    network_nodes, network_edges = build_relation_network(observations, memberships, rules, inference_results, aggregations)
    entities = config.get("entities", [])
    evaluation_rows, evaluation_templates = evaluate_pipeline(memberships, inference_results, entities, gold_dir)
    wiki_pages = generate_wiki(wiki_dir, sources, observations, memberships, inference_results, rules, aggregations, mamdani_results, evaluation_rows, patterns, entities, config)

    source_metadata = [
        {
            "source_id": src.source_id,
            "school_tag": src.school_tag,
            "region_tag": src.region_tag,
            "tradition_id": src.tradition_id,
            "text_family": src.text_family,
            "citation_family": src.citation_family,
            "source_authority": src.evidence_quality.source_authority,
            "text_integrity": src.evidence_quality.text_integrity,
            "semantic_clarity": src.evidence_quality.semantic_clarity,
        }
        for src in sources
    ]
    itemset_rows = [{"source_id": sid, "observations": sorted(items)} for sid, items in itemsets_by_source(observations).items()]

    write_csv(data_dir / "source_units.csv", sources)
    write_csv(data_dir / "source_metadata.csv", source_metadata)
    write_csv(data_dir / "observations.csv", observations)
    write_csv(data_dir / "memberships.csv", memberships)
    write_csv(data_dir / "observation_itemsets.csv", itemset_rows)
    write_csv(data_dir / "candidate_patterns.csv", patterns)
    write_csv(data_dir / "rules.csv", rules)
    write_csv(data_dir / "inference_results.csv", inference_results)
    write_csv(data_dir / "mamdani_results.csv", mamdani_results)
    write_csv(data_dir / "aggregation_results.csv", aggregations)
    write_csv(data_dir / "wiki_pages.csv", wiki_pages)
    write_text(data_dir / "unmapped_observations.log", unmapped_log(observations, config.get("mapping_policy", {}).get("trigger_review_if_confidence_above", 0.85)))
    write_csv(data_dir / "cooccurrence_stats.csv", patterns)
    write_csv(data_dir / "expert_rule_review.csv", _expert_review_template(patterns))
    write_csv(data_dir / "entities.csv", entities)
    write_csv(data_dir / "observation_mapping.csv", [{"raw": k, "standard_observation": v} for k, v in config.get("observation_mapping", {}).items()])
    write_csv(data_dir / "fuzzy_sets.csv", _flatten_fuzzy_sets(config.get("fuzzy_sets", {})))
    write_csv(data_dir / "relation_nodes.csv", network_nodes)
    write_csv(data_dir / "relation_edges.csv", network_edges)
    write_csv(data_dir / "evaluation_results.csv", evaluation_rows)
    write_csv(data_dir / "evaluation_gold_templates.csv", evaluation_templates)

    cov = coverage(observations, memberships)
    summary = {
        "source_count": len(sources),
        "observation_count": len(observations),
        "membership_count": len(memberships),
        "coverage": round(cov, 6),
        "candidate_pattern_count": len(patterns),
        "rule_count": len(rules),
        "inference_result_count": len(inference_results),
        "mamdani_result_count": len(mamdani_results),
        "evaluation_metric_count": len(evaluation_rows),
        "wiki_page_count": len(wiki_pages),
    }
    validation_rows = validate_build(config, sources, observations, memberships, rules, evaluation_rows, summary)
    write_csv(data_dir / "validation_report.csv", validation_rows)
    validation_path = wiki_dir / "audit" / "validation_report.md"
    write_text(validation_path, readiness_markdown(validation_rows))
    wiki_pages.append({"page_type": "audit", "page_path": str(validation_path)})

    audit_rows = capability_rows(summary)
    write_csv(data_dir / "implementation_audit.csv", audit_rows)
    audit_path = wiki_dir / "audit" / "implementation_audit.md"
    write_text(audit_path, capability_markdown(summary))
    wiki_pages.append({"page_type": "audit", "page_path": str(audit_path)})
    summary["wiki_page_count"] = len(wiki_pages)
    write_csv(data_dir / "wiki_pages.csv", wiki_pages)
    write_text(out / "summary.txt", "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n")
    assessment_row, assessment_md = assess_output(out)
    write_csv(data_dir / "completion_assessment.csv", [assessment_row])
    assessment_path = wiki_dir / "audit" / "completion_assessment.md"
    write_text(assessment_path, assessment_md)
    wiki_pages.append({"page_type": "audit", "page_path": str(assessment_path)})
    summary["completion_verdict"] = assessment_row["verdict"]
    summary["wiki_page_count"] = len(wiki_pages)

    manifest = build_manifest(
        input_path=input_path,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        summary=summary,
        use_azure_llm=use_azure_llm,
        rules_csv=rules_csv,
        gold_dir=gold_dir,
    )
    write_json(data_dir / "run_manifest.json", manifest)
    manifest_path = wiki_dir / "audit" / "run_manifest.md"
    write_text(manifest_path, manifest_markdown(manifest))
    wiki_pages.append({"page_type": "audit", "page_path": str(manifest_path)})
    summary["wiki_page_count"] = len(wiki_pages)
    manifest["summary"] = summary
    write_json(data_dir / "run_manifest.json", manifest)
    write_csv(data_dir / "wiki_pages.csv", wiki_pages)
    write_text(out / "summary.txt", "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n")
    return summary


def _expert_review_template(patterns: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "pattern_id": pattern.pattern_id,
            "observations": list(pattern.observations),
            "itemset": getattr(pattern, "itemset", list(pattern.observations)),
            "size": getattr(pattern, "size", len(pattern.observations)),
            "support": pattern.support,
            "lift": pattern.lift,
            "pmi": pattern.pmi,
            "fisher_p": getattr(pattern, "fisher_p", None),
            "possible_interpretation": pattern.possible_interpretation,
            "source_ids": pattern.source_ids,
            "source_count_summary": getattr(pattern, "source_count_summary", ""),
            "mapping_status_summary": getattr(pattern, "mapping_status_summary", {}),
            "book_names": pattern.book_names,
            "tradition_ids": pattern.tradition_ids,
            "representative_evidence": pattern.representative_evidence,
            "expert_decision": "pending",
            "antecedents_json": "[]",
            "suggested_consequent": "",
            "suggested_rule_weight": "",
            "applicable_context": "",
            "conflict_note": "",
            "review_status": getattr(pattern, "review_status", "pending"),
        }
        for pattern in patterns
    ]


def _flatten_fuzzy_sets(fuzzy_sets: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable, sets in fuzzy_sets.items():
        for fuzzy_set, spec in sets.items():
            rows.append({"variable": variable, "fuzzy_set": fuzzy_set, "points": spec.get("points", spec) if isinstance(spec, dict) else spec})
    return rows
