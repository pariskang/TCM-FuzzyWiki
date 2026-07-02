"""End-to-end orchestration for TCM-FuzzyWiki V5.0."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from .methodology import methodology_markdown, methodology_rows
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
    observations: list[Any] | None = None,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full deterministic pipeline.

    ``observations`` may be supplied by an external extraction stage (e.g. the
    resumable OpenAI-compatible LLM workflow); the full downstream — membership,
    mining, inference, aggregation, wiki, audit, validation, manifest — then runs
    unchanged, so external extractors can never drift from the main pipeline.
    ``manifest_extra`` is merged into ``run_manifest.json`` for provenance.
    """

    config = load_yaml(config_path)
    out = Path(output_dir)
    data_dir = out / "data"
    wiki_dir = out / "wiki"

    sources = read_chapters(input_path)
    if observations is None:
        extractor = LLMObservationExtractor(AzureChatGPTLLM(AzureChatGPTConfig.from_env())) if use_azure_llm else RuleBasedObservationExtractor()
        observations = extractor.extract(sources)
    observations = ObservationNormalizer(config.get("observation_mapping", {})).normalize(list(observations))
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
    mapping_policy = config.get("mapping_policy", {}) or {}
    if mapping_policy.get("log_unmapped", True):
        write_text(data_dir / "unmapped_observations.log", unmapped_log(observations, float(mapping_policy.get("trigger_review_if_confidence_above", 0.85))))
    write_csv(data_dir / "cooccurrence_stats.csv", patterns)
    write_csv(data_dir / "expert_rule_review.csv", _expert_review_template(patterns))
    write_csv(data_dir / "entities.csv", entities)
    write_csv(data_dir / "observation_mapping.csv", [{"raw": k, "standard_observation": v} for k, v in config.get("observation_mapping", {}).items()])
    write_csv(data_dir / "fuzzy_sets.csv", _flatten_fuzzy_sets(config.get("fuzzy_sets", {})))
    write_csv(data_dir / "relation_nodes.csv", network_nodes)
    write_csv(data_dir / "relation_edges.csv", network_edges)
    write_csv(data_dir / "evaluation_results.csv", evaluation_rows)
    write_csv(data_dir / "evaluation_gold_templates.csv", evaluation_templates)
    write_csv(data_dir / "ontology_lexicon.csv", _ontology_lexicon_rows(config))
    write_csv(data_dir / "expert_calibration_template.csv", _expert_calibration_template(observations, memberships, config))
    write_csv(data_dir / "source_stratification.csv", _source_stratification_rows(sources))
    write_csv(data_dir / "rule_lifecycle.csv", _rule_lifecycle_rows(patterns, rules))

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

    audit_rows = capability_rows(summary, config)
    write_csv(data_dir / "implementation_audit.csv", audit_rows)
    audit_path = wiki_dir / "audit" / "implementation_audit.md"
    write_text(audit_path, capability_markdown(summary, config))
    wiki_pages.append({"page_type": "audit", "page_path": str(audit_path)})
    summary["wiki_page_count"] = len(wiki_pages)
    write_csv(data_dir / "wiki_pages.csv", wiki_pages)
    write_text(out / "summary.txt", "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n")
    methodology_audit_rows = methodology_rows(summary)
    write_csv(data_dir / "methodology_compliance.csv", methodology_audit_rows)
    methodology_path = wiki_dir / "audit" / "methodology_compliance.md"
    write_text(methodology_path, methodology_markdown(methodology_audit_rows))
    wiki_pages.append({"page_type": "audit", "page_path": str(methodology_path)})
    summary["wiki_page_count"] = len(wiki_pages)

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
    if manifest_extra:
        manifest["execution"] = {**manifest.get("execution", {}), **manifest_extra.pop("execution", {})}
        manifest.update(manifest_extra)
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
            "confidence": getattr(pattern, "confidence", ""),
            "lift": pattern.lift,
            "pmi": pattern.pmi,
            "fisher_p": getattr(pattern, "fisher_p", None),
            "jaccard": getattr(pattern, "jaccard", ""),
            "source_count": getattr(pattern, "source_count", ""),
            "book_count": getattr(pattern, "book_count", ""),
            "tradition_count": getattr(pattern, "tradition_count", ""),
            "source_diversity": getattr(pattern, "source_diversity", ""),
            "tradition_entropy": getattr(pattern, "tradition_entropy", ""),
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


def _ontology_lexicon_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in config.get("entities", []):
        row = dict(entity)
        row["lexicon_section"] = "entity"
        rows.append(row)
    for raw, standard in config.get("observation_mapping", {}).items():
        rows.append(
            {
                "lexicon_section": "observation_mapping",
                "raw_term": raw,
                "standard_observation": standard,
                "review_status": "pending",
            }
        )
    for term, entry in config.get("linguistic_values", {}).items():
        maps_to = entry.get("maps_to", {}) if isinstance(entry, dict) else {}
        for variable, mapping in maps_to.items():
            rows.append(
                {
                    "lexicon_section": "linguistic_value",
                    "raw_term": term,
                    "feature": entry.get("feature", "") if isinstance(entry, dict) else "",
                    "variable": variable,
                    "fuzzy_set": mapping.get("fuzzy_set", "") if isinstance(mapping, dict) else "",
                    "prior_membership": mapping.get("prior_membership", "") if isinstance(mapping, dict) else "",
                    "status": mapping.get("status", entry.get("status", "bootstrap_prior")) if isinstance(mapping, dict) and isinstance(entry, dict) else "bootstrap_prior",
                    "icc": mapping.get("icc", "") if isinstance(mapping, dict) else "",
                    "review_status": mapping.get("review_status", entry.get("review_status", "pending")) if isinstance(mapping, dict) and isinstance(entry, dict) else "pending",
                }
            )
    return rows


def _expert_calibration_template(observations: list[Any], memberships: list[Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    obs_by_id = {obs.observation_id: obs for obs in observations}
    rows: list[dict[str, Any]] = []
    for mem in memberships:
        obs = obs_by_id.get(mem.observation_id)
        rows.append(
            {
                "calibration_id": f"CAL_{len(rows) + 1:06d}",
                "source_id": mem.source_id,
                "observation_id": mem.observation_id,
                "feature": getattr(obs, "feature", ""),
                "feature_value": getattr(obs, "feature_value", ""),
                "standard_observation": mem.standard_observation,
                "evidence_text": getattr(obs, "evidence_text", ""),
                "variable": mem.variable,
                "fuzzy_set": mem.fuzzy_set,
                "model_membership": mem.membership,
                "p5": mem.p5,
                "p95": mem.p95,
                "uncertainty_width": mem.uncertainty_width,
                "calculation_mode": mem.calculation_mode,
                "status": mem.status,
                "icc": mem.icc,
                "expert_accept_observation": "",
                "expert_accept_mapping": "",
                "expert_membership": "",
                "expert_comment": "",
            }
        )
    return rows


def _source_stratification_rows(sources: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "topic_hint": lambda source: [source.topic_hint],
        "tradition_id": lambda source: [source.tradition_id],
        "text_family": lambda source: [source.text_family],
        "citation_family": lambda source: [source.citation_family],
        "text_type": lambda source: [source.text_type],
        "school_tag": lambda source: source.school_tag,
        "region_tag": lambda source: source.region_tag,
    }
    for dimension, extractor in dimensions.items():
        counts: Counter[str] = Counter()
        for source in sources:
            for value in extractor(source):
                counts[str(value or "uncertain")] += 1
        for value, count in sorted(counts.items()):
            rows.append({"dimension": dimension, "value": value, "source_count": count})
    return rows


def _rule_lifecycle_rows(patterns: list[Any], rules: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        rows.append(
            {
                "artifact_id": pattern.pattern_id,
                "artifact_type": "candidate_pattern",
                "lifecycle_state": getattr(pattern, "review_status", "candidate_pattern"),
                "created_from": "observation_itemset",
                "next_expected_state": "expert_reviewed_pattern",
                "source_count": getattr(pattern, "source_count", ""),
                "support": getattr(pattern, "support", ""),
            }
        )
    for rule in rules:
        rows.append(
            {
                "artifact_id": rule.rule_id,
                "artifact_type": "rule",
                "lifecycle_state": rule.review_status,
                "created_from": rule.created_from or rule.pattern_id or rule.rule_origin,
                "next_expected_state": "active_rule" if rule.review_status in {"expert_verified", "expert_verified_rule"} else "expert_review_or_monitoring",
                "source_count": rule.source_count,
                "support": rule.support,
            }
        )
    return rows
