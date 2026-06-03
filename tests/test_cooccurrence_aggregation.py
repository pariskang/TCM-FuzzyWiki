from tcm_fuzzywiki.aggregation import aggregate
from tcm_fuzzywiki.cooccurrence import mine_candidate_patterns
from tcm_fuzzywiki.models import EvidenceQuality, FuzzyRule, InferenceResult, Observation, RuleAntecedent, SourceUnit


def test_candidate_patterns_include_sources_and_representative_evidence():
    sources = [
        SourceUnit("SRC_1", "Book A", tradition_id="trad_a"),
        SourceUnit("SRC_2", "Book B", tradition_id="trad_b"),
    ]
    observations = [
        Observation("OBS_1", "SRC_1", "pain_quality", "冷痛", "冷痛", 0.9, "pain_quality:cold", "mapped"),
        Observation("OBS_2", "SRC_1", "relieving_factor", "得温则缓", "得温则缓", 0.9, "relieving_factor:warmth_relieves", "mapped"),
        Observation("OBS_3", "SRC_2", "pain_quality", "冷痛", "冷痛", 0.9, "pain_quality:cold", "mapped"),
        Observation("OBS_4", "SRC_2", "relieving_factor", "得温则缓", "温之稍安", 0.9, "relieving_factor:warmth_relieves", "mapped"),
    ]
    patterns = mine_candidate_patterns(observations, sources, {"candidate_pattern_filter": {"min_support": 0.1, "min_lift": 0.1, "min_pmi": -10, "min_source_count": 1}})
    assert patterns
    assert patterns[0].source_ids
    assert patterns[0].representative_evidence


def test_aggregation_records_tradition_delta_values():
    sources = [
        SourceUnit("SRC_1", "Book A", tradition_id="trad_a", evidence_quality=EvidenceQuality(1, 1, 1)),
        SourceUnit("SRC_2", "Book B", tradition_id="trad_b", evidence_quality=EvidenceQuality(1, 1, 1)),
    ]
    rules = [FuzzyRule("RULE_1", "rule", "seed", [RuleAntecedent("x", "high")], "寒湿痹阻", rule_weight=1.0)]
    results = [
        InferenceResult("SRC_1", "RULE_1", "寒湿痹阻", "syndrome", 0.5, ["x.high"], [], "fired"),
        InferenceResult("SRC_2", "RULE_1", "寒湿痹阻", "syndrome", 0.5, ["x.high"], [], "fired"),
    ]
    rows = aggregate(results, sources, rules, {"tradition_independence_weights": {"trad_b": 0.5}})
    global_row = next(row for row in rows if row["aggregation_level"] == "global")
    assert global_row["delta_values"] == [1.0, 0.5]


def test_candidate_mining_does_not_prune_larger_itemsets_by_pair_lift():
    sources = [SourceUnit(f"SRC_{idx:03d}", "Book", tradition_id="trad") for idx in range(100)]
    observations = []
    for idx, source in enumerate(sources):
        items = ["item:dummy"]
        if idx < 80:
            items.extend(["item:a", "item:b"])
        if idx < 20:
            items.append("item:c")
        for item_idx, item in enumerate(items):
            observations.append(Observation(f"OBS_{idx}_{item_idx}", source.source_id, "item", item, item, 0.9, item, "mapped"))

    patterns = mine_candidate_patterns(
        observations,
        sources,
        {
            "candidate_pattern_filter": {
                "max_size": 3,
                "min_support": 0.1,
                "min_source_count": 10,
                "min_tradition_count": 1,
                "min_lift": 1.3,
                "min_pmi": 0,
            }
        },
    )

    assert any(pattern.observations == ("item:a", "item:b", "item:c") for pattern in patterns)


def test_candidate_pattern_source_ids_are_truncated_with_summary():
    sources = [SourceUnit(f"SRC_{idx:03d}", "Book", tradition_id="trad") for idx in range(40)]
    observations = []
    for idx, source in enumerate(sources):
        for item_idx, item in enumerate(["item:a", "item:b"]):
            observations.append(Observation(f"OBS_{idx}_{item_idx}", source.source_id, "item", item, item, 0.9, item, "mapped"))

    patterns = mine_candidate_patterns(
        observations,
        sources,
        {
            "candidate_pattern_filter": {
                "min_support": 0.1,
                "min_source_count": 10,
                "min_tradition_count": 1,
                "min_lift": 0.1,
                "min_pmi": -10,
                "max_source_ids_per_pattern": 5,
            }
        },
    )

    assert patterns
    assert patterns[0].source_count == 40
    assert len(patterns[0].source_ids) == 5
    assert "source_ids_truncated_to=5" in patterns[0].source_count_summary
