from pathlib import Path

import pandas as pd

from tcm_fuzzywiki.evaluation import evaluate_pipeline
from tcm_fuzzywiki.inference import infer
from tcm_fuzzywiki.mamdani import run_mamdani_sensitivity
from tcm_fuzzywiki.models import FuzzyRule, Membership, RuleAntecedent


def _fixture():
    memberships = [
        Membership("MEM_1", "OBS_1", "SRC_1", "pain_quality:cold", "cold_property", "high", 0.9, "overlap_integral"),
    ]
    rules = [
        FuzzyRule(
            rule_id="RULE_1",
            rule_name="cold rule",
            rule_origin="seed_rule",
            antecedents=[RuleAntecedent("cold_property", "high", 0.7, 1.0)],
            consequent_entity="寒湿痹阻",
            rule_weight=0.8,
            review_status="active_rule",
        )
    ]
    inference_results = infer(memberships, rules)
    return memberships, rules, inference_results


def test_mamdani_sensitivity_outputs_centroid():
    _, rules, inference_results = _fixture()
    rows = run_mamdani_sensitivity(inference_results, rules, {"mamdani_sensitivity": {"enabled": True, "points": 100}})
    assert rows
    assert rows[0]["source_id"] == "SRC_1"
    assert 0.0 <= rows[0]["centroid"] <= 1.0
    assert rows[0]["max_membership"] > 0.0


def test_evaluation_uses_gold_labels_when_available(tmp_path: Path):
    memberships, _, inference_results = _fixture()
    pd.DataFrame(
        [
            {
                "source_id": "SRC_1",
                "standard_observation": "pain_quality:cold",
                "variable": "cold_property",
                "fuzzy_set": "high",
                "expert_membership": 0.85,
            }
        ]
    ).to_csv(tmp_path / "expert_memberships.csv", index=False)
    rows, templates = evaluate_pipeline(memberships, inference_results, [], tmp_path)
    fcr = next(row for row in rows if row["metric"] == "FCR")
    assert fcr["status"] == "computed"
    assert fcr["value"] == 0.95
    assert any(template["file_name"] == "expert_memberships.csv" for template in templates)
