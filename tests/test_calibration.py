from pathlib import Path

import pandas as pd

from tcm_fuzzywiki.calibration import calibrate_config_from_experts, panel_icc, write_calibrated_config
from tcm_fuzzywiki.config import load_yaml


def test_calibrate_config_from_expert_scores(tmp_path: Path):
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E1", "score": 0.9},
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E2", "score": 0.8},
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E3", "score": 0.85},
        ]
    ).to_csv(scores, index=False)
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    calibrated, report = calibrate_config_from_experts(config, scores)
    mapping = calibrated["linguistic_values"]["冷痛"]["maps_to"]["cold_property"]
    assert mapping["calibrated_membership"] == 0.85
    assert mapping["review_status"] == "expert_reviewed"
    assert report[0]["expert_count"] == 3
    out_config = tmp_path / "calibrated.yaml"
    out_report = tmp_path / "report.csv"
    write_calibrated_config(calibrated, report, out_config, out_report)
    assert out_config.exists()
    assert out_report.exists()


def test_panel_icc_requires_multiple_items_and_experts():
    single_item = pd.DataFrame(
        [
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E1", "score": 0.9},
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E2", "score": 0.8},
        ]
    )
    assert panel_icc(single_item) is None


def test_panel_icc_separates_agreement_levels():
    def frame(noise: float) -> pd.DataFrame:
        rows = []
        for item, base in enumerate((0.1, 0.5, 0.9)):
            for rater, offset in enumerate((-noise, 0.0, noise)):
                rows.append(
                    {
                        "term": f"T{item}",
                        "variable": "v",
                        "fuzzy_set": "high",
                        "expert_id": f"E{rater}",
                        "score": min(1.0, max(0.0, base + offset)),
                    }
                )
        return pd.DataFrame(rows)

    high_agreement = panel_icc(frame(0.01))
    low_agreement = panel_icc(frame(0.4))
    assert high_agreement is not None and low_agreement is not None
    assert high_agreement > 0.9
    assert low_agreement < high_agreement


def test_panel_icc_perfect_agreement_is_one():
    rows = [
        {"term": term, "variable": "v", "fuzzy_set": "high", "expert_id": expert, "score": 0.5}
        for term in ("T1", "T2")
        for expert in ("E1", "E2")
    ]
    assert panel_icc(pd.DataFrame(rows)) == 1.0


def test_calibration_report_carries_icc_type_and_panel_icc(tmp_path: Path):
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E1", "score": 0.9},
            {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "expert_id": "E2", "score": 0.8},
            {"term": "刺痛", "variable": "blood_stasis_tendency", "fuzzy_set": "moderate", "expert_id": "E1", "score": 0.4},
            {"term": "刺痛", "variable": "blood_stasis_tendency", "fuzzy_set": "moderate", "expert_id": "E2", "score": 0.5},
        ]
    ).to_csv(scores, index=False)
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    _, report = calibrate_config_from_experts(config, scores)
    assert all(row["icc_type"] == "expert_agreement_proxy" for row in report)
    assert all(row["panel_icc"] is not None for row in report)
