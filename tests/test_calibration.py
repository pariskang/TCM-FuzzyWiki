from pathlib import Path

import pandas as pd

from tcm_fuzzywiki.calibration import calibrate_config_from_experts, write_calibrated_config
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
