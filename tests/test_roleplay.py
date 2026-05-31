from tcm_fuzzywiki.config import load_yaml
from tcm_fuzzywiki.roleplay import (
    DeterministicRoleplayScorer,
    LLMRoleplayExpertScorer,
    build_scoring_tasks,
    calibrate_from_roleplay_scores,
)


class FakeRoleplayLLM:
    def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        return {
            "scores": [
                {"term": "冷痛", "variable": "cold_property", "fuzzy_set": "high", "score": 0.91, "confidence": 0.88, "rationale": "寒性明确"}
            ]
        }


def test_build_scoring_tasks_from_config():
    tasks = build_scoring_tasks(load_yaml("configs/tcm_fuzzywiki.yaml"))
    assert any(task["term"] == "冷痛" and task["variable"] == "cold_property" for task in tasks)


def test_llm_roleplay_scorer_generates_auditable_rows():
    config = {"linguistic_values": {"冷痛": {"feature": "pain_quality", "maps_to": {"cold_property": {"fuzzy_set": "high", "prior_membership": 0.85}}}}}
    rows = LLMRoleplayExpertScorer(FakeRoleplayLLM()).score_config(config)
    assert len(rows) == 3
    scored = [row for row in rows if row["score"] == 0.91]
    assert scored
    assert scored[0]["score_source"] == "llm_roleplay"
    assert "expert_role" in scored[0]


def test_deterministic_roleplay_scores_can_calibrate(tmp_path):
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    rows = DeterministicRoleplayScorer().score_config(config)
    scores = tmp_path / "roleplay_scores.csv"
    output_config = tmp_path / "roleplay_calibrated.yaml"
    report = tmp_path / "roleplay_report.csv"
    calibrated, calibration_report = calibrate_from_roleplay_scores(config, rows, scores, output_config, report)
    assert scores.exists()
    assert output_config.exists()
    assert report.exists()
    assert calibration_report
    assert calibrated is not None
    mapping = calibrated["linguistic_values"]["冷痛"]["maps_to"]["cold_property"]
    assert mapping["calibration_source"] == "llm_roleplay_panel"
    assert mapping["review_status"] == "llm_roleplay_reviewed"
