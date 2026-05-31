from pathlib import Path

from tcm_fuzzywiki.rules import load_rules


def test_review_csv_pending_rows_are_skipped_and_accepted_rows_load(tmp_path: Path):
    csv_path = tmp_path / "reviewed_rules.csv"
    csv_path.write_text(
        "pattern_id,expert_decision,antecedents_json,suggested_consequent,suggested_rule_weight,review_status\n"
        "PATTERN_1,pending,[],寒湿痹阻,0.8,pending\n"
        'PATTERN_2,accepted,"[{""variable"":""cold_property"",""fuzzy_set"":""high"",""threshold"":0.7,""weight"":1.0}]",寒湿痹阻,0.8,active_rule\n',
        encoding="utf-8",
    )
    rules = load_rules({"seed_rules": []}, csv_path)
    assert len(rules) == 1
    assert rules[0].rule_id == "RULE_FROM_PATTERN_2"
    assert rules[0].consequent_entity == "寒湿痹阻"
