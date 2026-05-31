from pathlib import Path

import pandas as pd

from tcm_fuzzywiki.assessment import assess_output


def test_assessment_flags_missing_gold_as_caveat(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([{"severity": "info", "scope": "config"}]).to_csv(data / "validation_report.csv", index=False)
    pd.DataFrame([{"status": "implemented"}]).to_csv(data / "implementation_audit.csv", index=False)
    pd.DataFrame([{"metric": "FCR", "status": "needs_gold_standard"}]).to_csv(data / "evaluation_results.csv", index=False)
    row, markdown = assess_output(tmp_path)
    assert row["verdict"] == "research_ready_with_caveats"
    assert row["metrics_needing_gold_standard"] == 1
    assert "V5.0 完整性评估" in markdown
