from pathlib import Path

from tcm_fuzzywiki.pipeline import run_pipeline


def test_demo_pipeline_builds_auditable_outputs(tmp_path: Path):
    summary = run_pipeline("examples/bootstrap_chapters.csv", "configs/tcm_fuzzywiki.yaml", tmp_path)
    assert summary["source_count"] == 5
    assert summary["observation_count"] > 0
    assert summary["membership_count"] > 0
    assert summary["rule_count"] >= 4
    assert (tmp_path / "data" / "observations.csv").exists()
    assert (tmp_path / "data" / "candidate_patterns.csv").exists()
    assert (tmp_path / "data" / "relation_edges.csv").exists()
    assert (tmp_path / "data" / "mamdani_results.csv").exists()
    assert (tmp_path / "data" / "evaluation_results.csv").exists()
    assert (tmp_path / "data" / "evaluation_gold_templates.csv").exists()
    assert (tmp_path / "data" / "validation_report.csv").exists()
    assert (tmp_path / "data" / "implementation_audit.csv").exists()
    assert (tmp_path / "data" / "completion_assessment.csv").exists()
    assert (tmp_path / "data" / "run_manifest.json").exists()
    assert (tmp_path / "wiki" / "index.md").exists()
    assert any((tmp_path / "wiki" / "patterns").glob("PATTERN_*.md"))
    assert any((tmp_path / "wiki" / "entities").glob("*.md"))
    assert any((tmp_path / "wiki" / "syndromes").glob("*.md"))
    assert (tmp_path / "wiki" / "synthesis" / "global_syndrome_spectrum.md").exists()
    assert (tmp_path / "wiki" / "audit" / "validation_report.md").exists()
    assert (tmp_path / "wiki" / "audit" / "implementation_audit.md").exists()
    assert (tmp_path / "wiki" / "audit" / "completion_assessment.md").exists()
    assert (tmp_path / "wiki" / "audit" / "run_manifest.md").exists()
    assert (tmp_path / "wiki" / "audit" / "mamdani_sensitivity.md").exists()
    assert (tmp_path / "wiki" / "audit" / "evaluation_metrics.md").exists()
    source_page = (tmp_path / "wiki" / "sources" / "SRC_001.md").read_text(encoding="utf-8")
    assert "Observation" in source_page
    assert "形式化说明" in source_page
