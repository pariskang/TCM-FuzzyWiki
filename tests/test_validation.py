from tcm_fuzzywiki.config import load_yaml
from tcm_fuzzywiki.io import read_chapters
from tcm_fuzzywiki.validation import readiness_markdown, validate_config, validate_sources


def test_validation_accepts_default_config_and_demo_sources():
    config_findings = validate_config(load_yaml("configs/tcm_fuzzywiki.yaml"))
    assert any(row["code"] == "config_valid" for row in config_findings)
    source_findings = validate_sources(read_chapters("examples/bootstrap_chapters.csv"))
    assert any(row["code"] == "source_metadata_ready" for row in source_findings)
    markdown = readiness_markdown(config_findings + source_findings)
    assert "完备性与质量验证报告" in markdown
