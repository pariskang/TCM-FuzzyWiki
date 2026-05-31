from pathlib import Path

from tcm_fuzzywiki.provenance import build_manifest, config_sha256, file_sha256, manifest_markdown


def test_manifest_records_hashes_and_summary(tmp_path: Path):
    input_file = tmp_path / "input.csv"
    config_file = tmp_path / "config.yaml"
    input_file.write_text("a,b\n1,2\n", encoding="utf-8")
    config_file.write_text("x: 1\n", encoding="utf-8")
    manifest = build_manifest(
        input_path=input_file,
        config_path=config_file,
        output_dir=tmp_path / "out",
        config={"x": 1},
        summary={"source_count": 1},
    )
    assert manifest["input"]["sha256"] == file_sha256(input_file)
    assert manifest["config"]["loaded_config_sha256"] == config_sha256({"x": 1})
    assert "运行复现 Manifest" in manifest_markdown(manifest)
