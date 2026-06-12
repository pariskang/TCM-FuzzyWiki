from pathlib import Path

import pytest

from tcm_fuzzywiki.config import load_yaml
from tcm_fuzzywiki.models import SourceUnit
from tcm_fuzzywiki.pipeline import run_pipeline
from tcm_fuzzywiki.resume import extract_resumable, load_chunk_records, split_text


class FlakyLLM:
    """Fails configured (source_id, chunk_index) calls once, then succeeds."""

    def __init__(self, fail_once: set[tuple[str, int]]):
        self.fail_once = set(fail_once)
        self.calls: list[tuple[str, int]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        source_id = next(line.split(": ", 1)[1] for line in user_prompt.splitlines() if line.startswith("Source ID: "))
        chunk_line = next(line.split(": ", 1)[1] for line in user_prompt.splitlines() if line.startswith("当前分块: "))
        chunk_index = int(chunk_line.split("/")[0])
        key = (source_id, chunk_index)
        self.calls.append(key)
        if key in self.fail_once:
            self.fail_once.discard(key)
            raise RuntimeError(f"simulated transient failure for {key}")
        return {
            "observations": [
                {
                    "feature": "pain_quality",
                    "feature_value": "冷痛",
                    "evidence_text": f"{source_id}-chunk{chunk_index}-冷痛",
                    "extraction_confidence": 0.9,
                }
            ]
        }


def _sources() -> list[SourceUnit]:
    text = "腰痛而冷。\n" * 60  # long enough for several chunks at chunk_chars=120
    return [
        SourceUnit(source_id="SRC_A", book_name="书甲", original_text=text, tradition_id="t1"),
        SourceUnit(source_id="SRC_B", book_name="书乙", original_text=text, tradition_id="t2"),
    ]


def test_split_text_makes_forward_progress_on_pathological_overlap():
    text = "一" * 5000
    chunks = split_text(text, chunk_chars=300, overlap=290)
    assert chunks
    assert sum(len(c) for c in chunks) >= len(text)  # overlap duplicates allowed, no loss
    assert split_text("", 100, 10) == []
    assert split_text("short", 100, 10) == ["short"]


def test_chunk_level_resume_retries_only_failed_chunks(tmp_path: Path):
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    sources = _sources()
    llm = FlakyLLM(fail_once={("SRC_A", 2), ("SRC_B", 1)})

    obs1, report1 = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, workers=2, input_sha256="HASH1"
    )
    assert report1["chunks_failed"] == 2
    assert report1["source_status_counts"].get("partial_success", 0) >= 1
    first_run_calls = len(llm.calls)
    assert first_run_calls == report1["total_chunks"]

    # Resume: only the two failed chunks are re-executed.
    obs2, report2 = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, workers=2, input_sha256="HASH1"
    )
    assert len(llm.calls) - first_run_calls == 2
    assert report2["chunks_failed"] == 0
    assert report2["source_status_counts"] == {"success": 2}
    assert len(obs2) >= len(obs1)

    # Third run: nothing pending, deterministic IDs identical to second run.
    obs3, report3 = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, workers=2, input_sha256="HASH1"
    )
    assert len(llm.calls) - first_run_calls == 2  # no new calls
    assert report3["chunks_executed_this_run"] == 0
    assert [(o.observation_id, o.source_id, o.feature_value) for o in obs3] == [
        (o.observation_id, o.source_id, o.feature_value) for o in obs2
    ]
    assert (tmp_path / "extraction" / "observations_checkpoint.csv").exists()
    assert (tmp_path / "extraction" / "source_progress.csv").exists()


def test_resume_rejects_mismatched_input_hash(tmp_path: Path):
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    sources = _sources()
    llm = FlakyLLM(fail_once=set())
    extract_resumable(sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, input_sha256="HASH1")
    with pytest.raises(ValueError, match="manifest mismatch"):
        extract_resumable(sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, input_sha256="HASH2")
    # --no-resume starts fresh and succeeds with the new hash.
    _, report = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=120, chunk_overlap=10, input_sha256="HASH2", resume=False
    )
    assert report["chunks_failed"] == 0


def test_torn_checkpoint_line_is_skipped(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    good = '{"source_id": "S1", "chunk_index": 1, "status": "success", "chunk_sha256": "x", "observations": []}'
    path.write_text(good + "\n" + '{"source_id": "S1", "chunk_index": 2, "status": "succ', encoding="utf-8")
    records = load_chunk_records(path)
    assert list(records) == [("S1", 1)]


def test_injected_observations_run_full_pipeline(tmp_path: Path):
    config = load_yaml("configs/tcm_fuzzywiki.yaml")
    llm = FlakyLLM(fail_once=set())
    sources_csv = Path("examples/bootstrap_chapters.csv")
    from tcm_fuzzywiki.io import read_chapters

    sources = read_chapters(sources_csv)
    observations, report = extract_resumable(
        sources, config, tmp_path, llm, chunk_chars=2000, chunk_overlap=0, input_sha256="DEMO"
    )
    assert report["chunks_failed"] == 0
    summary = run_pipeline(
        sources_csv,
        "configs/tcm_fuzzywiki.yaml",
        tmp_path,
        observations=observations,
        manifest_extra={"llm_provider": "test_fake", "execution": {"extractor": "fake_llm"}},
    )
    assert summary["observation_count"] == len(observations)
    assert summary["membership_count"] > 0
    # Full standard artifact set — proves no drift versus run_pipeline.
    for name in [
        "observations.csv",
        "memberships.csv",
        "methodology_compliance.csv",
        "ontology_lexicon.csv",
        "expert_calibration_template.csv",
        "source_stratification.csv",
        "rule_lifecycle.csv",
        "completion_assessment.csv",
        "run_manifest.json",
    ]:
        assert (tmp_path / "data" / name).exists(), name
    import json

    manifest = json.loads((tmp_path / "data" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_provider"] == "test_fake"
    assert manifest["execution"]["extractor"] == "fake_llm"
