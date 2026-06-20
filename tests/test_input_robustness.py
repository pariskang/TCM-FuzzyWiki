"""Tests for read_chapters Chinese-column tolerance and build-llm input guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tcm_fuzzywiki.io import read_chapters


def _write_xlsx(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_excel(path, index=False)
    return path


def test_read_chapters_recognises_chinese_column_names(tmp_path: Path):
    """Raw Chinese-header XLSX must yield non-empty original_text without normalize-input.

    This is the exact failure mode that previously produced 88883 no_chunks
    sources: book_chapters.xlsx had Chinese headings, read_chapters fell back
    to "", and build-llm silently emitted thousands of empty-text units.
    """

    frame = pd.DataFrame(
        {
            "书名": ["伤寒论", "金匮要略"],
            "篇名": ["辨太阳病", "腰痛"],
            "朝代": ["汉", "汉"],
            "原文": ["太阳之为病，脉浮，头项强痛而恶寒。", "腰痛而冷，得温则缓，遇寒加重。"],
        }
    )
    xlsx = _write_xlsx(frame, tmp_path / "raw.xlsx")
    sources = read_chapters(xlsx)
    assert len(sources) == 2
    assert sources[0].book_name == "伤寒论"
    assert sources[0].chapter_title == "辨太阳病"
    assert sources[0].dynasty == "汉"
    assert "太阳之为病" in sources[0].original_text
    assert "腰痛而冷" in sources[1].original_text


def test_read_chapters_falls_back_to_longest_text_column(tmp_path: Path):
    """When no alias matches, the longest text column should still be used."""

    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "随便起的列名": ["腰痛而冷，得温则缓。" * 8, "刺痛不移，夜间加重。" * 8],
        }
    )
    xlsx = _write_xlsx(frame, tmp_path / "weird.xlsx")
    sources = read_chapters(xlsx)
    assert len(sources) == 2
    assert "腰痛而冷" in sources[0].original_text
    assert sources[0].source_id == "a"


def test_read_chapters_handles_non_integer_chapter_order(tmp_path: Path):
    """chapter_order may arrive as float/str; should not raise."""

    frame = pd.DataFrame(
        {
            "原文": ["一二三", "四五六"],
            "chapter_order": [1.0, "2"],
        }
    )
    sources = read_chapters(_write_xlsx(frame, tmp_path / "ord.xlsx"))
    assert [s.chapter_order for s in sources] == [1, 2]


def test_build_llm_guard_on_truly_empty_text(tmp_path: Path):
    """When every source has empty text, build-llm must fail-fast with guidance,
    not silently emit thousands of no_chunks entries.

    This is the exact regression that caused the Colab "88883 no_chunks" bug:
    the input columns didn't map to text_original/原文/etc, all sources had
    empty text, and the pipeline produced thousands of empty rows + 88w
    validation warnings.  The guard must fire before any LLM call.
    """

    csv_path = tmp_path / "blank.csv"
    csv_path.write_text("text_original,book_name\n,书1\n,书2\n,书3\n", encoding="utf-8")
    output = tmp_path / "out"
    # Provide a dummy API key so we are testing the guard, not env-var validation.
    import os

    env = {**os.environ, "MINIMAX_API_KEY": "dummy"}
    proc = subprocess.run(
        ["tcm-fuzzywiki", "build-llm", "--input", str(csv_path), "--config", "configs/tcm_fuzzywiki.yaml",
         "--output", str(output), "--workers", "1", "--max-retries", "1"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0, f"expected non-zero exit but got {proc.returncode}; stdout={proc.stdout}"
    combined = (proc.stderr or "") + (proc.stdout or "")
    assert "empty text" in combined, f"missing empty-text hint in: {combined[-500:]}"
    assert "normalize-input" in combined, f"missing normalize-input hint in: {combined[-500:]}"
    # Guard must fire BEFORE any extraction directory is written.
    assert not (output / "extraction").exists(), "guard fired too late; extraction dir already created"
