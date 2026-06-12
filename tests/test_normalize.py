import json
from pathlib import Path

import pandas as pd

from tcm_fuzzywiki.io import read_chapters
from tcm_fuzzywiki.normalize import normalize_chapter_table, safe_write_excel


def test_normalize_maps_chinese_columns_and_fills_defaults(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "书名": ["伤寒论", "金匮要略", "丹溪心法"],
            "篇名": ["辨太阳病", "", "腰痛"],
            "原文": ["太阳之为病，脉浮。", None, "腰痛而冷，得温则缓。"],
            "朝代": ["汉", "汉", "元"],
        }
    )
    raw_path = tmp_path / "raw.xlsx"
    safe_write_excel(raw, raw_path)

    out_path = tmp_path / "normalized.xlsx"
    frame, report = normalize_chapter_table(raw_path, out_path)

    assert report["column_mapping"]["book_name"] == "书名"
    assert report["column_mapping"]["text_original"] == "原文"
    assert report["dropped_empty_text_rows"] == 1  # the None 原文 row is removed
    assert report["final_rows"] == 2
    assert (out_path).exists()
    assert out_path.with_suffix(".csv").exists()
    assert Path(report["mapping_report_path"]).exists()
    assert json.loads(Path(report["mapping_report_path"]).read_text(encoding="utf-8"))["final_rows"] == 2

    # Auto-filled requirements of the V5.0 schema
    assert frame["source_id"].str.startswith("SRC_").all()
    assert (frame["tradition_id"] == "uncertain").all()
    # Order reflects original row positions (row 2 was dropped for empty text).
    assert frame["chapter_order"].tolist() == [1, 3]
    assert (frame["source_authority"] == 0.8).all()
    # Blank chapter title replaced rather than left empty
    assert frame["chapter_title"].str.len().gt(0).all()

    # The normalized CSV is directly consumable by the repo reader.
    units = read_chapters(out_path.with_suffix(".csv"))
    assert len(units) == 2
    assert units[0].book_name == "伤寒论"


def test_normalize_guesses_longest_text_column(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "标识": ["a", "b"],
            "无名长文本": ["腰痛而冷，得温则缓。" * 10, "刺痛不移，夜间加重。" * 10],
        }
    )
    raw_path = tmp_path / "raw.xlsx"
    safe_write_excel(raw, raw_path)
    frame, report = normalize_chapter_table(raw_path, tmp_path / "normalized.xlsx")
    assert "guessed_longest_text_column" in report["column_mapping"]["text_original"]
    assert frame["text_original"].str.contains("腰痛而冷").any()
