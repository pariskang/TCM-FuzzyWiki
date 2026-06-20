"""Input/output utilities for source evidence units and pipeline tables."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .models import EvidenceQuality, SourceUnit, clamp01


def _split_tags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ["uncertain"]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()] or ["uncertain"]
    text = str(value).strip()
    if not text:
        return ["uncertain"]
    for sep in [";", "；", ",", "，", "|"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()] or ["uncertain"]
    return [text]


def _text(row: pd.Series, *names: str, default: str = "uncertain") -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return default


# Chinese/English column aliases accepted directly by read_chapters, so users do
# not have to pre-normalize when their classical-text export uses common Chinese
# headings.  Kept in sync with tcm_fuzzywiki.normalize.COLUMN_ALIASES.
_ALIASES: dict[str, tuple[str, ...]] = {
    "source_id": ("source_id", "id", "编号", "章节id", "条目id"),
    "book_name": ("book_name", "书名", "古籍名称", "书籍", "book", "title"),
    "volume_name": ("volume_name", "卷名", "卷", "volume", "册"),
    "chapter_title": ("chapter_title", "章节标题", "章节", "篇名", "章名", "chapter", "section_title", "标题"),
    "chapter_order": ("chapter_order", "章节序号", "序号", "order", "chapter_index", "index"),
    "dynasty": ("dynasty", "朝代", "年代"),
    "author": ("author", "作者"),
    "text_original": ("text_original", "original_text", "原文", "正文", "内容", "chapter_text", "text", "古籍原文", "分章内容", "段落"),
    "text_punctuated": ("text_punctuated", "标点原文", "点校文本"),
    "text_modern": ("text_modern", "现代文", "白话文", "译文"),
    "chapter_summary": ("chapter_summary", "摘要", "小结", "summary"),
    "text_type": ("text_type", "文本类型", "类型"),
    "topic_hint": ("topic_hint", "主题", "主题提示", "topic"),
    "school_tag": ("school_tag", "流派", "学派"),
    "region_tag": ("region_tag", "地域", "地区"),
    "tradition_id": ("tradition_id", "学术传统", "传统"),
    "text_family": ("text_family", "文献类型家族", "文献类型"),
    "citation_family": ("citation_family", "引用谱系", "版本谱系"),
    "notes": ("notes", "备注", "说明"),
}


def _alias_text(row: pd.Series, target: str, default: str = "uncertain") -> str:
    return _text(row, *_ALIASES.get(target, (target,)), default=default)


def _guess_longest_text(row: pd.Series, exclude: set[str]) -> str:
    best = ""
    for column in row.index:
        if column in exclude:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if len(text) > len(best):
            best = text
    return best


def read_chapters(path: str | Path) -> list[SourceUnit]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported chapter input format: {path.suffix}")
    # Aliases let users pass classical-text tables with common Chinese column
    # names directly to read_chapters / build / build-llm.  When even the
    # broad alias set misses the body column (the common case for unusual
    # exports), fall back to the longest-text column of the row so we never
    # silently emit thousands of empty SourceUnits.
    text_alias_columns = {column for alias in _ALIASES["text_original"] if alias in frame.columns for column in [alias]}
    metadata_alias_columns = {
        alias for target, aliases in _ALIASES.items() if target != "text_original" for alias in aliases if alias in frame.columns
    }
    units: list[SourceUnit] = []
    for idx, row in frame.iterrows():
        source_id = _alias_text(row, "source_id", default=f"SRC_{idx + 1:04d}")
        quality = EvidenceQuality(
            source_authority=clamp01(row.get("source_authority", 0.5), 0.5),
            text_integrity=clamp01(row.get("text_integrity", 0.5), 0.5),
            semantic_clarity=clamp01(row.get("semantic_clarity", 0.5), 0.5),
        )
        order_value = None
        for name in _ALIASES["chapter_order"]:
            if name in row and pd.notna(row[name]):
                try:
                    order_value = int(float(row[name]))
                    break
                except (TypeError, ValueError):
                    continue
        original_text = _alias_text(row, "text_original", default="")
        if not original_text and not text_alias_columns:
            original_text = _guess_longest_text(row, exclude=metadata_alias_columns | {"source_authority", "text_integrity", "semantic_clarity"})
        units.append(
            SourceUnit(
                source_id=source_id,
                book_name=_alias_text(row, "book_name", default="uncertain"),
                volume_name=_alias_text(row, "volume_name", default="uncertain"),
                chapter_title=_alias_text(row, "chapter_title", default="uncertain"),
                chapter_order=order_value,
                dynasty=_alias_text(row, "dynasty", default="uncertain"),
                author=_alias_text(row, "author", default="uncertain"),
                text_type=_alias_text(row, "text_type", default="uncertain"),
                topic_hint=_alias_text(row, "topic_hint", default="uncertain"),
                notes=_alias_text(row, "notes", default=""),
                school_tag=_split_tags(next((row[alias] for alias in _ALIASES["school_tag"] if alias in row and pd.notna(row[alias])), "uncertain")),
                region_tag=_split_tags(next((row[alias] for alias in _ALIASES["region_tag"] if alias in row and pd.notna(row[alias])), "uncertain")),
                tradition_id=_alias_text(row, "tradition_id", default="uncertain"),
                text_family=_alias_text(row, "text_family", default="uncertain"),
                citation_family=_alias_text(row, "citation_family", default="uncertain"),
                original_text=original_text,
                text_punctuated=_alias_text(row, "text_punctuated", default=""),
                text_modern=_alias_text(row, "text_modern", default=""),
                chapter_summary=_alias_text(row, "chapter_summary", default=""),
                evidence_quality=quality,
            )
        )
    return units


def dataclass_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        row = asdict(item) if is_dataclass(item) else dict(item)
        for key, value in list(row.items()):
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value, ensure_ascii=False)
        rows.append(row)
    return rows


def write_csv(path: str | Path, items: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dataclass_rows(items)).to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
