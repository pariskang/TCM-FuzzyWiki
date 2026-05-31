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


def read_chapters(path: str | Path) -> list[SourceUnit]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported chapter input format: {path.suffix}")
    units: list[SourceUnit] = []
    for idx, row in frame.iterrows():
        source_id = _text(row, "source_id", default=f"SRC_{idx + 1:04d}")
        quality = EvidenceQuality(
            source_authority=clamp01(row.get("source_authority", 0.5), 0.5),
            text_integrity=clamp01(row.get("text_integrity", 0.5), 0.5),
            semantic_clarity=clamp01(row.get("semantic_clarity", 0.5), 0.5),
        )
        order_value = row.get("chapter_order")
        chapter_order = None if pd.isna(order_value) else int(order_value)
        units.append(
            SourceUnit(
                source_id=source_id,
                book_name=_text(row, "book_name", default="uncertain"),
                volume_name=_text(row, "volume_name", default="uncertain"),
                chapter_title=_text(row, "chapter_title", default="uncertain"),
                chapter_order=chapter_order,
                dynasty=_text(row, "dynasty", default="uncertain"),
                author=_text(row, "author", default="uncertain"),
                text_type=_text(row, "text_type", default="uncertain"),
                topic_hint=_text(row, "topic_hint", default="uncertain"),
                notes=_text(row, "notes", default=""),
                school_tag=_split_tags(row.get("school_tag", "uncertain")),
                region_tag=_split_tags(row.get("region_tag", "uncertain")),
                tradition_id=_text(row, "tradition_id", default="uncertain"),
                text_family=_text(row, "text_family", default="uncertain"),
                citation_family=_text(row, "citation_family", default="uncertain"),
                original_text=_text(row, "text_original", "original_text", default=""),
                text_punctuated=_text(row, "text_punctuated", default=""),
                text_modern=_text(row, "text_modern", default=""),
                chapter_summary=_text(row, "chapter_summary", default=""),
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
