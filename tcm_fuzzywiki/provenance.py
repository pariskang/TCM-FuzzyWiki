"""Reproducibility manifest generation for TCM-FuzzyWiki builds."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def file_sha256(path: str | Path | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    use_azure_llm: bool = False,
    rules_csv: str | Path | None = None,
    gold_dir: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": "tcm-fuzzywiki",
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
        },
        "config": {
            "path": str(config_path),
            "file_sha256": file_sha256(config_path),
            "loaded_config_sha256": config_sha256(config),
        },
        "optional_inputs": {
            "rules_csv": str(rules_csv) if rules_csv else "",
            "rules_csv_sha256": file_sha256(rules_csv),
            "gold_dir": str(gold_dir) if gold_dir else "",
        },
        "execution": {
            "extractor": "azure_chatgpt_llm" if use_azure_llm else "deterministic_rule_based",
            "output_dir": str(output_dir),
        },
        "summary": summary,
    }


def manifest_markdown(manifest: dict[str, Any]) -> str:
    summary_rows = "\n".join(f"| {key} | {value} |" for key, value in manifest.get("summary", {}).items())
    return f"""# 运行复现 Manifest

## 构建信息

| 字段 | 值 |
|---|---|
| Generated at UTC | {manifest['generated_at_utc']} |
| Package | {manifest['package']} {manifest['package_version']} |
| Python | {manifest['python_version']} |
| Platform | {manifest['platform']} |
| Extractor | {manifest['execution']['extractor']} |
| Output dir | {manifest['execution']['output_dir']} |

## 输入哈希

| 输入 | 路径 | SHA256 |
|---|---|---|
| Chapter input | {manifest['input']['path']} | {manifest['input']['sha256']} |
| Config file | {manifest['config']['path']} | {manifest['config']['file_sha256']} |
| Loaded config | - | {manifest['config']['loaded_config_sha256']} |
| Rules CSV | {manifest['optional_inputs']['rules_csv']} | {manifest['optional_inputs']['rules_csv_sha256']} |

## Summary

| 指标 | 值 |
|---|---:|
{summary_rows}

## 说明

该 manifest 用于复现实验和审计：输入文件、配置文件、加载后的配置内容、执行模式和输出 summary 均被记录。若任一 SHA256 改变，则该次构建不应被视为同一次可复现实验。
"""
