"""Build-output completeness assessment for TCM-FuzzyWiki V5.0.

The pipeline can be feature-complete as software while still not be "perfect" for
formal research claims if a run lacks gold standards, has validation warnings, or
uses bootstrap/LLM-roleplay calibration.  This module turns generated audit CSVs
into a concise verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def assess_output(output_dir: str | Path) -> tuple[dict[str, Any], str]:
    out = Path(output_dir)
    data_dir = out / "data"
    validation = _read_csv(data_dir / "validation_report.csv")
    audit = _read_csv(data_dir / "implementation_audit.csv")
    evaluation = _read_csv(data_dir / "evaluation_results.csv")

    validation_errors = _count(validation, "severity", "error")
    validation_warnings = _count(validation, "severity", "warning")
    mvp_capabilities = _count(audit, "status", "implemented_mvp")
    future_work = _count(audit, "status", "future_work")
    missing_gold = _count(evaluation, "status", "needs_gold_standard")

    if validation_errors or future_work:
        verdict = "not_ready"
        message = "存在阻塞错误或未实现能力，不能称为完美实现。"
    elif validation_warnings or missing_gold or mvp_capabilities:
        verdict = "research_ready_with_caveats"
        message = "软件链路可运行，但仍有需要专家/金标准/外部本体补齐的研究边界。"
    else:
        verdict = "formal_ready"
        message = "当前输出未发现验证错误、警告、缺失 gold 或 MVP 边界。"

    row = {
        "verdict": verdict,
        "message": message,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "implemented_mvp_capabilities": mvp_capabilities,
        "future_work_capabilities": future_work,
        "metrics_needing_gold_standard": missing_gold,
        "assessed_output_dir": str(out),
    }
    return row, assessment_markdown(row)


def assessment_markdown(row: dict[str, Any]) -> str:
    return f"""# V5.0 完整性评估

## Verdict

**{row['verdict']}**：{row['message']}

## Counts

| 项目 | 数量 |
|---|---:|
| Validation errors | {row['validation_errors']} |
| Validation warnings | {row['validation_warnings']} |
| Implemented-MVP capabilities | {row['implemented_mvp_capabilities']} |
| Future-work capabilities | {row['future_work_capabilities']} |
| Metrics needing gold standard | {row['metrics_needing_gold_standard']} |

## 解释

- `formal_ready` 才可近似回答“该次构建已达到形式化完备”。
- `research_ready_with_caveats` 表示代码链路可运行，但仍不能把缺失专家金标准、外部本体维护或 MVP 边界说成已经完美解决。
- `not_ready` 表示存在阻塞问题，应先修复后再进入正式分析。
"""


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _count(frame: pd.DataFrame, column: str, value: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((frame[column].astype(str) == value).sum())
