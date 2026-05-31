"""LLM role-play expert scoring for calibration priors.

This module lets a language model act as a reproducible panel of three reviewer
roles requested by the V5.0 workflow:
- modern evidence-based medicine expert
- Chinese medicine expert
- classical philology / paleography expert

The output is intentionally the same CSV shape accepted by the expert
calibration module, with extra audit columns preserving role, confidence,
rationale, and score provenance.  These scores can substitute for human expert
scores during bootstrap, while remaining explicitly tagged as LLM role-play.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import calibrate_config_from_experts, write_calibrated_config
from .io import write_csv
from .llmlite import ChatModel
from .models import clamp01


@dataclass(frozen=True, slots=True)
class RoleplayExpert:
    expert_id: str
    role_name: str
    system_prompt: str


ROLEPLAY_EXPERTS: tuple[RoleplayExpert, ...] = (
    RoleplayExpert(
        expert_id="LLM_EBM_EXPERT",
        role_name="现代循证医学专家",
        system_prompt=(
            "你是现代循证医学专家。你需要从现代医学/症状学/可验证证据角度，"
            "评估中医古籍 observation 到 fuzzy variable 的映射强度。"
            "不要给临床诊断建议，只给 0 到 1 的隶属度评分、置信度和简短理由。"
        ),
    ),
    RoleplayExpert(
        expert_id="LLM_TCM_EXPERT",
        role_name="中医专家",
        system_prompt=(
            "你是资深中医证候学与中医诊断学专家。你需要根据中医理论、辨证逻辑、"
            "寒热虚实、病机和传统术语习惯，评估 observation 到 fuzzy variable 的映射强度。"
            "不要直接给唯一诊断，只给 0 到 1 的隶属度评分、置信度和简短理由。"
        ),
    ),
    RoleplayExpert(
        expert_id="LLM_PHILOLOGY_EXPERT",
        role_name="古文字/训诂专家",
        system_prompt=(
            "你是古文字、训诂和中医古籍文献专家。你需要根据古汉语语义、异名、语境、"
            "文献表达习惯，评估 observation 术语与 fuzzy variable 映射的语言可靠性。"
            "不要给临床诊断建议，只给 0 到 1 的隶属度评分、置信度和简短理由。"
        ),
    ),
)


class LLMRoleplayExpertScorer:
    """Generate calibration scores from a ChatModel role-play panel."""

    def __init__(self, llm: ChatModel, roles: tuple[RoleplayExpert, ...] = ROLEPLAY_EXPERTS):
        self.llm = llm
        self.roles = roles

    def score_config(self, config: dict[str, Any], batch_size: int = 20) -> list[dict[str, Any]]:
        tasks = build_scoring_tasks(config)
        rows: list[dict[str, Any]] = []
        for role in self.roles:
            for batch in _chunks(tasks, batch_size):
                rows.extend(self._score_batch(role, batch))
        return rows

    def _score_batch(self, role: RoleplayExpert, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = {
            "role": role.role_name,
            "instructions": (
                "请为每个 task 返回 score 与 confidence，范围均为 0-1。"
                "score 表示 term 对 variable.fuzzy_set 的隶属度支持强度；"
                "confidence 表示你对该评分可靠性的自信程度。"
                "只返回 JSON 对象：{\"scores\":[{\"term\":...,\"variable\":...,\"fuzzy_set\":...,"
                "\"score\":0.0,\"confidence\":0.0,\"rationale\":\"...\"}]}。"
            ),
            "tasks": tasks,
        }
        result = self.llm.complete_json(role.system_prompt, json.dumps(payload, ensure_ascii=False))
        scores = result.get("scores", []) if isinstance(result, dict) else []
        rows: list[dict[str, Any]] = []
        task_keys = {(task["term"], task["variable"], task["fuzzy_set"]) for task in tasks}
        for item in scores:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("term", "")), str(item.get("variable", "")), str(item.get("fuzzy_set", "")))
            if key not in task_keys:
                continue
            rows.append(
                {
                    "term": key[0],
                    "variable": key[1],
                    "fuzzy_set": key[2],
                    "expert_id": role.expert_id,
                    "expert_role": role.role_name,
                    "score": round(clamp01(item.get("score", 0.0)), 6),
                    "confidence": round(clamp01(item.get("confidence", 0.5), 0.5), 6),
                    "rationale": str(item.get("rationale", ""))[:500],
                    "score_source": "llm_roleplay",
                }
            )
        missing = task_keys - {(row["term"], row["variable"], row["fuzzy_set"]) for row in rows}
        for term, variable, fuzzy_set in sorted(missing):
            rows.append(
                {
                    "term": term,
                    "variable": variable,
                    "fuzzy_set": fuzzy_set,
                    "expert_id": role.expert_id,
                    "expert_role": role.role_name,
                    "score": 0.0,
                    "confidence": 0.0,
                    "rationale": "LLM response missing this task; inserted zero-confidence score for audit completeness.",
                    "score_source": "llm_roleplay_missing_response",
                }
            )
        return rows


class DeterministicRoleplayScorer:
    """Offline deterministic scorer for tests and demos.

    It is not a language model; its rows are tagged `deterministic_roleplay_demo`.
    """

    def __init__(self, roles: tuple[RoleplayExpert, ...] = ROLEPLAY_EXPERTS):
        self.roles = roles

    def score_config(self, config: dict[str, Any], batch_size: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task in build_scoring_tasks(config):
            base = clamp01(task.get("prior_membership", 0.5), 0.5)
            for role in self.roles:
                adjustment = {
                    "LLM_EBM_EXPERT": -0.03,
                    "LLM_TCM_EXPERT": 0.03,
                    "LLM_PHILOLOGY_EXPERT": 0.0,
                }[role.expert_id]
                rows.append(
                    {
                        "term": task["term"],
                        "variable": task["variable"],
                        "fuzzy_set": task["fuzzy_set"],
                        "expert_id": role.expert_id,
                        "expert_role": role.role_name,
                        "score": round(clamp01(base + adjustment), 6),
                        "confidence": 0.7,
                        "rationale": "Deterministic offline role-play demo score based on configured prior membership.",
                        "score_source": "deterministic_roleplay_demo",
                    }
                )
        return rows


def build_scoring_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for term, entry in config.get("linguistic_values", {}).items():
        if not isinstance(entry, dict):
            continue
        feature = entry.get("feature", "")
        for variable, mapping in entry.get("maps_to", {}).items():
            if not isinstance(mapping, dict):
                continue
            tasks.append(
                {
                    "term": str(term),
                    "feature": feature,
                    "variable": str(variable),
                    "fuzzy_set": str(mapping.get("fuzzy_set", "high")),
                    "prior_membership": clamp01(mapping.get("calibrated_membership", mapping.get("prior_membership", 0.5)), 0.5),
                    "current_status": mapping.get("status", entry.get("status", "bootstrap_prior")),
                    "review_status": mapping.get("review_status", entry.get("review_status", "pending")),
                }
            )
    return tasks


def write_roleplay_scores(rows: list[dict[str, Any]], output_scores: str | Path) -> None:
    write_csv(output_scores, rows)


def calibrate_from_roleplay_scores(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    output_scores: str | Path,
    output_config: str | Path | None = None,
    report_csv: str | Path | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    write_roleplay_scores(rows, output_scores)
    if output_config is None:
        return None, []
    calibrated, report = calibrate_config_from_experts(config, output_scores)
    # Preserve provenance at mapping level after calibration.
    for term_entry in calibrated.get("linguistic_values", {}).values():
        if not isinstance(term_entry, dict):
            continue
        for mapping in term_entry.get("maps_to", {}).values():
            if isinstance(mapping, dict) and mapping.get("review_status") == "expert_reviewed":
                mapping["review_status"] = "llm_roleplay_reviewed"
                mapping["calibration_source"] = "llm_roleplay_panel"
    write_calibrated_config(calibrated, report, output_config, report_csv)
    return calibrated, report


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]
