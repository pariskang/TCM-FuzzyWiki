"""Observation-first extraction and normalization."""

from __future__ import annotations

from collections.abc import Mapping

from .llmlite import ChatModel
from .models import Observation, SourceUnit, clamp01


SYSTEM_PROMPT = """你是 TCM-FuzzyWiki V5.0 的 observation-first 抽取器。只抽取可观察事实，禁止输出证候、病机或诊断结论。返回 JSON 对象：{"observations":[{"feature":"...","feature_value":"...","evidence_text":"...","extraction_confidence":0.0}]}。"""

FEATURE_HINTS = """允许的 feature 示例：pain_location, pain_quality, relieving_factor, aggravating_factor, disease_course, tongue, pulse, symptom, sign, formula, herb, acupoint。"""


class ObservationExtractor:
    def extract(self, units: list[SourceUnit]) -> list[Observation]:
        raise NotImplementedError


class LLMObservationExtractor(ObservationExtractor):
    """LLM-backed extractor; downstream code never trusts it for conclusions."""

    def __init__(self, llm: ChatModel):
        self.llm = llm

    def extract(self, units: list[SourceUnit]) -> list[Observation]:
        observations: list[Observation] = []
        next_id = 1
        for unit in units:
            prompt = (
                f"{FEATURE_HINTS}\n"
                f"Source ID: {unit.source_id}\n"
                f"书名: {unit.book_name}\n章节: {unit.chapter_title}\n"
                f"原文: {unit.original_text or unit.text_punctuated}\n"
            )
            payload = self.llm.complete_json(SYSTEM_PROMPT, prompt)
            rows = payload.get("observations", []) if isinstance(payload, Mapping) else []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                observations.append(
                    Observation(
                        observation_id=f"OBS_{next_id:06d}",
                        source_id=unit.source_id,
                        feature=str(row.get("feature", "symptom")),
                        feature_value=str(row.get("feature_value", "")),
                        evidence_text=str(row.get("evidence_text", "")),
                        extraction_confidence=clamp01(row.get("extraction_confidence", 0.5), 0.5),
                    )
                )
                next_id += 1
        return observations


class RuleBasedObservationExtractor(ObservationExtractor):
    """Deterministic bootstrap extractor for demos, tests, and offline review."""

    PATTERNS: tuple[tuple[str, str, str, str, float], ...] = (
        ("腰", "pain_location", "腰部", "腰", 0.94),
        ("腰痛", "pain_location", "腰部", "腰痛", 0.94),
        ("冷痛", "pain_quality", "冷痛", "冷痛", 0.93),
        ("痛而冷", "pain_quality", "冷痛", "痛而冷", 0.93),
        ("腰痛而冷", "pain_quality", "冷痛", "腰痛而冷", 0.93),
        ("得温则缓", "relieving_factor", "得温则缓", "得温则缓", 0.91),
        ("温之稍安", "relieving_factor", "得温则缓", "温之稍安", 0.90),
        ("遇寒", "aggravating_factor", "遇寒加重", "遇寒", 0.89),
        ("久病", "disease_course", "久病", "久病", 0.88),
        ("不愈", "disease_course", "久病", "不愈", 0.82),
        ("刺痛", "pain_quality", "刺痛不移", "刺痛", 0.90),
        ("不移", "pain_quality", "刺痛不移", "不移", 0.86),
        ("夜间加重", "aggravating_factor", "夜间加重", "夜间加重", 0.88),
        ("苔黄腻", "tongue", "苔黄腻", "苔黄腻", 0.92),
        ("脉弦", "pulse", "脉弦", "脉弦", 0.90),
        ("腰膝酸软", "symptom", "腰膝酸软", "腰膝酸软", 0.93),
    )

    def extract(self, units: list[SourceUnit]) -> list[Observation]:
        observations: list[Observation] = []
        next_id = 1
        for unit in units:
            text = unit.original_text or unit.text_punctuated or unit.text_modern
            seen: set[tuple[str, str]] = set()
            for needle, feature, value, evidence, confidence in self.PATTERNS:
                if needle in text and (feature, value) not in seen:
                    observations.append(
                        Observation(
                            observation_id=f"OBS_{next_id:06d}",
                            source_id=unit.source_id,
                            feature=feature,
                            feature_value=value,
                            evidence_text=evidence,
                            extraction_confidence=confidence,
                        )
                    )
                    seen.add((feature, value))
                    next_id += 1
        return observations


class ObservationNormalizer:
    def __init__(self, mapping: Mapping[str, str]):
        self.mapping = {str(k): str(v) for k, v in mapping.items()}

    def normalize(self, observations: list[Observation]) -> list[Observation]:
        for obs in observations:
            candidates = [obs.feature_value, obs.evidence_text, f"{obs.feature}:{obs.feature_value}"]
            standard = next((self.mapping[item] for item in candidates if item in self.mapping), "")
            if standard:
                obs.standard_observation = standard
                obs.mapping_status = "mapped"
            else:
                obs.standard_observation = f"{obs.feature}:{obs.feature_value}"
                obs.mapping_status = "unmapped"
        return observations
