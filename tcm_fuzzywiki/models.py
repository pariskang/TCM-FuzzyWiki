"""Typed data models for the TCM-FuzzyWiki V5.0 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceQuality:
    source_authority: float = 0.5
    text_integrity: float = 0.5
    semantic_clarity: float = 0.5

    @property
    def score(self) -> float:
        return max(0.0, min(1.0, (self.source_authority + self.text_integrity + self.semantic_clarity) / 3.0))


@dataclass(slots=True)
class SourceUnit:
    source_id: str
    book_name: str
    volume_name: str = "uncertain"
    chapter_title: str = "uncertain"
    chapter_order: int | None = None
    dynasty: str = "uncertain"
    author: str = "uncertain"
    text_type: str = "uncertain"
    topic_hint: str = "uncertain"
    notes: str = ""
    school_tag: list[str] = field(default_factory=lambda: ["uncertain"])
    region_tag: list[str] = field(default_factory=lambda: ["uncertain"])
    tradition_id: str = "uncertain"
    text_family: str = "uncertain"
    citation_family: str = "uncertain"
    original_text: str = ""
    text_punctuated: str = ""
    text_modern: str = ""
    chapter_summary: str = ""
    evidence_quality: EvidenceQuality = field(default_factory=EvidenceQuality)


@dataclass(slots=True)
class Observation:
    observation_id: str
    source_id: str
    feature: str
    feature_value: str
    evidence_text: str
    extraction_confidence: float
    standard_observation: str = ""
    mapping_status: str = "unmapped"
    review_status: str = "pending"

    @property
    def observation_key(self) -> str:
        return self.standard_observation or f"{self.feature}:{self.feature_value}"


@dataclass(slots=True)
class Membership:
    membership_id: str
    observation_id: str
    source_id: str
    standard_observation: str
    variable: str
    fuzzy_set: str
    membership: float
    calculation_mode: str
    status: str = "bootstrap_prior"
    icc: float | None = None
    p5: float | None = None
    p95: float | None = None
    uncertainty_width: float | None = None

    @property
    def variable_key(self) -> str:
        return f"{self.variable}.{self.fuzzy_set}"


@dataclass(slots=True)
class CandidatePattern:
    pattern_id: str
    observations: tuple[str, ...]
    support: float
    confidence: float
    lift: float
    pmi: float
    jaccard: float
    source_count: int
    book_count: int
    tradition_count: int
    source_diversity: float
    tradition_entropy: float
    source_ids: list[str] = field(default_factory=list)
    book_names: list[str] = field(default_factory=list)
    tradition_ids: list[str] = field(default_factory=list)
    representative_evidence: list[str] = field(default_factory=list)
    possible_interpretation: list[str] = field(default_factory=list)
    status: str = "candidate_pattern"


@dataclass(slots=True)
class RuleAntecedent:
    variable: str
    fuzzy_set: str
    threshold: float = 0.0
    weight: float = 1.0

    @property
    def variable_key(self) -> str:
        return f"{self.variable}.{self.fuzzy_set}"


@dataclass(slots=True)
class FuzzyRule:
    rule_id: str
    rule_name: str
    rule_origin: str
    antecedents: list[RuleAntecedent]
    consequent_entity: str
    consequent_type: str = "syndrome"
    rule_weight: float = 1.0
    pattern_id: str = ""
    support: float = 0.0
    confidence: float = 0.0
    lift: float = 0.0
    pmi: float = 0.0
    source_count: int = 0
    source_ids: list[str] = field(default_factory=list)
    book_count: int = 0
    tradition_count: int = 0
    tradition_ids: list[str] = field(default_factory=list)
    source_diversity: float = 0.0
    tradition_entropy: float = 0.0
    applicable_context: str = ""
    conflict_with: str = ""
    expert_acceptance_rate: float = 0.0
    expert_disagreement_note: str = ""
    review_status: str = "active_rule"
    created_from: str = ""
    last_updated: str = ""


@dataclass(slots=True)
class InferenceResult:
    source_id: str
    rule_id: str
    consequent_entity: str
    consequent_type: str
    activation: float
    supporting_variables: list[str]
    missing_variables: list[str]
    status: str
    p5: float | None = None
    p95: float | None = None


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
