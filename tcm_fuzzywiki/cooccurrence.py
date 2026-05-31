"""Observation co-occurrence mining and candidate pattern discovery."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any

from .models import CandidatePattern, Observation, SourceUnit


def itemsets_by_source(observations: list[Observation]) -> dict[str, set[str]]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for obs in observations:
        if obs.standard_observation:
            by_source[obs.source_id].add(obs.standard_observation)
    return dict(by_source)


def mine_candidate_patterns(
    observations: list[Observation],
    sources: list[SourceUnit],
    config: dict[str, Any],
    max_size: int = 4,
) -> list[CandidatePattern]:
    source_items = itemsets_by_source(observations)
    total_sources = max(1, len(source_items))
    source_meta = {s.source_id: s for s in sources}
    evidence_by_source_item: dict[tuple[str, str], list[str]] = defaultdict(list)
    for obs in observations:
        if obs.standard_observation and obs.evidence_text:
            evidence_by_source_item[(obs.source_id, obs.standard_observation)].append(obs.evidence_text)
    single_counts: Counter[str] = Counter()
    itemset_counts: Counter[tuple[str, ...]] = Counter()
    itemset_sources: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for source_id, items in source_items.items():
        for item in items:
            single_counts[item] += 1
        sorted_items = sorted(items)
        for size in range(2, min(max_size, len(sorted_items)) + 1):
            for combo in itertools.combinations(sorted_items, size):
                itemset_counts[combo] += 1
                itemset_sources[combo].add(source_id)

    filters = config.get("candidate_pattern_filter", {})
    min_support = float(filters.get("min_support", 0.03))
    min_lift = float(filters.get("min_lift", 1.5))
    min_pmi = float(filters.get("min_pmi", 0.5))
    min_source_count = int(filters.get("min_source_count", 3))
    min_tradition_count = int(filters.get("min_tradition_count", 1))

    candidates: list[CandidatePattern] = []
    for idx, (combo, count) in enumerate(itemset_counts.most_common(), start=1):
        support = count / total_sources
        individual_probs = [single_counts[item] / total_sources for item in combo]
        expected = math.prod(max(prob, 1e-12) for prob in individual_probs)
        lift = support / expected if expected > 0 else 0.0
        pmi = math.log2(lift) if lift > 0 else 0.0
        min_single = min(single_counts[item] for item in combo)
        jaccard = count / max(1, sum(single_counts[item] for item in combo) - count)
        confidence = count / max(1, min_single)
        source_ids = itemset_sources[combo]
        books = {source_meta[sid].book_name for sid in source_ids if sid in source_meta}
        traditions = {source_meta[sid].tradition_id for sid in source_ids if sid in source_meta}
        tradition_entropy = _normalized_entropy([source_meta[sid].tradition_id for sid in source_ids if sid in source_meta])
        source_diversity = len(books) / max(1, len(source_ids))
        representative_evidence = _representative_evidence(combo, source_ids, evidence_by_source_item)
        if (
            support >= min_support
            and lift >= min_lift
            and pmi >= min_pmi
            and len(source_ids) >= min_source_count
            and len(traditions) >= min_tradition_count
        ):
            candidates.append(
                CandidatePattern(
                    pattern_id=f"PATTERN_{idx:06d}",
                    observations=combo,
                    support=round(support, 6),
                    confidence=round(confidence, 6),
                    lift=round(lift, 6),
                    pmi=round(pmi, 6),
                    jaccard=round(jaccard, 6),
                    source_count=len(source_ids),
                    book_count=len(books),
                    tradition_count=len(traditions),
                    source_diversity=round(source_diversity, 6),
                    tradition_entropy=round(tradition_entropy, 6),
                    source_ids=sorted(source_ids),
                    book_names=sorted(books),
                    tradition_ids=sorted(traditions),
                    representative_evidence=representative_evidence,
                    possible_interpretation=_suggest_interpretations(combo),
                )
            )
    return candidates


def _representative_evidence(
    combo: tuple[str, ...],
    source_ids: set[str],
    evidence_by_source_item: dict[tuple[str, str], list[str]],
    limit: int = 5,
) -> list[str]:
    snippets: list[str] = []
    for source_id in sorted(source_ids):
        parts: list[str] = []
        for item in combo:
            parts.extend(evidence_by_source_item.get((source_id, item), []))
        if parts:
            snippets.append(f"{source_id}: {'；'.join(dict.fromkeys(parts))}")
        if len(snippets) >= limit:
            break
    return snippets


def _normalized_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy else 0.0


def _suggest_interpretations(combo: tuple[str, ...]) -> list[str]:
    joined = " ".join(combo)
    suggestions: list[str] = []
    if "pain_quality:cold" in joined or "relieving_factor:warmth_relieves" in joined:
        suggestions.extend(["寒湿痹阻", "寒凝经脉", "阳虚寒凝"])
    if "course:chronic" in joined or "symptom:waist_knee_soreness" in joined:
        suggestions.append("肾虚证")
    if "pain_quality:stabbing_fixed" in joined or "aggravating_factor:night_worse" in joined:
        suggestions.append("瘀血阻络")
    if "tongue:yellow_greasy" in joined:
        suggestions.append("湿热下注")
    return list(dict.fromkeys(suggestions))
