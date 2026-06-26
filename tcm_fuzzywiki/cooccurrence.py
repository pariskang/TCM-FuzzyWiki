"""Observation co-occurrence mining and candidate pattern discovery.

The miner uses an Apriori-style vertical tidset implementation instead of
brute-force per-source combinations. Candidate itemsets are generated only by
joining previously support-frequent itemsets, then counted through compact
integer source bitsets. This keeps the search bounded by Apriori support pruning
rather than by materializing the full ``C(vocabulary, k)`` space.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

from .models import CandidatePattern, Observation, SourceUnit

try:  # Optional dependency: co-occurrence mining must still run without scipy.
    from scipy.stats import fisher_exact as _scipy_fisher_exact
except ImportError:  # pragma: no cover - depends on optional environment package.
    _scipy_fisher_exact = None

SMALL_SAMPLE_SOURCE_THRESHOLD = 30
DEFAULT_TOP_K_PATTERNS = 1000
DEFAULT_MAX_SIZE = 3
# Safety backstop for ``max_size: 0`` (auto/unlimited). Apriori already stops at
# the deepest support-frequent level (``if not current: break``) and each level is
# bounded by max_pair_candidates / max_join_candidates, so this ceiling only guards
# against pathological inputs (e.g. a single source listing dozens of identical
# items at very low min_support). 12-item co-occurrence patterns are already extreme
# for TCM corpora, so the natural support break almost always fires far sooner.
ABSOLUTE_MAX_SIZE = 12
DEFAULT_MAX_SOURCE_IDS_PER_PATTERN = 100
_EPSILON = 1e-12
Tidset = int


def itemsets_by_source(observations: list[Observation]) -> dict[str, set[str]]:
    """Return stable, normalized observation itemsets keyed by source_id.

    ``standard_observation`` is preferred. If it is blank, we fall back to the
    explicit ``feature:feature_value`` observation key so unmapped but meaningful
    observations do not disappear from audit tables. Text is normalized and
    deduplicated per source to avoid vocabulary inflation from duplicate raw
    mentions or spacing variants.
    """

    by_source: dict[str, set[str]] = defaultdict(set)
    for obs in observations:
        item = _observation_item(obs)
        if item:
            by_source[obs.source_id].add(item)
    return dict(by_source)


def mine_candidate_patterns(
    observations: list[Observation],
    sources: list[SourceUnit],
    config: dict[str, Any],
    max_size: int = DEFAULT_MAX_SIZE,
) -> list[CandidatePattern]:
    """Mine statistically filtered candidate observation patterns.

    The public signature is intentionally kept compatible with the downstream
    pipeline. Configuration lives under ``candidate_pattern_filter``. For normal
    samples, effective minimum source count is adaptive:
    ``max(config_min_source_count, ceil(N * min_support))``. For ``N < 30``, the
    miner emits explicitly exploratory rows with a bounded ``top_k_patterns`` so
    demo/smoke builds remain auditable without pretending lift/PMI are reliable.
    """

    filters = config.get("candidate_pattern_filter", {})
    # max_size <= 0 means "auto/unlimited": mine itemsets up to the deepest level
    # that still clears min_support, capped by ABSOLUTE_MAX_SIZE as a safety
    # backstop. Per-level explosion is bounded by max_pair/join_candidates below.
    configured_max_size = int(filters.get("max_size", max_size or DEFAULT_MAX_SIZE))
    max_size = ABSOLUTE_MAX_SIZE if configured_max_size <= 0 else max(2, min(configured_max_size, ABSOLUTE_MAX_SIZE))
    min_support = float(filters.get("min_support", 0.03))
    min_lift = float(filters.get("min_lift", 1.5))
    min_pmi = float(filters.get("min_pmi", 0.5))
    configured_min_source_count = int(filters.get("min_source_count", 3))
    min_tradition_count = int(filters.get("min_tradition_count", 1))
    fisher_p_max = filters.get("fisher_p_max")
    fisher_p_threshold = float(fisher_p_max) if fisher_p_max is not None else None
    max_source_ids_per_pattern = int(filters.get("max_source_ids_per_pattern", DEFAULT_MAX_SOURCE_IDS_PER_PATTERN))
    max_pair_candidates = int(filters.get("max_pair_candidates", 200_000))
    max_join_candidates = int(filters.get("max_join_candidates", 100_000))

    source_items = itemsets_by_source(observations)
    all_source_ids = sorted({source.source_id for source in sources} or set(source_items))
    total_sources = len(all_source_ids)
    if total_sources == 0 or not source_items:
        return []

    small_sample = total_sources < SMALL_SAMPLE_SOURCE_THRESHOLD
    effective_min_tradition_count = 1 if small_sample else min_tradition_count
    top_k_patterns = int(filters.get("top_k_patterns", DEFAULT_TOP_K_PATTERNS if small_sample else 0) or 0)
    adaptive_support_count = max(1, math.ceil(total_sources * min_support))
    production_min_source_count = max(configured_min_source_count, adaptive_support_count)
    effective_min_source_count = adaptive_support_count if small_sample else production_min_source_count

    source_id_by_idx = all_source_ids
    source_idx_by_id = {source_id: idx for idx, source_id in enumerate(source_id_by_idx)}
    source_meta = {s.source_id: s for s in sources}

    evidence_by_source_item: dict[tuple[int, str], list[str]] = defaultdict(list)
    mapping_status_by_item: dict[str, Counter[str]] = defaultdict(Counter)
    for obs in observations:
        item = _observation_item(obs)
        source_idx = source_idx_by_id.get(obs.source_id)
        if not item or source_idx is None:
            continue
        mapping_status_by_item[item][obs.mapping_status or "unknown"] += 1
        if obs.evidence_text:
            evidence_by_source_item[(source_idx, item)].append(obs.evidence_text)

    single_tidsets = _build_single_tidsets(source_items, source_idx_by_id)
    single_counts: Counter[str] = Counter({item: _tidset_count(tidset) for item, tidset in single_tidsets.items()})
    frequent_tidsets_by_size = _mine_support_frequent_tidsets(
        single_tidsets, max_size, effective_min_source_count, max_pair_candidates, max_join_candidates
    )

    raw_candidates: list[tuple[tuple[str, ...], Tidset, dict[str, Any]]] = []
    for size in range(2, max_size + 1):
        for combo, tidset in frequent_tidsets_by_size.get(size, {}).items():
            metrics = _metrics_for_itemset(combo, tidset, single_counts, total_sources)
            metrics["source_count"] = _tidset_count(tidset)
            fisher_p = _fisher_p_value(combo, tidset, single_counts, total_sources)
            metrics["fisher_p"] = fisher_p
            if not _passes_statistical_filters(metrics, min_lift, min_pmi, fisher_p_threshold):
                continue
            traditions = _traditions_for_tidset(tidset, source_id_by_idx, source_meta)
            if len(traditions) < effective_min_tradition_count:
                continue
            raw_candidates.append((combo, tidset, metrics))

    raw_candidates.sort(key=lambda row: _candidate_sort_key(row[2]), reverse=True)
    if top_k_patterns > 0:
        raw_candidates = raw_candidates[:top_k_patterns]

    candidates: list[CandidatePattern] = []
    for out_idx, (combo, tidset, metrics) in enumerate(raw_candidates, start=1):
        source_count = _tidset_count(tidset)
        all_source_ids_iter = (source_id_by_idx[idx] for idx in _iter_tidset_indexes(tidset))
        all_source_ids = list(all_source_ids_iter)
        output_source_ids = all_source_ids[: max(0, max_source_ids_per_pattern)]
        books = {source_meta[sid].book_name for sid in all_source_ids if sid in source_meta}
        traditions = {source_meta[sid].tradition_id for sid in all_source_ids if sid in source_meta}
        tradition_entropy = _normalized_entropy([source_meta[sid].tradition_id for sid in all_source_ids if sid in source_meta])
        source_diversity = len(books) / max(1, source_count)
        representative_evidence = _representative_evidence(combo, tidset, source_id_by_idx, evidence_by_source_item)
        review_status = "exploratory_only" if small_sample else "pending"
        summary = _source_count_summary(
            source_count=source_count,
            total_sources=total_sources,
            small_sample=small_sample,
            scipy_available=_scipy_fisher_exact is not None,
            production_min_source_count=production_min_source_count,
            effective_min_source_count=effective_min_source_count,
            source_ids_truncated=source_count > len(output_source_ids),
            max_source_ids=len(output_source_ids),
        )
        candidates.append(
            CandidatePattern(
                pattern_id=f"PATTERN_{out_idx:06d}",
                observations=combo,
                support=round(metrics["support"], 6),
                confidence=round(metrics["confidence"], 6),
                lift=round(metrics["lift"], 6),
                pmi=round(metrics["pmi"], 6),
                jaccard=round(metrics["jaccard"], 6),
                source_count=source_count,
                book_count=len(books),
                tradition_count=len(traditions),
                source_diversity=round(source_diversity, 6),
                tradition_entropy=round(tradition_entropy, 6),
                source_ids=output_source_ids,
                book_names=sorted(books),
                tradition_ids=sorted(traditions),
                representative_evidence=representative_evidence,
                possible_interpretation=_suggest_interpretations(combo),
                status="candidate_pattern",
                itemset=list(combo),
                size=len(combo),
                fisher_p=None if metrics["fisher_p"] is None else round(metrics["fisher_p"], 12),
                source_count_summary=summary,
                mapping_status_summary=_mapping_status_summary(combo, mapping_status_by_item),
                review_status=review_status,
            )
        )
    return candidates


def _observation_item(obs: Observation) -> str:
    raw = obs.standard_observation or f"{obs.feature}:{obs.feature_value}"
    return _normalize_item_text(raw)


def _normalize_item_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if ":" in text:
        feature, feature_value = text.split(":", 1)
        return f"{feature.strip().lower()}:{feature_value.strip().lower()}"
    return text.lower()


def _build_single_tidsets(source_items: dict[str, set[str]], source_idx_by_id: dict[str, int]) -> dict[str, Tidset]:
    tidsets: dict[str, Tidset] = defaultdict(int)
    for source_id, items in source_items.items():
        source_bit = 1 << source_idx_by_id[source_id]
        for item in items:
            tidsets[item] |= source_bit
    return dict(tidsets)


def _mine_support_frequent_tidsets(
    single_tidsets: dict[str, Tidset],
    max_size: int,
    min_count: int,
    max_pair_candidates: int = 200_000,
    max_join_candidates: int = 100_000,
) -> dict[int, dict[tuple[str, ...], Tidset]]:
    """Mine frequent itemsets using only anti-monotonic support pruning.

    Lift, PMI and Fisher p-value are intentionally *not* used here because they
    are not Apriori anti-monotonic and would incorrectly suppress larger
    itemsets whose smaller subsets fail those later statistical filters.
    """

    frequent_by_size: dict[int, dict[tuple[str, ...], Tidset]] = {
        1: {(item,): tidset for item, tidset in single_tidsets.items() if _tidset_count(tidset) >= min_count}
    }
    previous = frequent_by_size[1]
    for size in range(2, max_size + 1):
        candidates = _frequent_pair_candidates(previous, min_count, max_pair_candidates) if size == 2 else _apriori_join(previous, size, max_join_candidates)
        current: dict[tuple[str, ...], Tidset] = {}
        for combo, parent_a, parent_b in candidates:
            tidset = previous[parent_a] & previous[parent_b]
            if _tidset_count(tidset) >= min_count:
                current[combo] = tidset
        if not current:
            break
        frequent_by_size[size] = current
        previous = current
    return frequent_by_size


def _frequent_pair_candidates(
    previous: dict[tuple[str, ...], Tidset], min_count: int, max_pair_candidates: int
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    singletons = sorted(previous)
    source_count_by_item = {item: _tidset_count(previous[item]) for item in singletons}
    pair_upper_bounds = _pair_upper_bounds(singletons, previous, source_count_by_item)
    candidates: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple[str, str]] = set()
    for item, partner_counts in pair_upper_bounds.items():
        for partner, upper_bound in partner_counts.items():
            combo = tuple(sorted((item, partner)))
            if combo in seen or upper_bound < min_count:
                continue
            seen.add(combo)
            candidates.append((combo, (combo[0],), (combo[1],)))
    if max_pair_candidates > 0 and len(candidates) > max_pair_candidates:
        candidates.sort(key=lambda row: min(source_count_by_item[row[1]], source_count_by_item[row[2]]), reverse=True)
        candidates = candidates[:max_pair_candidates]
    return candidates


def _pair_upper_bounds(
    singletons: list[tuple[str, ...]], previous: dict[tuple[str, ...], Tidset], source_count_by_item: dict[tuple[str, ...], int]
) -> dict[str, Counter[str]]:
    by_source: dict[int, list[str]] = defaultdict(list)
    for singleton in singletons:
        item = singleton[0]
        for source_idx in _iter_tidset_indexes(previous[singleton]):
            by_source[source_idx].append(item)

    upper_bounds: dict[str, Counter[str]] = defaultdict(Counter)
    for items in by_source.values():
        sorted_items = sorted(items, key=lambda value: source_count_by_item[(value,)])
        for pos, item in enumerate(sorted_items):
            upper_bounds[item].update(sorted_items[pos + 1 :])
    return upper_bounds


def _apriori_join(
    previous: dict[tuple[str, ...], Tidset],
    size: int,
    max_join_candidates: int = 100_000,
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    previous_keys = sorted(previous)
    previous_key_set = set(previous_keys)
    candidates: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for i, left in enumerate(previous_keys):
        for right in previous_keys[i + 1 :]:
            if left[:-1] != right[:-1]:
                if right[:-1] > left[:-1]:
                    break
                continue
            combo = tuple(sorted((*left, right[-1])))
            if len(combo) != size:
                continue
            if all(tuple(part) in previous_key_set for part in _subsets_of_size(combo, size - 1)):
                candidates.append((combo, left, right))
                if max_join_candidates > 0 and len(candidates) >= max_join_candidates:
                    return candidates
    return candidates


def _subsets_of_size(combo: tuple[str, ...], subset_size: int) -> Iterable[tuple[str, ...]]:
    for idx in range(len(combo)):
        subset = combo[:idx] + combo[idx + 1 :]
        if len(subset) == subset_size:
            yield subset


def _metrics_for_itemset(combo: tuple[str, ...], tidset: Tidset, single_counts: Counter[str], total_sources: int) -> dict[str, float]:
    count = _tidset_count(tidset)
    support = count / total_sources
    individual_probs = [single_counts[item] / total_sources for item in combo]
    expected = math.prod(max(prob, _EPSILON) for prob in individual_probs)
    lift = support / expected if expected > 0 else 0.0
    pmi = math.log2(lift) if lift > 0 else float("-inf")
    min_single = min(single_counts[item] for item in combo)
    jaccard = count / max(1, sum(single_counts[item] for item in combo) - count)
    confidence = count / max(1, min_single)
    return {"support": support, "lift": lift, "pmi": pmi, "jaccard": jaccard, "confidence": confidence}


def _fisher_p_value(combo: tuple[str, ...], tidset: Tidset, single_counts: Counter[str], total_sources: int) -> float | None:
    if len(combo) != 2 or _scipy_fisher_exact is None:
        return None
    a = _tidset_count(tidset)
    left_count = single_counts[combo[0]]
    right_count = single_counts[combo[1]]
    b = max(0, left_count - a)
    c = max(0, right_count - a)
    d = max(0, total_sources - a - b - c)
    try:
        _odds_ratio, p_value = _scipy_fisher_exact([[a, b], [c, d]], alternative="greater")
    except Exception:  # pragma: no cover - defensive only; mining should not crash.
        return None
    return float(p_value)


def _passes_statistical_filters(
    metrics: dict[str, float | None], min_lift: float, min_pmi: float, fisher_p_threshold: float | None
) -> bool:
    fisher_p = metrics.get("fisher_p")
    fisher_ok = fisher_p_threshold is None or fisher_p is None or fisher_p <= fisher_p_threshold
    return bool(metrics["lift"] >= min_lift and metrics["pmi"] >= min_pmi and fisher_ok)


def _candidate_sort_key(metrics: dict[str, float | None]) -> tuple[float, float, float, float]:
    fisher_p = metrics.get("fisher_p")
    fisher_score = 0.0 if fisher_p is None else -float(fisher_p)
    return (float(metrics.get("source_count") or 0), float(metrics["lift"] or 0), float(metrics["pmi"] or 0), fisher_score)


def _representative_evidence(
    combo: tuple[str, ...],
    source_tidset: Tidset,
    source_id_by_idx: list[str],
    evidence_by_source_item: dict[tuple[int, str], list[str]],
    limit: int = 5,
) -> list[str]:
    snippets: list[str] = []
    for source_idx in _iter_tidset_indexes(source_tidset):
        parts: list[str] = []
        for item in combo:
            parts.extend(evidence_by_source_item.get((source_idx, item), []))
        if parts:
            snippets.append(f"{source_id_by_idx[source_idx]}: {'；'.join(dict.fromkeys(parts))}")
        if len(snippets) >= limit:
            break
    return snippets


def _traditions_for_tidset(source_tidset: Tidset, source_id_by_idx: list[str], source_meta: dict[str, SourceUnit]) -> set[str]:
    traditions: set[str] = set()
    for source_idx in _iter_tidset_indexes(source_tidset):
        source_id = source_id_by_idx[source_idx]
        if source_id in source_meta:
            traditions.add(source_meta[source_id].tradition_id)
    return traditions


def _source_count_summary(
    *,
    source_count: int,
    total_sources: int,
    small_sample: bool,
    scipy_available: bool,
    production_min_source_count: int,
    effective_min_source_count: int,
    source_ids_truncated: bool,
    max_source_ids: int,
) -> str:
    sample_note = "exploratory_only; lift/PMI unreliable for N<30" if small_sample else "confirmatory_filtering"
    fisher_note = "fisher_available" if scipy_available else "fisher_unavailable_optional_dependency"
    threshold_note = f"effective_min_source_count={effective_min_source_count}; production_min_source_count={production_min_source_count}"
    source_id_note = f"source_ids_truncated_to={max_source_ids}" if source_ids_truncated else "source_ids_complete"
    return f"{source_count}/{total_sources} sources; {sample_note}; {threshold_note}; {fisher_note}; {source_id_note}"


def _mapping_status_summary(combo: tuple[str, ...], mapping_status_by_item: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {item: dict(mapping_status_by_item.get(item, Counter())) for item in combo}


def _tidset_count(tidset: Tidset) -> int:
    return tidset.bit_count()


def _iter_tidset_indexes(tidset: Tidset) -> Iterable[int]:
    while tidset:
        lowest_bit = tidset & -tidset
        yield lowest_bit.bit_length() - 1
        tidset ^= lowest_bit


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
