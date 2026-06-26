"""Resumable chunk-level LLM observation extraction with crash-safe checkpoints.

Design goals (fixes over the original Colab adapter):

1. **Chunk-level resume** instead of source-level: a ``partial_success`` source no
   longer freezes its failed chunks forever — on resume only missing/failed
   chunks are re-extracted.
2. **Append-only JSONL checkpoint**: each completed chunk is one appended line,
   so a crash can at worst lose the line being written (torn lines are skipped
   on reload).  No O(N^2) read-rewrite of a growing CSV on every source.
3. **Resume manifest with hash verification**: resuming against a different
   input file or different chunking parameters is rejected instead of silently
   mixing incompatible checkpoints.  Each chunk record also stores the chunk
   text SHA256 and is only reused when it still matches.
4. **Deterministic observation IDs**: IDs are assigned at assembly time in
   (input source order, chunk index, row order), so the same checkpoint always
   produces the same ``observations.csv`` regardless of thread completion order.
5. **Single source of truth downstream**: the assembled observations are fed to
   the standard :func:`tcm_fuzzywiki.pipeline.run_pipeline`, so every audit /
   validation / wiki artifact stays in sync with the repository pipeline.
"""

from __future__ import annotations

import concurrent.futures as futures
import datetime as _dt
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .extraction import FEATURE_HINTS, SYSTEM_PROMPT, ObservationNormalizer
from .io import write_csv, write_text
from .llmlite import ChatModel
from .models import Observation, SourceUnit, clamp01

EXTRACTION_DIRNAME = "extraction"
CHUNKS_FILENAME = "extraction_chunks.jsonl"
MANIFEST_FILENAME = "extraction_manifest.json"

ALLOWED_FEATURES = {
    "symptom",
    "sign",
    "tongue",
    "pulse",
    "body_part",
    "pathogen",
    "trigger",
    "relieving_factor",
    "aggravating_factor",
    "duration",
    "severity",
    "pain_location",
    "pain_quality",
    "disease_course",
    "formula",
    "herb",
    "acupoint",
    "method",
    "mechanism",
    "other",
}


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def split_text(text: str, chunk_chars: int = 1800, overlap: int = 80) -> list[str]:
    """Split text into chunks at newline/punctuation boundaries with overlap.

    Forward progress is guaranteed (``start`` strictly increases) even for
    pathological overlap/boundary combinations, so the splitter cannot loop.
    """

    text = (text or "").strip()
    if not text:
        return []
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]

    min_advance = max(200, chunk_chars // 3)
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        hard_end = min(start + chunk_chars, n)
        end = hard_end
        if hard_end < n:
            window_start = start + min(min_advance, max(1, hard_end - start - 1))
            newline = text.rfind("\n", window_start, hard_end)
            if newline > start:
                end = newline
            else:
                punct = max(text.rfind(p, window_start, hard_end) for p in ("。", "；", ";", ".", "\n"))
                if punct > start:
                    end = punct + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - max(0, overlap), start + 1)
    return chunks


@dataclass(slots=True)
class ChunkTask:
    source_index: int
    source_id: str
    chunk_index: int
    chunk_total: int
    text: str
    text_sha256: str


def plan_chunks(sources: list[SourceUnit], chunk_chars: int, chunk_overlap: int) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    for source_index, unit in enumerate(sources):
        text = unit.original_text or unit.text_punctuated or unit.text_modern or ""
        chunks = split_text(text, chunk_chars=chunk_chars, overlap=chunk_overlap)
        for chunk_index, chunk in enumerate(chunks, start=1):
            tasks.append(
                ChunkTask(
                    source_index=source_index,
                    source_id=unit.source_id,
                    chunk_index=chunk_index,
                    chunk_total=len(chunks),
                    text=chunk,
                    text_sha256=sha256_text(chunk),
                )
            )
    return tasks


def build_chunk_prompt(
    unit: SourceUnit,
    chunk_text: str,
    chunk_index: int,
    chunk_total: int,
    max_observations_per_chunk: int,
) -> str:
    if max_observations_per_chunk and max_observations_per_chunk > 0:
        budget_line = (
            f"本块最多输出 {max_observations_per_chunk} 条，优先选择医学信息密度最高、可映射性最强的条目。\n"
        )
    else:
        budget_line = (
            "不限制输出条数：穷尽抽取原文片段中所有能直接找到证据的 observation，"
            "不要为了控制数量而遗漏任何实体或关系。\n"
        )
    return (
        f"{FEATURE_HINTS}\n"
        "你正在为 TCM-FuzzyWiki 构建 observation-first 数据。\n"
        "只抽取原文片段中能直接找到证据的 observation，不要抽象推理，不要输出证候诊断结论。\n"
        "每条 observation 必须满足：feature_value 是原文中出现或紧贴原文的短语；evidence_text 是原文证据短句。\n"
        f"{budget_line}"
        "严格只返回 JSON 对象，不要 Markdown，不要解释，不要 <think>。\n"
        'JSON schema: {"observations":[{"feature":"...","feature_value":"...","evidence_text":"...","extraction_confidence":0.0}]}\n\n'
        f"Source ID: {unit.source_id}\n"
        f"书名: {unit.book_name}\n"
        f"卷名: {unit.volume_name}\n"
        f"章节: {unit.chapter_title}\n"
        f"朝代: {unit.dynasty}\n"
        f"作者: {unit.author}\n"
        f"当前分块: {chunk_index}/{chunk_total}\n"
        f"原文片段:\n{chunk_text}\n"
    )


def coerce_observation_rows(payload: Any, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = payload.get("observations", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "symptom")).strip() or "symptom"
        if feature not in ALLOWED_FEATURES:
            feature = "other"
        feature_value = str(row.get("feature_value", "")).strip()
        if not feature_value:
            continue
        out.append(
            {
                "feature": feature,
                "feature_value": feature_value,
                "evidence_text": str(row.get("evidence_text", "")).strip(),
                "extraction_confidence": clamp01(row.get("extraction_confidence", 0.5), 0.5),
            }
        )
        if max_rows > 0 and len(out) >= max_rows:
            break
    return out


class _CheckpointWriter:
    """Thread-safe append-only JSONL writer (one fsync-free append per chunk)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def load_chunk_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load the latest record per (source_id, chunk_index); torn lines are skipped."""

    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from a crash mid-write
            source_id = str(record.get("source_id", ""))
            try:
                chunk_index = int(record.get("chunk_index", 0))
            except (TypeError, ValueError):
                continue
            if not source_id or chunk_index <= 0:
                continue
            key = (source_id, chunk_index)
            # Later lines win, but never let an error overwrite an earlier success:
            # a successful chunk needs no re-extraction even if a later retry failed.
            if records.get(key, {}).get("status") == "success" and record.get("status") != "success":
                continue
            records[key] = record
    return records


def _verify_or_write_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
    resume: bool,
) -> None:
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"checkpoint": existing.get(key), "current": value}
            for key, value in expected.items()
            if str(existing.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(
                "Extraction checkpoint manifest mismatch — refusing to resume against different "
                f"input/parameters: {json.dumps(mismatches, ensure_ascii=False)}. "
                "Use a new --output directory or pass --no-resume to start fresh."
            )
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(expected)
    payload["created_at"] = _now_iso()
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _ProgressReporter:
    """Live progress bar for chunk extraction.

    Uses ``tqdm`` when available (renders nicely in both terminals and Colab);
    otherwise falls back to a throttled single-line ``\\r`` counter on stderr so
    the core package keeps no hard dependency.  Either way it only *displays*
    progress — durability comes from the append-only JSONL checkpoint.
    """

    def __init__(self, total: int, *, skipped: int = 0, enabled: bool = True):
        self.total = total
        self.enabled = enabled and total > 0
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.rows = 0
        self._bar = None
        self._last_text = 0.0
        if not self.enabled:
            return
        try:  # pragma: no cover - exercised only when tqdm is installed.
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=total,
                desc="extract",
                unit="chunk",
                dynamic_ncols=True,
                initial=0,
            )
            if skipped:
                self._bar.set_postfix_str(f"resumed={skipped}", refresh=False)
        except Exception:
            self._bar = None

    def update(self, record: dict[str, Any]) -> None:
        self.done += 1
        if record.get("status") == "success":
            self.ok += 1
            self.rows += int(record.get("n_observations", 0) or 0)
        else:
            self.failed += 1
        if not self.enabled:
            return
        if self._bar is not None:  # pragma: no cover - requires tqdm.
            self._bar.set_postfix(ok=self.ok, fail=self.failed, rows=self.rows, refresh=False)
            self._bar.update(1)
            return
        now = time.monotonic()
        if now - self._last_text >= 0.5 or self.done >= self.total:
            self._last_text = now
            pct = (self.done / self.total * 100.0) if self.total else 100.0
            end = "\n" if self.done >= self.total else ""
            print(
                f"\r[extract] {self.done}/{self.total} ({pct:5.1f}%) ok={self.ok} fail={self.failed} rows={self.rows}",
                end=end,
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:  # pragma: no cover - requires tqdm.
            self._bar.close()


def _write_progress_json(
    ext_dir: Path,
    *,
    total_chunks: int,
    skipped: int,
    completed: int,
    pending: int,
    failed: int,
    observation_count: int,
    phase: str,
) -> None:
    processed = skipped + completed
    pct = round(processed / total_chunks * 100.0, 2) if total_chunks else 100.0
    write_text(
        ext_dir / "progress.json",
        json.dumps(
            {
                "timestamp": _now_iso(),
                "phase": phase,
                "total_chunks": total_chunks,
                "skipped_resumed": skipped,
                "completed_this_run": completed,
                "pending_this_run": pending,
                "failed_this_run": failed,
                "processed_chunks": processed,
                "percent_complete": pct,
                "observation_count": observation_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _write_partial_outputs(
    ext_dir: Path,
    sources: list[SourceUnit],
    records: dict[tuple[str, int], dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """Assemble current records and persist partial observations/progress CSVs.

    Returns the assembled observation count so callers can surface it live.
    """

    observations, progress_rows = _assemble_observations(sources, records, config)
    write_csv(ext_dir / "observations_checkpoint.csv", observations)
    write_csv(ext_dir / "source_progress.csv", progress_rows)
    return len(observations)


def extract_resumable(
    sources: list[SourceUnit],
    config: dict[str, Any],
    output_dir: str | Path,
    llm: ChatModel,
    *,
    chunk_chars: int = 1800,
    chunk_overlap: int = 80,
    max_observations_per_chunk: int = 0,
    workers: int = 3,
    resume: bool = True,
    input_sha256: str = "",
    model_label: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = False,
    save_interval_sec: float = 20.0,
) -> tuple[list[Observation], dict[str, Any]]:
    """Extract observations with chunk-level checkpoints; safe to interrupt and rerun.

    Returns ``(observations, report)``.  Observations are normalized and carry
    deterministic IDs derived from input order, independent of thread timing.
    """

    out = Path(output_dir)
    ext_dir = out / EXTRACTION_DIRNAME
    chunks_path = ext_dir / CHUNKS_FILENAME
    manifest_path = ext_dir / MANIFEST_FILENAME

    if not resume:
        chunks_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    _verify_or_write_manifest(
        manifest_path,
        {
            "input_sha256": input_sha256,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "max_observations_per_chunk": max_observations_per_chunk,
            "model_label": model_label,
        },
        resume=resume,
    )

    tasks = plan_chunks(sources, chunk_chars, chunk_overlap)
    records = load_chunk_records(chunks_path)
    source_by_id = {unit.source_id: unit for unit in sources}

    def _is_done(task: ChunkTask) -> bool:
        record = records.get((task.source_id, task.chunk_index))
        return bool(record and record.get("status") == "success" and record.get("chunk_sha256") == task.text_sha256)

    pending = [task for task in tasks if not _is_done(task)]
    skipped = len(tasks) - len(pending)
    writer = _CheckpointWriter(chunks_path)
    completed = 0
    failed_now = 0
    reporter = _ProgressReporter(len(pending), skipped=skipped, enabled=show_progress)
    running_observations = sum(int(r.get("n_observations", 0) or 0) for r in records.values() if r.get("status") == "success")

    def _emit_status(extra: str = "", phase: str = "extract") -> None:
        write_text(
            ext_dir / "live_status.txt",
            f"timestamp={_now_iso()}\ntotal_chunks={len(tasks)}\nskipped_resumed={skipped}\n"
            f"completed_this_run={completed}/{len(pending)}\nfailed_this_run={failed_now}\n{extra}",
        )
        _write_progress_json(
            ext_dir,
            total_chunks=len(tasks),
            skipped=skipped,
            completed=completed,
            pending=len(pending),
            failed=failed_now,
            observation_count=running_observations,
            phase=phase,
        )

    def _process(task: ChunkTask) -> dict[str, Any]:
        unit = source_by_id[task.source_id]
        prompt = build_chunk_prompt(unit, task.text, task.chunk_index, task.chunk_total, max_observations_per_chunk)
        started = time.time()
        record: dict[str, Any] = {
            "timestamp": _now_iso(),
            "source_id": task.source_id,
            "chunk_index": task.chunk_index,
            "chunk_total": task.chunk_total,
            "chunk_sha256": task.text_sha256,
        }
        try:
            if hasattr(llm, "complete_json_with_meta"):
                payload, meta = llm.complete_json_with_meta(SYSTEM_PROMPT, prompt)
            else:
                payload, meta = llm.complete_json(SYSTEM_PROMPT, prompt), {}
            rows = coerce_observation_rows(payload, max_observations_per_chunk)
            record.update(
                status="success",
                observations=rows,
                n_observations=len(rows),
                elapsed_sec=round(time.time() - started, 3),
                usage=meta.get("usage", {}),
                finish_reason=meta.get("finish_reason", ""),
                attempt=meta.get("attempt", ""),
            )
        except Exception as exc:
            record.update(
                status="error",
                observations=[],
                n_observations=0,
                elapsed_sec=round(time.time() - started, 3),
                error=repr(exc)[:2000],
            )
        return record

    _emit_status("phase=start")
    if pending:
        last_save = time.monotonic()
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {executor.submit(_process, task): task for task in pending}
            for future in futures.as_completed(future_map):
                record = future.result()
                key = (str(record["source_id"]), int(record["chunk_index"]))
                if not (records.get(key, {}).get("status") == "success" and record["status"] != "success"):
                    records[key] = record
                writer.append(record)
                completed += 1
                if record["status"] == "success":
                    running_observations += int(record.get("n_observations", 0) or 0)
                else:
                    failed_now += 1
                reporter.update(record)
                _emit_status(f"last_chunk={key[0]}#{key[1]} last_status={record['status']}")
                # Real-time save: periodically flush assembled observations so
                # partial results are durable and inspectable mid-run (throttled
                # by time to keep assembly cost bounded on very large corpora).
                now = time.monotonic()
                if save_interval_sec > 0 and now - last_save >= save_interval_sec:
                    last_save = now
                    running_observations = _write_partial_outputs(ext_dir, sources, records, config)
                if progress_callback is not None:
                    progress_callback(record)
    reporter.close()
    _emit_status("phase=assembly", phase="assembly")

    observations, progress_rows = _assemble_observations(sources, records, config)
    write_csv(ext_dir / "observations_checkpoint.csv", observations)
    write_csv(ext_dir / "source_progress.csv", progress_rows)
    running_observations = len(observations)
    _emit_status("phase=done", phase="done")
    error_rows = [
        {k: v for k, v in record.items() if k != "observations"}
        for record in records.values()
        if record.get("status") != "success"
    ]
    write_csv(ext_dir / "llm_errors.csv", error_rows)
    usage_rows = [
        {
            "source_id": record["source_id"],
            "chunk_index": record["chunk_index"],
            "chunk_total": record.get("chunk_total", ""),
            "n_observations": record.get("n_observations", 0),
            "elapsed_sec": record.get("elapsed_sec", ""),
            "finish_reason": record.get("finish_reason", ""),
            "attempt": record.get("attempt", ""),
            **{f"usage_{k}": v for k, v in (record.get("usage") or {}).items() if not isinstance(v, (dict, list))},
        }
        for record in records.values()
        if record.get("status") == "success"
    ]
    write_csv(ext_dir / "llm_usage.csv", usage_rows)

    status_counts: dict[str, int] = {}
    for row in progress_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    report = {
        "total_sources": len(sources),
        "total_chunks": len(tasks),
        "chunks_resumed_from_checkpoint": skipped,
        "chunks_executed_this_run": len(pending),
        "chunks_failed": sum(1 for task in tasks if not _is_done(task)),
        "observation_count": len(observations),
        "source_status_counts": status_counts,
        "checkpoint_dir": str(ext_dir),
    }
    return observations, report


def _assemble_observations(
    sources: list[SourceUnit],
    records: dict[tuple[str, int], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[Observation], list[dict[str, Any]]]:
    """Assemble deduplicated observations with deterministic IDs and per-source status."""

    observations: list[Observation] = []
    progress_rows: list[dict[str, Any]] = []
    next_id = 1
    for unit in sources:
        chunk_indexes = sorted(index for source_id, index in records if source_id == unit.source_id)
        seen: set[tuple[str, str, str]] = set()
        n_success = n_failed = 0
        errors: list[str] = []
        for chunk_index in chunk_indexes:
            record = records[(unit.source_id, chunk_index)]
            if record.get("status") != "success":
                n_failed += 1
                if record.get("error"):
                    errors.append(str(record["error"])[:300])
                continue
            n_success += 1
            for row in record.get("observations", []):
                key = (row["feature"], row["feature_value"], row.get("evidence_text", ""))
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    Observation(
                        observation_id=f"OBS_{next_id:06d}",
                        source_id=unit.source_id,
                        feature=row["feature"],
                        feature_value=row["feature_value"],
                        evidence_text=row.get("evidence_text", ""),
                        extraction_confidence=clamp01(row.get("extraction_confidence", 0.5), 0.5),
                    )
                )
                next_id += 1
        if n_failed and n_success:
            status = "partial_success"
        elif n_failed:
            status = "error"
        elif n_success:
            status = "success"
        else:
            status = "no_chunks"
        progress_rows.append(
            {
                "source_id": unit.source_id,
                "status": status,
                "chunks_success": n_success,
                "chunks_failed": n_failed,
                "n_observations": len(seen),
                "error": " | ".join(errors)[:1000],
            }
        )
    normalized = ObservationNormalizer(config.get("observation_mapping", {})).normalize(observations)
    return normalized, progress_rows
