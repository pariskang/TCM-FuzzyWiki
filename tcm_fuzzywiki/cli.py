"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from .assessment import assess_output
from .calibration import calibrate_config_from_experts, write_calibrated_config
from .config import load_yaml
from .io import read_chapters
from .pipeline import run_pipeline
from .llmlite import (
    AnthropicCompatibleConfig,
    AnthropicCompatibleLLM,
    AzureChatGPTConfig,
    AzureChatGPTLLM,
    OpenAICompatibleConfig,
    OpenAICompatibleLLM,
)
from .provenance import file_sha256
from .resume import extract_resumable
from .roleplay import DeterministicRoleplayScorer, LLMRoleplayExpertScorer, calibrate_from_roleplay_scores
from .validation import readiness_markdown, validate_config, validate_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="TCM-FuzzyWiki V5.0 builder")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build data tables and Markdown wiki from XLSX/CSV chapters")
    build.add_argument("--input", required=True, help="Input .xlsx or .csv chapter table")
    build.add_argument("--config", default="configs/tcm_fuzzywiki.yaml", help="YAML config path")
    build.add_argument("--output", default="build", help="Output directory")
    build.add_argument("--azure-llm", action="store_true", help="Use Azure ChatGPT via llmlite adapter instead of deterministic extractor")
    build.add_argument("--rules-csv", help="Optional expert-reviewed rules.csv")
    build.add_argument("--gold-dir", help="Optional directory with expert gold-standard CSV files for evaluation")



    build_llm = sub.add_parser(
        "build-llm",
        help="Build via an OpenAI-compatible LLM (e.g. MiniMax-M3) with chunk-level resumable checkpoints",
    )
    build_llm.add_argument("--input", required=True, help="Input .xlsx or .csv chapter table")
    build_llm.add_argument("--config", default="configs/tcm_fuzzywiki.yaml", help="YAML config path")
    build_llm.add_argument("--output", required=True, help="Output directory (also holds extraction/ checkpoints)")
    build_llm.add_argument("--provider", choices=["openai", "anthropic", "azure"], default="openai", help="LLM wire protocol; openai/anthropic target MiniMax-M3 by default, azure targets an Azure OpenAI-compatible deployment (e.g. Kimi-K2.5)")
    build_llm.add_argument("--model", default=None, help="Model / Azure deployment name (default env OPENAI_MODEL/ANTHROPIC_MODEL/AZURE_OPENAI_DEPLOYMENT or MiniMax-M3)")
    build_llm.add_argument("--base-url", default=None, help="Base URL / Azure endpoint (default MiniMax: /v1 for openai, /anthropic for anthropic; for azure use https://<resource>.openai.azure.com or set AZURE_OPENAI_ENDPOINT)")
    build_llm.add_argument("--temperature", type=float, default=0.0)
    build_llm.add_argument("--max-tokens", type=int, default=3000, help="max_tokens per chunk completion")
    build_llm.add_argument("--thinking", choices=["adaptive", "disabled", "none"], default="disabled", help="MiniMax thinking mode; 'none' omits the field for non-MiniMax servers")
    build_llm.add_argument("--use-response-format", action="store_true", help="Send response_format json_object (auto-fallback if rejected)")
    build_llm.add_argument("--chunk-chars", type=int, default=1800)
    build_llm.add_argument("--chunk-overlap", type=int, default=80)
    build_llm.add_argument("--max-observations-per-chunk", type=int, default=12)
    build_llm.add_argument("--workers", type=int, default=3)
    build_llm.add_argument("--request-timeout", type=float, default=180.0)
    build_llm.add_argument("--max-retries", type=int, default=4)
    build_llm.add_argument("--retry-sleep", type=float, default=4.0)
    build_llm.add_argument("--limit", type=int, default=0, help="Process only the first N sources (smoke test)")
    build_llm.add_argument("--no-resume", action="store_true", help="Discard extraction checkpoints and start fresh")
    build_llm.add_argument("--strict", action="store_true", help="Exit non-zero if any chunk still failed after retries, before downstream build")
    build_llm.add_argument("--rules-csv", help="Optional expert-reviewed rules.csv")
    build_llm.add_argument("--gold-dir", help="Optional directory with expert gold-standard CSV files")

    normalize = sub.add_parser("normalize-input", help="Normalize an arbitrary chapter XLSX/CSV into the recommended input schema")
    normalize.add_argument("--input", required=True, help="Raw .xlsx or .csv chapter table")
    normalize.add_argument("--output", required=True, help="Normalized .xlsx output path (a sibling .csv is also written)")
    normalize.add_argument("--sheet", default="0", help="Sheet name or index for .xlsx inputs (default 0)")
    normalize.add_argument("--report", help="Optional column-mapping JSON report path")

    assess = sub.add_parser("assess", help="Assess whether an existing build output is formal-ready or still has caveats")
    assess.add_argument("--output", default="build/demo", help="Build output directory containing data/*.csv audit files")

    roleplay = sub.add_parser("roleplay-score", help="Use LLM role-play experts to score linguistic mappings for calibration")
    roleplay.add_argument("--config", default="configs/tcm_fuzzywiki.yaml", help="Input YAML config path")
    roleplay.add_argument("--output-scores", required=True, help="CSV path for generated role-play expert scores")
    roleplay.add_argument("--output-config", help="Optional calibrated YAML config path")
    roleplay.add_argument("--report", help="Optional calibration report CSV path")
    roleplay.add_argument("--azure-llm", action="store_true", help="Use Azure ChatGPT via environment variables")
    roleplay.add_argument("--offline-demo", action="store_true", help="Use deterministic offline demo scores instead of an LLM")
    roleplay.add_argument("--batch-size", type=int, default=20, help="Number of mappings per LLM scoring request")

    calibrate = sub.add_parser("calibrate", help="Calibrate bootstrap linguistic values from expert membership scores")
    calibrate.add_argument("--config", default="configs/tcm_fuzzywiki.yaml", help="Input YAML config path")
    calibrate.add_argument("--expert-scores", required=True, help="CSV with term,variable,fuzzy_set,expert_id,score columns")
    calibrate.add_argument("--output-config", required=True, help="Path for calibrated YAML config")
    calibrate.add_argument("--report", help="Optional calibration report CSV path")

    doctor = sub.add_parser("doctor", help="Validate config and optional input metadata without running the full pipeline")
    doctor.add_argument("--config", default="configs/tcm_fuzzywiki.yaml", help="YAML config path")
    doctor.add_argument("--input", help="Optional .xlsx or .csv chapter table to validate")

    demo = sub.add_parser("run-demo", help="Run the bundled bootstrap demo")
    demo.add_argument("--output", default="build/demo", help="Output directory")

    args = parser.parse_args()
    if args.command == "build-llm":
        summary = _run_build_llm(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "normalize-input":
        from .normalize import normalize_chapter_table

        sheet: int | str = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        _, report = normalize_chapter_table(args.input, args.output, args.report, sheet_name=sheet)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "assess":
        _, markdown = assess_output(args.output)
        print(markdown)
        return
    if args.command == "roleplay-score":
        config = load_yaml(args.config)
        if args.offline_demo:
            scorer = DeterministicRoleplayScorer()
        elif args.azure_llm:
            scorer = LLMRoleplayExpertScorer(AzureChatGPTLLM(AzureChatGPTConfig.from_env()))
        else:
            raise SystemExit("roleplay-score requires --azure-llm for LLM scoring or --offline-demo for deterministic local scoring")
        rows = scorer.score_config(config, args.batch_size)
        _, report = calibrate_from_roleplay_scores(config, rows, args.output_scores, args.output_config, args.report)
        print(json.dumps({"roleplay_scores": len(rows), "output_scores": args.output_scores, "calibrated_mappings": len(report), "output_config": args.output_config}, ensure_ascii=False, indent=2))
        return
    if args.command == "calibrate":
        config = load_yaml(args.config)
        calibrated, report = calibrate_config_from_experts(config, args.expert_scores)
        write_calibrated_config(calibrated, report, args.output_config, args.report)
        print(json.dumps({"calibrated_mappings": len(report), "output_config": args.output_config, "report": args.report}, ensure_ascii=False, indent=2))
        return
    if args.command == "doctor":
        config = load_yaml(args.config)
        findings = validate_config(config)
        if args.input:
            findings.extend(validate_sources(read_chapters(args.input)))
        print(readiness_markdown(findings))
        return
    if args.command == "run-demo":
        input_path = Path("examples/bootstrap_chapters.csv")
        summary = run_pipeline(input_path, "configs/tcm_fuzzywiki.yaml", args.output)
    else:
        summary = run_pipeline(args.input, args.config, args.output, args.azure_llm, args.rules_csv, args.gold_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_build_llm(args: argparse.Namespace) -> dict:
    """Resumable LLM extraction followed by the full standard pipeline."""

    config = load_yaml(args.config)
    sources = read_chapters(args.input)
    input_for_pipeline = Path(args.input)
    if args.limit and args.limit > 0:
        sources = sources[: args.limit]

    # Sanity check: refuse to silently process tens of thousands of empty-text
    # sources when the input column names didn't map to text_original/原文/etc.
    # This is the most common Colab failure mode for unfamiliar XLSX exports.
    n_total = len(sources)
    n_with_text = sum(1 for source in sources if (source.original_text or source.text_punctuated or source.text_modern).strip())
    if n_total > 0 and n_with_text == 0:
        raise SystemExit(
            f"All {n_total} sources read from {args.input!s} have empty text. "
            "The chapter-body column was not recognised. Run "
            "`tcm-fuzzywiki normalize-input --input <raw> --output <normalized.xlsx>` first "
            "and pass the generated .csv (or .xlsx) as --input to build-llm. "
            "Alternatively rename your body column to one of: "
            "text_original / original_text / 原文 / 正文 / 内容 / chapter_text / text / 古籍原文 / 分章内容 / 段落."
        )
    if n_total > 0 and n_with_text < n_total * 0.5:
        print(
            f"warning: only {n_with_text}/{n_total} sources have usable text; "
            "consider running `tcm-fuzzywiki normalize-input` and re-running with the normalized output.",
            flush=True,
        )

    if args.provider == "anthropic":
        thinking = {"type": "enabled", "budget_tokens": 1024} if args.thinking == "adaptive" else None
        llm = AnthropicCompatibleLLM(
            AnthropicCompatibleConfig.from_env(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                thinking=thinking,
            )
        )
        provider_label, model_label, base_url = "anthropic", llm.config.model, llm.config.base_url
    elif args.provider == "azure":
        # Azure's OpenAI-compatible /openai/v1 API; the MiniMax-only `thinking`
        # field is never sent (Azure deployments such as Kimi-K2.5 reject it).
        llm = OpenAICompatibleLLM(
            OpenAICompatibleConfig.from_azure(
                model=args.model,
                endpoint=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                use_response_format=args.use_response_format,
            )
        )
        provider_label, model_label, base_url = "azure_openai_compatible", llm.config.model, llm.config.base_url
    else:
        extra_body = {} if args.thinking == "none" else {"thinking": {"type": args.thinking}}
        llm = OpenAICompatibleLLM(
            OpenAICompatibleConfig.from_env(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                use_response_format=args.use_response_format,
                extra_body=extra_body,
            )
        )
        provider_label, model_label, base_url = "openai_compatible", llm.config.model, llm.config.base_url

    observations, report = extract_resumable(
        sources,
        config,
        args.output,
        llm,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        max_observations_per_chunk=args.max_observations_per_chunk,
        workers=args.workers,
        resume=not args.no_resume,
        input_sha256=file_sha256(args.input),
        model_label=llm.config.model,
    )
    print(json.dumps({"extraction_report": report}, ensure_ascii=False, indent=2))
    if args.strict and report["chunks_failed"] > 0:
        raise SystemExit(
            f"--strict: {report['chunks_failed']} chunk(s) still failed; rerun the same command to resume, "
            f"see {report['checkpoint_dir']}/llm_errors.csv"
        )

    if args.limit and args.limit > 0:
        import pandas as pd

        subset_path = Path(args.output) / "extraction" / "input_subset.csv"
        frame = pd.read_excel(args.input) if input_for_pipeline.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.input)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        frame.head(args.limit).to_csv(subset_path, index=False, encoding="utf-8-sig")
        input_for_pipeline = subset_path

    summary = run_pipeline(
        input_for_pipeline,
        args.config,
        args.output,
        rules_csv=args.rules_csv,
        gold_dir=args.gold_dir,
        observations=observations,
        manifest_extra={
            "execution": {"extractor": f"{provider_label}_llm_resumable"},
            "llm_provider": provider_label,
            "llm_model": model_label,
            "llm_base_url": base_url,
            "extraction_report": report,
        },
    )
    return summary


if __name__ == "__main__":
    main()
