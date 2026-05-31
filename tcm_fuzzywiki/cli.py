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
from .llmlite import AzureChatGPTConfig, AzureChatGPTLLM
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


if __name__ == "__main__":
    main()
