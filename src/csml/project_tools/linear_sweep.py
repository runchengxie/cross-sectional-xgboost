from __future__ import annotations

import argparse
import copy
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..config_utils import resolve_pipeline_config


@dataclass(frozen=True)
class SweepJob:
    model: str
    alpha: float
    l1_ratio: float | None
    run_name: str
    config_path: Path


def _resolve_path(path_text: str | Path) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _format_float(value: float) -> str:
    return format(float(value), ".12g")


def _parse_float_grid(values: list[str] | None, default: list[float], field_name: str) -> list[float]:
    if values is None:
        return list(default)
    out: list[float] = []
    for entry in values:
        for part in str(entry).split(","):
            text = part.strip()
            if not text:
                continue
            try:
                out.append(float(text))
            except ValueError as exc:
                raise SystemExit(f"Invalid float in --{field_name}: {text}") from exc
    deduped = list(dict.fromkeys(out))
    if not deduped:
        raise SystemExit(f"--{field_name} produced no valid values.")
    return deduped


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _default_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_ridge_job(
    base_cfg: dict[str, Any],
    *,
    run_name_prefix: str,
    alpha: float,
    config_dir: Path,
    runs_dir_override: str | None,
    sample_weight_mode: str,
    random_state: Any,
) -> SweepJob:
    alpha_text = _format_float(alpha)
    run_name = f"{run_name_prefix}ridge_a{alpha_text}"
    cfg = copy.deepcopy(base_cfg)

    model_cfg = _ensure_mapping(cfg, "model")
    model_cfg["type"] = "ridge"
    model_cfg["params"] = {"alpha": float(alpha)}
    if random_state is not None:
        model_cfg["params"]["random_state"] = random_state
    model_cfg["sample_weight_mode"] = sample_weight_mode

    eval_cfg = _ensure_mapping(cfg, "eval")
    eval_cfg["run_name"] = run_name
    if runs_dir_override is not None:
        eval_cfg["output_dir"] = runs_dir_override

    config_path = config_dir / f"ridge_a{alpha_text}.yml"
    _write_yaml(config_path, cfg)
    return SweepJob(
        model="ridge",
        alpha=float(alpha),
        l1_ratio=None,
        run_name=run_name,
        config_path=config_path,
    )


def _build_elasticnet_job(
    base_cfg: dict[str, Any],
    *,
    run_name_prefix: str,
    alpha: float,
    l1_ratio: float,
    config_dir: Path,
    runs_dir_override: str | None,
    sample_weight_mode: str,
    random_state: Any,
) -> SweepJob:
    alpha_text = _format_float(alpha)
    l1_text = _format_float(l1_ratio)
    run_name = f"{run_name_prefix}en_a{alpha_text}_l{l1_text}"
    cfg = copy.deepcopy(base_cfg)

    model_cfg = _ensure_mapping(cfg, "model")
    model_cfg["type"] = "elasticnet"
    model_cfg["params"] = {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}
    if random_state is not None:
        model_cfg["params"]["random_state"] = random_state
    model_cfg["sample_weight_mode"] = sample_weight_mode

    eval_cfg = _ensure_mapping(cfg, "eval")
    eval_cfg["run_name"] = run_name
    if runs_dir_override is not None:
        eval_cfg["output_dir"] = runs_dir_override

    config_path = config_dir / f"elasticnet_a{alpha_text}_l{l1_text}.yml"
    _write_yaml(config_path, cfg)
    return SweepJob(
        model="elasticnet",
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        run_name=run_name,
        config_path=config_path,
    )


def _write_jobs_csv(path: Path, jobs: list[SweepJob]) -> None:
    fieldnames = ["order", "model", "alpha", "l1_ratio", "run_name", "config_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, job in enumerate(jobs, start=1):
            writer.writerow(
                {
                    "order": idx,
                    "model": job.model,
                    "alpha": job.alpha,
                    "l1_ratio": job.l1_ratio,
                    "run_name": job.run_name,
                    "config_path": str(job.config_path),
                }
            )


def _write_run_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["order", "run_name", "config_path", "status", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_linear_sweep_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="config/hk_selected.yml",
        help="Base config path or built-in name (default: config/hk_selected.yml)",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="hk_sel_",
        help="run_name prefix for all generated runs (default: hk_sel_)",
    )
    parser.add_argument(
        "--sweeps-dir",
        default="out/sweeps",
        help="Root directory for sweep artifacts (default: out/sweeps)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Sweep tag used as subdirectory name (default: current timestamp)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Override eval.output_dir for generated configs (default: keep base config value)",
    )
    parser.add_argument(
        "--ridge-alpha",
        action="append",
        default=None,
        help="Comma-separated ridge alpha values (default: 0.01,0.1,1,10,100)",
    )
    parser.add_argument(
        "--elasticnet-alpha",
        action="append",
        default=None,
        help="Comma-separated elasticnet alpha values (default: 0.01,0.1,1)",
    )
    parser.add_argument(
        "--elasticnet-l1-ratio",
        action="append",
        default=None,
        help="Comma-separated elasticnet l1_ratio values (default: 0.1,0.5,0.9)",
    )
    parser.add_argument(
        "--skip-ridge",
        action="store_true",
        help="Skip ridge jobs",
    )
    parser.add_argument(
        "--skip-elasticnet",
        action="store_true",
        help="Skip elasticnet jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate configs/jobs.csv; do not run pipeline or summarize",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one run fails",
    )
    parser.add_argument(
        "--skip-summarize",
        action="store_true",
        help="Skip automatic summarize step after runs",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary CSV path (default: <sweep-dir>/runs_summary.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    ridge_alphas = _parse_float_grid(args.ridge_alpha, [0.01, 0.1, 1, 10, 100], "ridge-alpha")
    en_alphas = _parse_float_grid(args.elasticnet_alpha, [0.01, 0.1, 1], "elasticnet-alpha")
    en_l1s = _parse_float_grid(
        args.elasticnet_l1_ratio,
        [0.1, 0.5, 0.9],
        "elasticnet-l1-ratio",
    )
    if args.skip_ridge and args.skip_elasticnet:
        raise SystemExit("Both --skip-ridge and --skip-elasticnet are set; no jobs to run.")

    resolved = resolve_pipeline_config(args.config)
    base_cfg = copy.deepcopy(resolved.data)
    base_eval = base_cfg.get("eval", {}) if isinstance(base_cfg.get("eval"), dict) else {}
    base_model = base_cfg.get("model", {}) if isinstance(base_cfg.get("model"), dict) else {}
    base_params = base_model.get("params", {}) if isinstance(base_model.get("params"), dict) else {}

    runs_dir_text = args.runs_dir
    if runs_dir_text is None:
        runs_dir_text = str(base_eval.get("output_dir", "out/runs"))
    runs_dir = _resolve_path(runs_dir_text)

    sweep_tag = (args.tag or _default_tag()).strip()
    if not sweep_tag:
        raise SystemExit("Sweep tag cannot be empty.")
    sweep_root = _resolve_path(args.sweeps_dir)
    sweep_dir = sweep_root / sweep_tag
    config_dir = sweep_dir / "configs"
    jobs_csv_path = sweep_dir / "jobs.csv"
    results_csv_path = sweep_dir / "run_results.csv"
    summary_output = _resolve_path(args.summary_output) if args.summary_output else sweep_dir / "runs_summary.csv"

    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    sample_weight_mode = str(base_model.get("sample_weight_mode", "date_equal"))
    random_state = base_params.get("random_state", 42)

    jobs: list[SweepJob] = []
    if not args.skip_ridge:
        for alpha in ridge_alphas:
            jobs.append(
                _build_ridge_job(
                    base_cfg,
                    run_name_prefix=args.run_name_prefix,
                    alpha=alpha,
                    config_dir=config_dir,
                    runs_dir_override=args.runs_dir,
                    sample_weight_mode=sample_weight_mode,
                    random_state=random_state,
                )
            )
    if not args.skip_elasticnet:
        for alpha in en_alphas:
            for l1_ratio in en_l1s:
                jobs.append(
                    _build_elasticnet_job(
                        base_cfg,
                        run_name_prefix=args.run_name_prefix,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        config_dir=config_dir,
                        runs_dir_override=args.runs_dir,
                        sample_weight_mode=sample_weight_mode,
                        random_state=random_state,
                    )
                )

    if not jobs:
        raise SystemExit("No sweep jobs generated.")
    run_names = [job.run_name for job in jobs]
    if len(run_names) != len(set(run_names)):
        raise SystemExit("Duplicate run_name detected in generated jobs. Adjust prefix/grid values.")

    _write_jobs_csv(jobs_csv_path, jobs)
    logging.info("Generated %d configs under %s", len(jobs), config_dir)
    logging.info("Jobs manifest written to %s", jobs_csv_path)

    run_results: list[dict[str, Any]] = []
    failed_count = 0
    started_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if not args.dry_run:
        from .. import pipeline

        for idx, job in enumerate(jobs, start=1):
            status = "ok"
            error_text = ""
            logging.info("[%d/%d] Running %s", idx, len(jobs), job.run_name)
            try:
                pipeline.run(str(job.config_path))
            except KeyboardInterrupt:
                raise
            except SystemExit as exc:
                status = "failed"
                error_text = str(exc)
                failed_count += 1
                logging.error("Run failed: %s (%s)", job.run_name, error_text or "SystemExit")
                if not args.continue_on_error:
                    run_results.append(
                        {
                            "order": idx,
                            "run_name": job.run_name,
                            "config_path": str(job.config_path),
                            "status": status,
                            "error": error_text,
                        }
                    )
                    break
            except Exception as exc:  # pragma: no cover - defensive
                status = "failed"
                error_text = str(exc)
                failed_count += 1
                logging.error("Run failed: %s (%s)", job.run_name, error_text)
                if not args.continue_on_error:
                    run_results.append(
                        {
                            "order": idx,
                            "run_name": job.run_name,
                            "config_path": str(job.config_path),
                            "status": status,
                            "error": error_text,
                        }
                    )
                    break

            run_results.append(
                {
                    "order": idx,
                    "run_name": job.run_name,
                    "config_path": str(job.config_path),
                    "status": status,
                    "error": error_text,
                }
            )
    else:
        logging.info("Dry-run enabled: skip pipeline.run and summarize.")

    _write_run_results_csv(results_csv_path, run_results)
    logging.info("Run results written to %s", results_csv_path)

    summary_status = "skipped"
    summary_error = ""
    if not args.dry_run and not args.skip_summarize:
        from . import summarize_runs

        summarize_argv = [
            "--runs-dir",
            str(runs_dir),
            "--run-name-prefix",
            args.run_name_prefix,
            "--since",
            started_at,
            "--output",
            str(summary_output),
            "--log-level",
            args.log_level,
        ]
        try:
            summarize_runs.main(summarize_argv)
            summary_status = "ok"
        except SystemExit as exc:
            summary_status = "failed"
            summary_error = str(exc)
            logging.error("Summarize failed: %s", summary_error or "SystemExit")

    result = {
        "sweep_dir": str(sweep_dir),
        "config_dir": str(config_dir),
        "jobs_csv": str(jobs_csv_path),
        "run_results_csv": str(results_csv_path),
        "summary_output": str(summary_output),
        "job_count": len(jobs),
        "failed_count": failed_count,
        "summary_status": summary_status,
        "summary_error": summary_error,
    }

    if args.dry_run:
        return result
    if failed_count > 0 and not args.continue_on_error:
        raise SystemExit("Sweep stopped on first failure. Re-run with --continue-on-error to keep going.")
    if failed_count > 0:
        raise SystemExit(f"Sweep finished with {failed_count} failed runs.")
    if summary_status == "failed":
        raise SystemExit(f"Runs finished, but summarize failed: {summary_error}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate linear model sweep configs (ridge/elasticnet), run them, "
            "then summarize results."
        )
    )
    add_linear_sweep_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
