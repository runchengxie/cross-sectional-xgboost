from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

RUN_DIR_PATTERN = re.compile(
    r"^(?P<run_name>.+)_(?P<timestamp>\d{8}_\d{6})_(?P<config_hash>[0-9a-fA-F]{8})$"
)

FIELDNAMES = [
    "source_runs_dir",
    "run_dir",
    "run_name",
    "run_timestamp",
    "config_hash",
    "summary_path",
    "config_path",
    "market",
    "data_provider",
    "data_start_date",
    "data_end_date",
    "universe_mode",
    "label_horizon_days",
    "label_shift_days",
    "eval_top_k",
    "backtest_top_k",
    "transaction_cost_bps",
    "eval_rebalance_frequency",
    "backtest_rebalance_frequency",
    "eval_buffer_exit",
    "eval_buffer_entry",
    "backtest_buffer_exit",
    "backtest_buffer_entry",
    "eval_ic_mean",
    "eval_ic_ir",
    "eval_long_short",
    "eval_turnover_mean",
    "backtest_periods",
    "backtest_total_return",
    "backtest_ann_return",
    "backtest_ann_vol",
    "backtest_sharpe",
    "backtest_max_drawdown",
    "backtest_avg_turnover",
    "backtest_avg_cost_drag",
    "flag_short_sample",
    "flag_negative_long_short",
    "flag_high_turnover",
    "score",
    "status",
    "error",
]


def _resolve_path(path_text: str | Path) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_run_dir_name(run_dir_name: str) -> tuple[str | None, str | None, str | None]:
    match = RUN_DIR_PATTERN.match(run_dir_name)
    if not match:
        return None, None, None
    return match.group("run_name"), match.group("timestamp"), match.group("config_hash")


def _iter_summary_files(runs_dirs: list[Path]) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for root in runs_dirs:
        if not root.exists():
            logging.warning("Runs dir not found; skip: %s", root)
            continue
        if not root.is_dir():
            logging.warning("Runs dir is not a directory; skip: %s", root)
            continue
        for summary_path in root.rglob("summary.json"):
            if not summary_path.is_file():
                continue
            resolved = summary_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append((root, resolved))
    entries.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
    return entries


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json top-level payload must be an object")
    return payload


def _load_used_config(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"config_parse_error: {exc}"
    if payload is None:
        return {}, None
    if not isinstance(payload, dict):
        return {}, "config_parse_error: config.used.yml top-level payload must be an object"
    return payload, None


def _init_row(source_runs_dir: Path, summary_path: Path) -> dict[str, Any]:
    row = {field: None for field in FIELDNAMES}
    row["source_runs_dir"] = str(source_runs_dir)
    row["summary_path"] = str(summary_path)
    row["run_dir"] = str(summary_path.parent)
    row["config_path"] = str(summary_path.parent / "config.used.yml")
    row["status"] = "ok"
    return row


def _apply_flags_and_score(row: dict[str, Any], args: argparse.Namespace) -> None:
    periods = _to_float(row.get("backtest_periods"))
    if periods is not None:
        row["flag_short_sample"] = periods < float(args.short_sample_periods)

    eval_long_short = _to_float(row.get("eval_long_short"))
    if eval_long_short is not None:
        row["flag_negative_long_short"] = eval_long_short < 0.0

    bt_turnover = _to_float(row.get("backtest_avg_turnover"))
    if bt_turnover is not None:
        row["flag_high_turnover"] = bt_turnover > float(args.high_turnover_threshold)

    sharpe = _to_float(row.get("backtest_sharpe"))
    if sharpe is None:
        return
    max_drawdown = _to_float(row.get("backtest_max_drawdown"))
    avg_cost_drag = _to_float(row.get("backtest_avg_cost_drag"))
    drawdown_penalty = abs(max_drawdown) if max_drawdown is not None else 0.0
    cost_penalty = avg_cost_drag if avg_cost_drag is not None else 0.0
    row["score"] = sharpe - float(args.score_drawdown_weight) * drawdown_penalty - float(
        args.score_cost_weight
    ) * cost_penalty


def _extract_row(source_runs_dir: Path, summary_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = _init_row(source_runs_dir, summary_path)
    errors: list[str] = []
    try:
        summary = _load_summary(summary_path)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"summary_parse_error: {exc}"
        return row

    run_dir = summary_path.parent
    config_path = run_dir / "config.used.yml"
    config, config_error = _load_used_config(config_path)
    if config_error:
        errors.append(config_error)

    fallback_name, fallback_timestamp, fallback_hash = _parse_run_dir_name(run_dir.name)

    row["run_name"] = _first_non_empty(
        _get_nested(summary, "run", "name"),
        fallback_name,
    )
    row["run_timestamp"] = _first_non_empty(
        _get_nested(summary, "run", "timestamp"),
        fallback_timestamp,
    )
    row["config_hash"] = _first_non_empty(
        _get_nested(summary, "run", "config_hash"),
        fallback_hash,
    )

    row["market"] = _first_non_empty(
        _get_nested(summary, "data", "market"),
        config.get("market") if isinstance(config, dict) else None,
    )
    row["data_provider"] = _first_non_empty(
        _get_nested(summary, "data", "provider"),
        _get_nested(config, "data", "provider"),
    )
    row["data_start_date"] = _first_non_empty(
        _get_nested(summary, "data", "start_date"),
        _get_nested(config, "data", "start_date"),
    )
    row["data_end_date"] = _first_non_empty(
        _get_nested(summary, "data", "end_date"),
        _get_nested(config, "data", "end_date"),
    )
    row["universe_mode"] = _first_non_empty(
        _get_nested(summary, "universe", "mode"),
        _get_nested(config, "universe", "mode"),
    )

    row["label_horizon_days"] = _first_non_empty(
        _get_nested(summary, "label", "horizon_days"),
        _get_nested(config, "label", "horizon_days"),
    )
    row["label_shift_days"] = _first_non_empty(
        _get_nested(summary, "label", "shift_days"),
        _get_nested(config, "label", "shift_days"),
    )

    row["eval_top_k"] = _first_non_empty(
        _get_nested(summary, "eval", "top_k"),
        _get_nested(config, "eval", "top_k"),
    )
    row["backtest_top_k"] = _first_non_empty(
        _get_nested(summary, "backtest", "top_k"),
        _get_nested(config, "backtest", "top_k"),
        row["eval_top_k"],
    )
    row["transaction_cost_bps"] = _first_non_empty(
        _get_nested(summary, "backtest", "transaction_cost_bps"),
        _get_nested(config, "backtest", "transaction_cost_bps"),
        _get_nested(config, "eval", "transaction_cost_bps"),
    )
    row["eval_rebalance_frequency"] = _first_non_empty(
        _get_nested(summary, "eval", "rebalance_frequency"),
        _get_nested(config, "eval", "rebalance_frequency"),
    )
    row["backtest_rebalance_frequency"] = _first_non_empty(
        _get_nested(summary, "backtest", "rebalance_frequency"),
        _get_nested(config, "backtest", "rebalance_frequency"),
        row["eval_rebalance_frequency"],
    )
    row["eval_buffer_exit"] = _first_non_empty(
        _get_nested(summary, "eval", "buffer_exit"),
        _get_nested(config, "eval", "buffer_exit"),
    )
    row["eval_buffer_entry"] = _first_non_empty(
        _get_nested(summary, "eval", "buffer_entry"),
        _get_nested(config, "eval", "buffer_entry"),
    )
    row["backtest_buffer_exit"] = _first_non_empty(
        _get_nested(summary, "backtest", "buffer_exit"),
        _get_nested(config, "backtest", "buffer_exit"),
        row["eval_buffer_exit"],
    )
    row["backtest_buffer_entry"] = _first_non_empty(
        _get_nested(summary, "backtest", "buffer_entry"),
        _get_nested(config, "backtest", "buffer_entry"),
        row["eval_buffer_entry"],
    )

    row["eval_ic_mean"] = _get_nested(summary, "eval", "ic", "mean")
    row["eval_ic_ir"] = _get_nested(summary, "eval", "ic", "ir")
    row["eval_long_short"] = _get_nested(summary, "eval", "long_short")
    row["eval_turnover_mean"] = _get_nested(summary, "eval", "turnover_mean")

    backtest_stats = _get_nested(summary, "backtest", "stats")
    if isinstance(backtest_stats, dict):
        row["backtest_periods"] = backtest_stats.get("periods")
        row["backtest_total_return"] = backtest_stats.get("total_return")
        row["backtest_ann_return"] = backtest_stats.get("ann_return")
        row["backtest_ann_vol"] = backtest_stats.get("ann_vol")
        row["backtest_sharpe"] = backtest_stats.get("sharpe")
        row["backtest_max_drawdown"] = backtest_stats.get("max_drawdown")
        row["backtest_avg_turnover"] = backtest_stats.get("avg_turnover")
        row["backtest_avg_cost_drag"] = backtest_stats.get("avg_cost_drag")
    else:
        row["status"] = "no_backtest"

    _apply_flags_and_score(row, args)
    if errors:
        row["error"] = "; ".join(errors)
    return row


def _resolve_output_path(args: argparse.Namespace, runs_dirs: list[Path]) -> Path:
    if args.output:
        return _resolve_path(args.output)
    default_root = runs_dirs[0] if runs_dirs else _resolve_path("out/runs")
    return default_root / "runs_summary.csv"


def add_summarize_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--runs-dir",
        action="append",
        default=None,
        help=(
            "Root directory to scan recursively for run folders (repeatable). "
            "Default: out/runs"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <first-runs-dir>/runs_summary.csv)",
    )
    parser.add_argument(
        "--short-sample-periods",
        type=int,
        default=24,
        help="Flag short sample when backtest_periods is below this value (default: 24)",
    )
    parser.add_argument(
        "--high-turnover-threshold",
        type=float,
        default=0.7,
        help="Flag high turnover when backtest_avg_turnover exceeds this value (default: 0.7)",
    )
    parser.add_argument(
        "--score-drawdown-weight",
        type=float,
        default=0.5,
        help="Score penalty weight for |max_drawdown| (default: 0.5)",
    )
    parser.add_argument(
        "--score-cost-weight",
        type=float,
        default=10.0,
        help="Score penalty weight for avg_cost_drag (default: 10.0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    runs_dirs = [_resolve_path(path) for path in (args.runs_dir or ["out/runs"])]
    summary_entries = _iter_summary_files(runs_dirs)
    if not summary_entries:
        targets = ", ".join(str(path) for path in runs_dirs)
        raise SystemExit(f"No summary.json files found under: {targets}")

    output_path = _resolve_output_path(args, runs_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_extract_row(source_dir, summary_path, args) for source_dir, summary_path in summary_entries]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Summary written to %s (%d runs)", output_path, len(rows))
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved run results from summary.json + config.used.yml"
    )
    add_summarize_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
