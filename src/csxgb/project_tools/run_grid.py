from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from ..backtest import backtest_topk
from ..config_utils import resolve_pipeline_config
from ..metrics import daily_ic_series, estimate_turnover, quantile_returns, summarize_ic
from ..rebalance import get_rebalance_dates


def _resolve_output_path(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _parse_int_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for entry in values:
        for part in entry.split(","):
            text = part.strip()
            if not text:
                continue
            items.append(int(text))
    return items


def _parse_float_list(values: list[str]) -> list[float]:
    items: list[float] = []
    for entry in values:
        for part in entry.split(","):
            text = part.strip()
            if not text:
                continue
            items.append(float(text))
    return items


def _safe_run_name(base: str, top_k: int, cost_bps: float) -> str:
    cost_text = ("%g" % cost_bps).replace(".", "p")
    return f"{base}_k{top_k}_bps{cost_text}"


def _find_latest_summary(output_dir: Path, run_name: str) -> Path | None:
    pattern = f"{run_name}_*/summary.json"
    candidates = list(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_nested(payload: dict, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_date_list(values: Optional[list[str]]) -> list[pd.Timestamp]:
    if not values:
        return []
    out: list[pd.Timestamp] = []
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            continue
        out.append(pd.Timestamp(parsed))
    out = sorted(out)
    deduped: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for date in out:
        if date in seen:
            continue
        seen.add(date)
        deduped.append(date)
    return deduped


def _resolve_rebalance_dates(
    summary_dates: Optional[list[str]],
    scored_data: pd.DataFrame,
    frequency: str,
    min_symbols_per_date: int,
) -> list[pd.Timestamp]:
    parsed = _parse_date_list(summary_dates)
    if parsed:
        available = set(pd.to_datetime(scored_data["trade_date"].unique()))
        return [date for date in parsed if date in available]

    trade_dates = sorted(pd.to_datetime(scored_data["trade_date"].unique()))
    rebalance_dates = get_rebalance_dates(trade_dates, frequency)
    if min_symbols_per_date > 1:
        counts = scored_data.groupby("trade_date")["ts_code"].nunique()
        valid_dates = set(counts[counts >= min_symbols_per_date].index)
        rebalance_dates = [date for date in rebalance_dates if date in valid_dates]
    return rebalance_dates


def _init_row(top_k: int, cost_bps: float, run_name: str, summary_path: Path | None) -> dict:
    return {
        "run_name": run_name,
        "top_k": top_k,
        "cost_bps": cost_bps,
        "summary_path": str(summary_path) if summary_path else None,
        "output_dir": None,
        "label_horizon_days": None,
        "eval_ic_mean": None,
        "eval_ic_ir": None,
        "eval_long_short": None,
        "eval_turnover_mean": None,
        "backtest_periods": None,
        "backtest_total_return": None,
        "backtest_ann_return": None,
        "backtest_ann_vol": None,
        "backtest_sharpe": None,
        "backtest_max_drawdown": None,
        "backtest_avg_turnover": None,
        "backtest_avg_cost_drag": None,
        "status": "ok",
        "error": None,
    }


def add_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="config/hk.yml",
        help="Base config path or built-in name (default: config/hk.yml)",
    )
    parser.add_argument(
        "--top-k",
        action="append",
        default=None,
        help="Comma-separated top_k values (default: 5,10,20)",
    )
    parser.add_argument(
        "--cost-bps",
        action="append",
        default=None,
        help="Comma-separated cost bps per side (default: 15,25,40)",
    )
    parser.add_argument(
        "--output",
        default="out/runs/grid_summary.csv",
        help="Output CSV path (default: out/runs/grid_summary.csv)",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Optional prefix for run_name (default: base config stem)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a grid of Top-K and cost bps settings")
    add_grid_args(parser)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    resolved = resolve_pipeline_config(args.config)
    base_cfg = resolved.data
    base_label = args.run_name_prefix or resolved.label

    top_k_entries = args.top_k or ["5,10,20"]
    cost_entries = args.cost_bps or ["15,25,40"]
    top_k_values = list(dict.fromkeys(_parse_int_list(top_k_entries)))
    cost_values = list(dict.fromkeys(_parse_float_list(cost_entries)))
    combos = [(top_k, cost) for top_k in top_k_values for cost in cost_values]
    if not combos:
        raise SystemExit("No valid parameter combinations.")

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_run_name = f"{base_label}_grid_base"
    base_run_cfg = copy.deepcopy(base_cfg)
    base_run_cfg.setdefault("eval", {})
    base_run_cfg.setdefault("backtest", {})
    base_run_cfg["eval"]["run_name"] = base_run_name
    base_run_cfg["eval"]["save_artifacts"] = True
    base_run_cfg["eval"]["top_k"] = int(top_k_values[0])
    base_run_cfg["backtest"]["top_k"] = int(top_k_values[0])
    base_run_cfg["eval"]["transaction_cost_bps"] = float(cost_values[0])
    base_run_cfg["backtest"]["transaction_cost_bps"] = float(cost_values[0])

    from .. import pipeline

    logging.info("Running base pipeline once for %s combinations ...", len(combos))
    with tempfile.TemporaryDirectory(prefix="csxgb_grid_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        cfg_path = tmp_dir_path / f"{base_run_name}.yml"
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(base_run_cfg, handle, sort_keys=False)
        pipeline.run(str(cfg_path))

    output_dir = _resolve_output_path(str(base_run_cfg.get("eval", {}).get("output_dir", "out/runs")))
    summary_path = _find_latest_summary(output_dir, base_run_name)
    if summary_path is None or not summary_path.exists():
        raise SystemExit(f"Base run summary not found for run_name={base_run_name}.")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    run_dir = _resolve_output_path(str(_get_nested(summary, "run", "output_dir") or output_dir))
    scored_file = _get_nested(summary, "eval", "scored_file")
    if not scored_file:
        raise SystemExit("Grid requires eval.scored_file artifact from pipeline run.")

    scored_path = Path(str(scored_file))
    if not scored_path.is_absolute():
        candidate = (run_dir / scored_path).resolve()
        scored_path = candidate if candidate.exists() else _resolve_output_path(str(scored_file))
    if not scored_path.exists():
        raise SystemExit(f"Scored data file not found: {scored_path}")

    scored_data = pd.read_parquet(scored_path)
    if scored_data.empty:
        raise SystemExit("Scored data is empty; cannot run grid.")
    scored_data["trade_date"] = pd.to_datetime(scored_data["trade_date"])

    eval_signal_col = str(_get_nested(summary, "eval", "scored_signal_col") or "signal_eval")
    if eval_signal_col not in scored_data.columns:
        eval_signal_col = "pred"
    bt_signal_col = str(
        _get_nested(summary, "eval", "scored_signal_backtest_col") or "signal_backtest"
    )
    if bt_signal_col not in scored_data.columns:
        bt_signal_col = eval_signal_col

    target_col = str(base_run_cfg.get("label", {}).get("target_col", "future_return"))
    price_col = str(base_run_cfg.get("data", {}).get("price_col", "close"))
    for required in ("trade_date", "ts_code", target_col, price_col, eval_signal_col):
        if required not in scored_data.columns:
            raise SystemExit(f"Missing required column in scored data: {required}")

    min_symbols_per_date = int(_get_nested(summary, "data", "min_symbols_per_date") or 1)
    eval_frequency = str(
        _get_nested(summary, "eval", "rebalance_frequency")
        or base_run_cfg.get("eval", {}).get("rebalance_frequency", "W")
    )
    backtest_frequency = str(
        _get_nested(summary, "backtest", "rebalance_frequency")
        or base_run_cfg.get("backtest", {}).get("rebalance_frequency", eval_frequency)
    )
    eval_rebalance_dates = _resolve_rebalance_dates(
        _get_nested(summary, "eval", "rebalance_dates"),
        scored_data,
        eval_frequency,
        min_symbols_per_date,
    )
    backtest_rebalance_dates = _resolve_rebalance_dates(
        _get_nested(summary, "backtest", "rebalance_dates"),
        scored_data,
        backtest_frequency,
        min_symbols_per_date,
    )

    eval_cfg = base_run_cfg.get("eval", {}) if isinstance(base_run_cfg.get("eval"), dict) else {}
    backtest_cfg = (
        base_run_cfg.get("backtest", {}) if isinstance(base_run_cfg.get("backtest"), dict) else {}
    )
    label_cfg = base_run_cfg.get("label", {}) if isinstance(base_run_cfg.get("label"), dict) else {}

    n_quantiles = int(eval_cfg.get("n_quantiles", 5))
    eval_buffer_exit = int(eval_cfg.get("buffer_exit", backtest_cfg.get("buffer_exit", 0)))
    eval_buffer_entry = int(eval_cfg.get("buffer_entry", backtest_cfg.get("buffer_entry", 0)))

    backtest_enabled = bool(backtest_cfg.get("enabled", True))
    backtest_long_only = bool(backtest_cfg.get("long_only", True))
    backtest_short_k = backtest_cfg.get("short_k")
    if backtest_short_k is not None:
        backtest_short_k = int(backtest_short_k)
    backtest_shift_days = int(_get_nested(summary, "backtest", "shift_days") or label_cfg.get("shift_days", 0))
    backtest_trading_days = int(
        _get_nested(summary, "backtest", "trading_days_per_year")
        or backtest_cfg.get("trading_days_per_year", 252)
    )
    backtest_exit_mode = str(backtest_cfg.get("exit_mode", "rebalance")).strip().lower()
    backtest_exit_horizon_days = _get_nested(summary, "backtest", "exit_horizon_days")
    if backtest_exit_horizon_days is None:
        backtest_exit_horizon_days = backtest_cfg.get("exit_horizon_days")
    if backtest_exit_mode == "label_horizon" and backtest_exit_horizon_days is None:
        backtest_exit_horizon_days = label_cfg.get("horizon_days")
    if backtest_exit_horizon_days is not None:
        backtest_exit_horizon_days = int(backtest_exit_horizon_days)
    backtest_buffer_exit = int(backtest_cfg.get("buffer_exit", 0))
    backtest_buffer_entry = int(backtest_cfg.get("buffer_entry", 0))
    backtest_exit_price_policy = str(
        _get_nested(summary, "backtest", "exit_price_policy")
        or backtest_cfg.get("exit_price_policy", "strict")
    ).strip().lower()
    backtest_exit_fallback_policy = str(
        _get_nested(summary, "backtest", "exit_fallback_policy")
        or backtest_cfg.get("exit_fallback_policy", "ffill")
    ).strip().lower()
    tradable_col = _get_nested(summary, "backtest", "tradable_col")
    if tradable_col is None:
        tradable_col = backtest_cfg.get("tradable_col", "is_tradable")
    tradable_col = str(tradable_col).strip() if tradable_col is not None else None
    if tradable_col and tradable_col not in scored_data.columns:
        tradable_col = None

    rows: list[dict] = []
    for top_k, cost_bps in combos:
        run_name = _safe_run_name(base_label, top_k, cost_bps)
        row = _init_row(top_k, cost_bps, run_name, summary_path)
        row["output_dir"] = _get_nested(summary, "run", "output_dir")
        row["label_horizon_days"] = _get_nested(summary, "label", "horizon_days")
        try:
            eval_slice = scored_data[scored_data["trade_date"].isin(eval_rebalance_dates)].copy()
            ic_stats = summarize_ic(daily_ic_series(eval_slice, target_col, eval_signal_col))
            row["eval_ic_mean"] = ic_stats.get("mean")
            row["eval_ic_ir"] = ic_stats.get("ir")

            quantile_ts = quantile_returns(eval_slice, eval_signal_col, target_col, n_quantiles)
            quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
            row["eval_long_short"] = (
                float(quantile_mean.iloc[-1] - quantile_mean.iloc[0])
                if not quantile_mean.empty
                else None
            )

            k = min(int(top_k), eval_slice["ts_code"].nunique()) if not eval_slice.empty else 0
            if k > 0 and eval_rebalance_dates:
                turnover = estimate_turnover(
                    eval_slice,
                    eval_signal_col,
                    k,
                    eval_rebalance_dates,
                    buffer_exit=eval_buffer_exit,
                    buffer_entry=eval_buffer_entry,
                )
                row["eval_turnover_mean"] = float(turnover.mean()) if not turnover.empty else None

            if backtest_enabled:
                bt_result = backtest_topk(
                    scored_data,
                    pred_col=bt_signal_col,
                    price_col=price_col,
                    rebalance_dates=backtest_rebalance_dates,
                    top_k=int(top_k),
                    shift_days=backtest_shift_days,
                    cost_bps=float(cost_bps),
                    trading_days_per_year=backtest_trading_days,
                    exit_mode=backtest_exit_mode,
                    exit_horizon_days=backtest_exit_horizon_days,
                    long_only=backtest_long_only,
                    short_k=backtest_short_k,
                    buffer_exit=backtest_buffer_exit,
                    buffer_entry=backtest_buffer_entry,
                    tradable_col=tradable_col,
                    exit_price_policy=backtest_exit_price_policy,
                    exit_fallback_policy=backtest_exit_fallback_policy,
                )
                if bt_result is None:
                    row["status"] = "no_backtest"
                else:
                    bt_stats, _, _, _, _ = bt_result
                    row["backtest_periods"] = bt_stats.get("periods")
                    row["backtest_total_return"] = bt_stats.get("total_return")
                    row["backtest_ann_return"] = bt_stats.get("ann_return")
                    row["backtest_ann_vol"] = bt_stats.get("ann_vol")
                    row["backtest_sharpe"] = bt_stats.get("sharpe")
                    row["backtest_max_drawdown"] = bt_stats.get("max_drawdown")
                    row["backtest_avg_turnover"] = bt_stats.get("avg_turnover")
                    row["backtest_avg_cost_drag"] = bt_stats.get("avg_cost_drag")
            else:
                row["status"] = "no_backtest"
        except Exception as exc:  # pragma: no cover - defensive
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)

    fieldnames = [
        "run_name",
        "top_k",
        "cost_bps",
        "summary_path",
        "output_dir",
        "label_horizon_days",
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
        "status",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Summary written to %s", output_path)


if __name__ == "__main__":
    main()
