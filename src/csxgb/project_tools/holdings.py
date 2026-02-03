from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config_utils import resolve_pipeline_config


def _normalize_as_of_token(value: str | None) -> str:
    if value is None:
        return "t-1"
    text = str(value).strip()
    if not text:
        return "t-1"
    lowered = text.lower()
    if lowered in {"today", "t", "now"}:
        return "today"
    if lowered in {"t-1", "yesterday", "last_trading_day", "last_completed_trading_day"}:
        return "t-1"
    return text


def _resolve_as_of(value: str | None) -> pd.Timestamp:
    token = _normalize_as_of_token(value)
    today = pd.Timestamp.now().normalize()
    if token == "today":
        return today
    if token == "t-1":
        return today - pd.Timedelta(days=1)
    text = str(token).strip()
    compact = text.replace("-", "")
    if compact.isdigit() and len(compact) == 8:
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"Invalid --as-of date: {value}")
    return parsed.normalize()


def _resolve_output_dir(config_path: str | None) -> tuple[Path, str]:
    resolved = resolve_pipeline_config(config_path)
    cfg = resolved.data
    eval_cfg = cfg.get("eval") if isinstance(cfg, dict) else None
    eval_cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
    output_dir = eval_cfg.get("output_dir", "out/runs")
    run_name = eval_cfg.get("run_name") or resolved.label
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    return output_path, str(run_name)


def _find_latest_summary(output_dir: Path, run_name: str, top_k: int | None) -> Path | None:
    pattern = f"{run_name}_*/summary.json"
    candidates = list(output_dir.glob(pattern))
    if not candidates:
        return None
    if top_k is None:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    filtered: list[Path] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        backtest = payload.get("backtest") if isinstance(payload, dict) else None
        if not isinstance(backtest, dict):
            continue
        summary_top_k = backtest.get("top_k")
        if summary_top_k is None:
            continue
        try:
            if int(summary_top_k) == int(top_k):
                filtered.append(path)
        except (TypeError, ValueError):
            continue
    if not filtered:
        return None
    return max(filtered, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(config_path: str | None, run_dir: str | None, top_k: int | None) -> Path:
    if run_dir:
        candidate = Path(run_dir).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            raise SystemExit(f"Run directory not found: {candidate}")
        return candidate

    output_dir, run_name = _resolve_output_dir(config_path)
    summary_path = _find_latest_summary(output_dir, run_name, top_k)
    if summary_path is not None:
        return summary_path.parent
    if top_k is not None:
        raise SystemExit(
            f"No runs found for {run_name} with top_k={top_k} under {output_dir}"
        )
    candidates = [p for p in output_dir.glob(f"{run_name}_*") if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No runs found for {run_name} under {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_date_column(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if parsed.notna().any():
        return parsed.dt.normalize()
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.dt.normalize()


def _format_float(value: object, decimals: int) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{decimals}f}"


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    lines = [header_line, sep_line]
    for row in rows:
        lines.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def _render_text(df: pd.DataFrame, as_of: pd.Timestamp, entry_date: pd.Timestamp) -> str:
    rebalance_date = None
    if "rebalance_date" in df.columns:
        rebalance_date = str(df["rebalance_date"].iloc[0])
    lines = [
        f"As-of: {as_of.strftime('%Y-%m-%d')}",
        f"Entry date: {entry_date.strftime('%Y-%m-%d')}",
    ]
    if rebalance_date:
        lines.append(f"Rebalance date: {rebalance_date}")
    lines.append(f"Holdings: {len(df)}")

    display = df.copy()
    if "side" not in display.columns:
        display["side"] = "long"
    cols = ["ts_code", "weight", "signal", "rank", "side"]
    for col in cols:
        if col not in display.columns:
            display[col] = np.nan
    display = display[cols]
    rows = []
    for _, row in display.iterrows():
        rows.append(
            [
                str(row["ts_code"]),
                _format_float(row["weight"], 4),
                _format_float(row["signal"], 6),
                str(int(row["rank"])) if pd.notna(row["rank"]) else "",
                str(row["side"]),
            ]
        )
    lines.append("")
    lines.append(_format_table(rows, ["ts_code", "weight", "signal", "rank", "side"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show latest holdings from a saved run")
    parser.add_argument(
        "--config",
        help="Pipeline config path or built-in name (default: default).",
    )
    parser.add_argument(
        "--run-dir",
        help="Explicit run directory to read (overrides --config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional Top-K filter when selecting the latest run.",
    )
    parser.add_argument(
        "--as-of",
        default="t-1",
        help="As-of date (YYYYMMDD, YYYY-MM-DD, today, t-1). Default: t-1.",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "json"],
        help="Output format (text/csv/json). Default: text.",
    )
    parser.add_argument(
        "--out",
        help="Optional output path (default: stdout).",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args.config, args.run_dir, args.top_k)
    if args.top_k is not None:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = None
            backtest = summary.get("backtest") if isinstance(summary, dict) else None
            summary_top_k = backtest.get("top_k") if isinstance(backtest, dict) else None
            if summary_top_k is not None:
                try:
                    if int(summary_top_k) != int(args.top_k):
                        raise SystemExit(
                            f"Run top_k={summary_top_k} does not match requested --top-k={args.top_k}."
                        )
                except (TypeError, ValueError):
                    pass
    positions_path = run_dir / "positions_by_rebalance.csv"
    if not positions_path.exists():
        raise SystemExit(f"positions_by_rebalance.csv not found in {run_dir}")

    df = pd.read_csv(positions_path)
    if df.empty:
        raise SystemExit("positions_by_rebalance.csv is empty.")

    if "entry_date" not in df.columns:
        raise SystemExit("positions_by_rebalance.csv is missing entry_date.")

    entry_dates = _parse_date_column(df["entry_date"])
    if entry_dates.isna().all():
        raise SystemExit("Failed to parse entry_date column.")

    as_of = _resolve_as_of(args.as_of)
    eligible = entry_dates <= as_of
    if not eligible.any():
        raise SystemExit("No holdings available before the requested --as-of date.")
    latest_entry = entry_dates[eligible].max()
    selection = df[entry_dates == latest_entry].copy()
    if selection.empty:
        raise SystemExit("No holdings found for the latest entry date.")

    if "ts_code" not in selection.columns:
        raise SystemExit("positions_by_rebalance.csv is missing ts_code.")
    if "side" not in selection.columns:
        selection["side"] = "long"
    if "rank" not in selection.columns:
        selection["rank"] = np.nan

    selection.sort_values(["side", "rank", "ts_code"], inplace=True, na_position="last")

    if args.format == "text":
        content = _render_text(selection, as_of, latest_entry)
    elif args.format == "csv":
        content = selection.to_csv(index=False)
    else:
        payload = {
            "as_of": as_of.strftime("%Y-%m-%d"),
            "entry_date": latest_entry.strftime("%Y-%m-%d"),
            "rebalance_date": selection["rebalance_date"].iloc[0]
            if "rebalance_date" in selection.columns
            else None,
            "holdings": selection.to_dict(orient="records"),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(content)


if __name__ == "__main__":
    main()
