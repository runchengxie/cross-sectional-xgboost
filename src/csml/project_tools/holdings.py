from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..config_utils import resolve_pipeline_config
from ..date_utils import resolve_date_token
from .symbols import ensure_symbol_columns


def _normalize_provider(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"rqdata", "rqdatac"}:
        return "rqdata"
    return text


def _normalize_market(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _resolve_date_context(
    config_path: str | None,
    summary: object | None,
) -> tuple[str | None, str | None]:
    market = None
    provider = None

    if isinstance(summary, dict):
        data = summary.get("data")
        if isinstance(data, dict):
            market = _normalize_market(data.get("market"))
            provider = _normalize_provider(data.get("provider"))

    if (market is not None and provider is not None) or not config_path:
        return market, provider

    try:
        resolved = resolve_pipeline_config(config_path)
    except Exception:
        return market, provider
    cfg = resolved.data if isinstance(resolved.data, dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg, dict) else None
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    rq_cfg = data_cfg.get("rqdata") if isinstance(data_cfg, dict) else None
    rq_cfg = rq_cfg if isinstance(rq_cfg, dict) else {}

    if market is None:
        market = _normalize_market(
            cfg.get("market")
            or data_cfg.get("market")
            or rq_cfg.get("market")
        )
    if provider is None:
        provider = _normalize_provider(data_cfg.get("provider"))
    return market, provider


def _resolve_as_of(
    value: str | None,
    *,
    market: str | None = None,
    provider: str | None = None,
) -> pd.Timestamp:
    try:
        return resolve_date_token(
            value,
            default="t-1",
            market=market,
            provider=provider,
            warn_to_stderr=True,
            warn_label="--as-of",
        )
    except SystemExit as exc:
        # Keep CLI-facing error text stable.
        raise SystemExit(f"Invalid --as-of date: {value}") from exc


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
    if top_k is None:
        latest_path = output_dir / "latest.json"
        if latest_path.exists():
            try:
                latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                latest_payload = None
            if isinstance(latest_payload, dict):
                latest_name = latest_payload.get("run_name")
                if latest_name and str(latest_name) != str(run_name):
                    latest_payload = None
            if isinstance(latest_payload, dict):
                latest_dir = latest_payload.get("run_dir")
                if latest_dir:
                    candidate = Path(latest_dir).expanduser()
                    if not candidate.is_absolute():
                        candidate = (Path.cwd() / candidate).resolve()
                    if candidate.exists():
                        return candidate
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


def _resolve_summary_path(value: object, run_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    raw = Path(text)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((run_dir / raw).resolve())
    candidates.append(run_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def _format_date_value(value: object) -> str | None:
    if value is None:
        return None
    parsed = _parse_date_column(pd.Series([value])).iloc[0]
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        if unicodedata.east_asian_width(char) in {"F", "W"}:
            width += 2
            continue
        width += 1
    return width


def _ljust_display(text: str, width: int) -> str:
    return text + (" " * max(0, width - _display_width(text)))


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [_display_width(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], _display_width(cell))
    header_line = "  ".join(_ljust_display(header, widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    lines = [header_line, sep_line]
    for row in rows:
        lines.append("  ".join(_ljust_display(row[idx], widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def _render_text(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    entry_date: pd.Timestamp,
    *,
    data_end_date: pd.Timestamp | None = None,
    source: str | None = None,
    run_dir: Path | None = None,
    positions_path: Path | None = None,
) -> str:
    rebalance_date = None
    if "rebalance_date" in df.columns:
        rebalance_date = str(df["rebalance_date"].iloc[0])
    signal_asof = None
    if "signal_asof" in df.columns:
        signal_asof = str(df["signal_asof"].iloc[0])
    elif rebalance_date:
        signal_asof = rebalance_date
    next_entry_date = None
    if "next_entry_date" in df.columns:
        next_entry_date = str(df["next_entry_date"].iloc[0]) or None
    lines = [
        f"As-of: {as_of.strftime('%Y-%m-%d')}",
        f"Entry date: {entry_date.strftime('%Y-%m-%d')}",
    ]
    if signal_asof:
        formatted_signal = _format_date_value(signal_asof)
        if formatted_signal:
            lines.append(f"Signal as-of: {formatted_signal}")
    if next_entry_date:
        formatted_next = _format_date_value(next_entry_date)
        if formatted_next:
            lines.append(
                f"Holding window: {entry_date.strftime('%Y-%m-%d')} -> {formatted_next} (next rebalance)"
            )
    if data_end_date is not None:
        lines.append(f"Data end date: {data_end_date.strftime('%Y-%m-%d')}")
    if source:
        lines.append(f"Source: {source}")
    if run_dir is not None:
        lines.append(f"Run dir: {run_dir}")
    if positions_path is not None:
        lines.append(f"Positions file: {positions_path}")
    if rebalance_date:
        formatted_rebalance = _format_date_value(rebalance_date)
        lines.append(f"Rebalance date: {formatted_rebalance or rebalance_date}")
    lines.append(f"Holdings: {len(df)}")
    if data_end_date is not None and as_of > data_end_date:
        lines.append(
            "Warning: as-of is after data end date; holdings may be stale."
        )

    display = df.copy()
    if "side" not in display.columns:
        display["side"] = "long"
    cols = ["stock_ticker", "weight", "signal", "rank", "side"]
    for col in cols:
        if col not in display.columns:
            display[col] = np.nan
    display = display[cols]
    rows = []
    for _, row in display.iterrows():
        rows.append(
            [
                str(row["stock_ticker"]),
                _format_float(row["weight"], 4),
                _format_float(row["signal"], 6),
                str(int(row["rank"])) if pd.notna(row["rank"]) else "",
                str(row["side"]),
            ]
        )
    lines.append("")
    lines.append(_format_table(rows, ["stock_ticker", "weight", "signal", "rank", "side"]))
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
        "--source",
        default="auto",
        choices=["auto", "backtest", "live"],
        help="Positions source (auto/backtest/live). Default: auto.",
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
    summary = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None
    date_market, date_provider = _resolve_date_context(args.config, summary)

    def _pick_summary_positions_path(kind: str) -> Path | None:
        if not isinstance(summary, dict):
            return None
        if kind == "live":
            live_section = summary.get("live")
            if not isinstance(live_section, dict):
                return None
            for key in ("positions_file", "current_file"):
                resolved = _resolve_summary_path(live_section.get(key), run_dir)
                if resolved is not None:
                    return resolved
            return None
        positions_section = summary.get("positions")
        if not isinstance(positions_section, dict):
            return None
        for key in ("by_rebalance_file", "current_file"):
            resolved = _resolve_summary_path(positions_section.get(key), run_dir)
            if resolved is not None:
                return resolved
        return None

    def _pick_positions_path(kind: str) -> Path | None:
        if kind == "live":
            live_path = run_dir / "positions_by_rebalance_live.csv"
            if live_path.exists():
                return live_path
            live_current = run_dir / "positions_current_live.csv"
            if live_current.exists():
                return live_current
            return None
        backtest_path = run_dir / "positions_by_rebalance.csv"
        if backtest_path.exists():
            return backtest_path
        backtest_current = run_dir / "positions_current.csv"
        if backtest_current.exists():
            return backtest_current
        return None

    def _resolve_positions_path(kind: str) -> Path | None:
        return _pick_summary_positions_path(kind) or _pick_positions_path(kind)

    source = args.source
    if source == "auto":
        positions_path = _resolve_positions_path("live")
        if positions_path is not None:
            source = "live"
        else:
            positions_path = _resolve_positions_path("backtest")
            source = "backtest"
    elif source == "live":
        positions_path = _resolve_positions_path("live")
    else:
        positions_path = _resolve_positions_path("backtest")

    if positions_path is None or not positions_path.exists():
        raise SystemExit(f"No positions file found for source '{source}' in {run_dir}")

    df = pd.read_csv(positions_path)
    if df.empty:
        raise SystemExit(f"{positions_path.name} is empty.")

    if "entry_date" not in df.columns:
        raise SystemExit(f"{positions_path.name} is missing entry_date.")

    entry_dates = _parse_date_column(df["entry_date"])
    if entry_dates.isna().all():
        raise SystemExit("Failed to parse entry_date column.")

    as_of = _resolve_as_of(
        args.as_of,
        market=date_market,
        provider=date_provider,
    )
    eligible = entry_dates <= as_of
    if not eligible.any():
        raise SystemExit("No holdings available before the requested --as-of date.")
    latest_entry = entry_dates[eligible].max()
    selection = df[entry_dates == latest_entry].copy()
    if selection.empty:
        raise SystemExit("No holdings found for the latest entry date.")

    selection = ensure_symbol_columns(selection, context=positions_path.name)
    if "side" not in selection.columns:
        selection["side"] = "long"
    if "rank" not in selection.columns:
        selection["rank"] = np.nan

    selection.sort_values(["side", "rank", "ts_code"], inplace=True, na_position="last")

    data_end_date = None
    if isinstance(summary, dict):
        data_end_raw = summary.get("data", {}).get("end_date")
        if data_end_raw:
            data_end_date = _parse_date_column(pd.Series([data_end_raw])).iloc[0]
            if pd.isna(data_end_date):
                data_end_date = None

    if args.format == "text":
        content = _render_text(
            selection,
            as_of,
            latest_entry,
            data_end_date=data_end_date,
            source=source,
            run_dir=run_dir,
            positions_path=positions_path,
        )
    elif args.format == "csv":
        content = selection.to_csv(index=False)
    else:
        signal_asof = None
        if "signal_asof" in selection.columns:
            signal_asof = selection["signal_asof"].iloc[0]
        elif "rebalance_date" in selection.columns:
            signal_asof = selection["rebalance_date"].iloc[0]
        next_entry_date = None
        if "next_entry_date" in selection.columns:
            next_entry_date = selection["next_entry_date"].iloc[0]
        holding_window = None
        formatted_next = _format_date_value(next_entry_date) if next_entry_date else None
        if formatted_next:
            holding_window = (
                f"{latest_entry.strftime('%Y-%m-%d')} -> {formatted_next} (next rebalance)"
            )
        payload = {
            "as_of": as_of.strftime("%Y-%m-%d"),
            "entry_date": latest_entry.strftime("%Y-%m-%d"),
            "signal_asof": _format_date_value(signal_asof) if signal_asof else None,
            "next_entry_date": formatted_next,
            "holding_window": holding_window,
            "rebalance_date": _format_date_value(selection["rebalance_date"].iloc[0])
            if "rebalance_date" in selection.columns
            else None,
            "data_end_date": data_end_date.strftime("%Y-%m-%d") if data_end_date is not None else None,
            "market": date_market,
            "data_provider": date_provider,
            "source": source,
            "run_dir": str(run_dir),
            "positions_file": str(positions_path),
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
