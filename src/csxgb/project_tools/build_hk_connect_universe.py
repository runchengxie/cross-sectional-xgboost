# -*- coding: utf-8 -*-
"""Build a PIT HK Connect universe with liquidity filtering."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import yaml

from ..config_utils import resolve_config

DEFAULTS = {
    "mode": "backtest",
    "start_date": None,
    "end_date": None,
    "as_of": None,
    "rebalance_frequency": "M",
    "lookback_days": 60,
    "min_window_days": 30,
    "top_quantile": 0.8,
    "min_turnover": 0.0,
    "out": "out/universe/universe_by_date.csv",
    "latest_out": "out/universe/hk_connect_symbols.txt",
    "append_date": True,
    "write_meta": True,
    "meta_out": "out/universe/universe_by_date.meta.yml",
    "rqdata_user": None,
    "rqdata_pass": None,
}


def validate_yyyymmdd(value: str, label: str) -> str:
    if not value or len(value) != 8 or not value.isdigit():
        raise SystemExit(f"{label} must be in YYYYMMDD format.")
    return value


def parse_date(value: str) -> pd.Timestamp:
    value = validate_yyyymmdd(value, "date")
    return pd.to_datetime(value, format="%Y%m%d")


def normalize_mode(value: str | None) -> str:
    if not value:
        return "backtest"
    text = str(value).strip().lower()
    if text in {"backtest", "bt"}:
        return "backtest"
    if text in {"daily", "live", "update"}:
        return "daily"
    if text in {"as_of", "asof"}:
        return "as_of"
    raise SystemExit(f"Unsupported mode: {value}")


def normalize_date_input(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y%m%d")
    if isinstance(value, (int, float)) and not pd.isna(value):
        value = str(int(value))
    text = str(value).strip()
    if not text:
        return None
    return text


def normalize_date_token(value: object, label: str) -> str | None:
    text = normalize_date_input(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"today", "t"}:
        return "today"
    if lowered in {"t-1", "yesterday", "last_trading_day", "last_completed_trading_day"}:
        return "t-1"
    return validate_yyyymmdd(text, label)


def load_yaml_config(path: str | Path | None) -> dict:
    resolved = resolve_config(
        path,
        package="csxgb.config",
        default_name="universe.hk_connect.yml",
    )
    return resolved.data


def extract_universe_config(cfg: dict) -> dict:
    if not cfg:
        return {}
    if isinstance(cfg.get("hk_connect_universe"), dict):
        cfg = cfg["hk_connect_universe"]
    normalized = {}
    for key, value in cfg.items():
        normalized[str(key).strip().replace("-", "_")] = value
    rq_cfg = cfg.get("rqdata")
    if isinstance(rq_cfg, dict):
        if "username" in rq_cfg and "rqdata_user" not in normalized:
            normalized["rqdata_user"] = rq_cfg.get("username")
        if "password" in rq_cfg and "rqdata_pass" not in normalized:
            normalized["rqdata_pass"] = rq_cfg.get("password")
    return normalized


def merge_settings(defaults: dict, cfg: dict, cli: dict) -> dict:
    merged = defaults.copy()
    for key, value in cfg.items():
        if key in defaults and value is not None:
            merged[key] = value
    for key, value in cli.items():
        if key in defaults and value is not None:
            merged[key] = value
    return merged


def normalize_hk_symbol(order_book_id: str) -> str:
    text = str(order_book_id or "").strip().upper()
    if not text:
        return text
    if text.endswith(".XHKG"):
        text = text[:-5]
    if text.endswith(".HK"):
        text = text[:-3]
    if text.isdigit():
        text = text.zfill(5)
    return f"{text}.HK"


def get_rebalance_dates(dates: list[pd.Timestamp], freq: str) -> list[pd.Timestamp]:
    if not freq or freq.upper() == "D":
        return list(dates)
    date_df = pd.DataFrame({"date": pd.to_datetime(dates)})
    date_df["period"] = date_df["date"].dt.to_period(freq)
    return date_df.groupby("period")["date"].max().sort_values().tolist()


def coerce_trading_dates(dates) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(list(dates)))


def require_rqdata(username: str | None, password: str | None):
    try:
        import rqdatac
    except ImportError as exc:
        raise SystemExit(f"rqdatac is required ({exc}).")
    try:
        import rqdatac_hk  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"rqdatac_hk is required ({exc}).")

    init_kwargs = {}
    if username:
        init_kwargs["username"] = username
    if password:
        init_kwargs["password"] = password
    try:
        rqdatac.init(**init_kwargs)
    except Exception as exc:
        raise SystemExit(f"rqdatac.init failed: {exc}")
    return rqdatac


def resolve_last_trading_date(
    rqdatac,
    as_of: pd.Timestamp,
    market: str,
    include_today: bool,
) -> pd.Timestamp:
    as_of = pd.to_datetime(as_of).normalize()
    lookbacks = [366, 365 * 5]
    for days in lookbacks:
        start = (as_of - pd.Timedelta(days=days)).strftime("%Y%m%d")
        end = as_of.strftime("%Y%m%d")
        dates = rqdatac.get_trading_dates(start, end, market=market)
        if not dates:
            continue
        candidates = [d.normalize() for d in pd.to_datetime(dates)]
        if include_today:
            candidates = [d for d in candidates if d <= as_of]
        else:
            candidates = [d for d in candidates if d < as_of]
        if candidates:
            return max(candidates)
    raise SystemExit("Unable to resolve a trading date from the calendar.")


def resolve_as_of_date(rqdatac, token: str, market: str) -> pd.Timestamp:
    token = token.lower()
    today = pd.Timestamp.now().normalize()
    if token == "today":
        return resolve_last_trading_date(rqdatac, today, market, include_today=True)
    if token == "t-1":
        return resolve_last_trading_date(rqdatac, today, market, include_today=False)
    return resolve_last_trading_date(rqdatac, parse_date(token), market, include_today=True)


def format_output_path(path_value: str | None, date_tag: str, append_date: bool) -> Path | None:
    if not path_value:
        return None
    text = str(path_value)
    if "{as_of}" in text or "{date}" in text:
        text = text.format(as_of=date_tag, date=date_tag)
        return Path(text)
    if not append_date:
        return Path(text)
    path = Path(text)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    new_name = f"{stem}_{date_tag}{suffix}"
    return path.with_name(new_name)


def default_meta_path(out_path: Path) -> Path:
    suffix = "".join(out_path.suffixes)
    stem = out_path.name[: -len(suffix)] if suffix else out_path.name
    return out_path.with_name(f"{stem}.meta.yml")


def fetch_southbound_membership(rqdatac, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, set[str]]:
    membership = {}
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        sh_list = rqdatac.hk.get_southbound_eligible_secs(trading_type="sh", date=date_str)
        sz_list = rqdatac.hk.get_southbound_eligible_secs(trading_type="sz", date=date_str)
        combined = set(sh_list or []) | set(sz_list or [])
        membership[date.normalize()] = combined
    return membership


def prepare_turnover_table(
    rqdatac,
    order_book_ids: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df = rqdatac.get_price(
        order_book_ids,
        start_date,
        end_date,
        frequency="1d",
        fields=["total_turnover"],
        market="hk",
        expect_df=True,
    )
    if df is None or df.empty:
        raise SystemExit("get_price returned no turnover data.")

    if isinstance(df.index, pd.MultiIndex):
        if "order_book_id" in df.index.names:
            turnover = df["total_turnover"].unstack("order_book_id")
        else:
            turnover = df["total_turnover"].unstack(level=0)
    else:
        turnover = df[["total_turnover"]].rename(columns={"total_turnover": order_book_ids[0]})

    turnover.index = pd.to_datetime(turnover.index)
    return turnover.sort_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build HK Connect universe (PIT + liquidity).")
    parser.add_argument(
        "--config",
        help="YAML config path (optional). If omitted, uses packaged default.",
    )
    parser.add_argument("--mode", choices=["backtest", "daily"], help="Mode: backtest or daily")
    parser.add_argument("--start-date", help="Start date in YYYYMMDD")
    parser.add_argument("--end-date", help="End date in YYYYMMDD (or T-1/today in daily mode)")
    parser.add_argument("--as-of", help="Single as-of date in YYYYMMDD (overrides mode)")
    parser.add_argument("--rebalance-frequency", help="Rebalance frequency (default: M)")
    parser.add_argument("--lookback-days", type=int, help="Lookback trading days (default: 60)")
    parser.add_argument("--min-window-days", type=int, help="Minimum valid days in window (default: 30)")
    parser.add_argument(
        "--top-quantile",
        type=float,
        help="Liquidity quantile threshold; 0.8 keeps top 20 percent by turnover",
    )
    parser.add_argument(
        "--min-turnover",
        type=float,
        help="Absolute turnover threshold applied before quantile (default: 0)",
    )
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--latest-out", help="Optional symbols_file output for latest rebalance date")
    parser.add_argument("--meta-out", help="Optional meta YAML output path")
    parser.add_argument(
        "--append-date",
        dest="append_date",
        action="store_true",
        default=None,
        help="Append as-of date to output filenames in daily/as-of mode",
    )
    parser.add_argument(
        "--no-append-date",
        dest="append_date",
        action="store_false",
        help="Disable date suffix in output filenames",
    )
    parser.add_argument(
        "--write-meta",
        dest="write_meta",
        action="store_true",
        default=None,
        help="Write meta YAML alongside the output CSV",
    )
    parser.add_argument(
        "--no-write-meta",
        dest="write_meta",
        action="store_false",
        help="Disable meta YAML output",
    )
    parser.add_argument("--rqdata-user", help="RQData username (optional)")
    parser.add_argument("--rqdata-pass", help="RQData password (optional)")
    args = parser.parse_args(argv)

    cfg = extract_universe_config(load_yaml_config(args.config))
    settings = merge_settings(DEFAULTS, cfg, vars(args))

    mode = normalize_mode(settings.get("mode"))
    start_token = normalize_date_token(settings.get("start_date"), "start-date")
    end_token = normalize_date_token(settings.get("end_date"), "end-date")
    as_of_token = normalize_date_token(settings.get("as_of"), "as-of")

    if start_token in {"today", "t-1"}:
        raise SystemExit("start-date must be in YYYYMMDD format.")

    if mode == "backtest" and end_token in {"today", "t-1"}:
        mode = "daily"
    if mode == "daily" and args.end_date is None and not as_of_token:
        if end_token not in {None, "today", "t-1"}:
            end_token = None

    load_dotenv()
    rq_user = settings.get("rqdata_user") or os.getenv("RQDATA_USERNAME") or os.getenv("RQDATA_USER")
    rq_pass = settings.get("rqdata_pass") or os.getenv("RQDATA_PASSWORD")
    rqdatac = require_rqdata(rq_user, rq_pass)

    market = "hk"
    as_of_input = None
    end_source = "fixed"
    if as_of_token:
        as_of_input = as_of_token
        as_of_date = resolve_as_of_date(rqdatac, as_of_token, market)
        start_date = as_of_date
        end_date = as_of_date
        mode = "as_of"
        end_source = "as_of"
    else:
        if not start_token:
            raise SystemExit("Provide --start-date (or set start_date in config), or use --as-of.")
        start_date = parse_date(start_token)
        if mode == "daily":
            if end_token in {None, "t-1"}:
                end_date = resolve_last_trading_date(
                    rqdatac, pd.Timestamp.now(), market, include_today=False
                )
                end_source = "t-1"
            elif end_token == "today":
                end_date = resolve_last_trading_date(
                    rqdatac, pd.Timestamp.now(), market, include_today=True
                )
                end_source = "today"
            else:
                end_date = parse_date(end_token)
        else:
            if not end_token:
                raise SystemExit("Provide --end-date (or set end_date in config) for backtest mode.")
            if end_token in {"today", "t-1"}:
                raise SystemExit("end-date must be YYYYMMDD in backtest mode; use mode=daily for T-1.")
            end_date = parse_date(end_token)

    if start_date > end_date:
        raise SystemExit("start-date must be <= end-date.")

    top_quantile = float(settings.get("top_quantile"))
    if not 0 < top_quantile < 1:
        raise SystemExit("top-quantile must be between 0 and 1.")

    lookback_days = int(settings.get("lookback_days"))
    min_window_days = int(settings.get("min_window_days"))
    min_turnover = float(settings.get("min_turnover") or 0.0)
    if lookback_days <= 0:
        raise SystemExit("lookback-days must be positive.")
    if min_window_days <= 0:
        raise SystemExit("min-window-days must be positive.")

    date_tag = end_date.strftime("%Y%m%d")
    append_outputs = bool(settings.get("append_date")) and mode in {"daily", "as_of"}
    out_path = format_output_path(settings.get("out"), date_tag, append_outputs)
    latest_out_path = format_output_path(settings.get("latest_out"), date_tag, append_outputs)

    write_meta = bool(settings.get("write_meta"))
    if write_meta:
        if settings.get("meta_out"):
            meta_out_path = format_output_path(settings.get("meta_out"), date_tag, False)
        else:
            meta_out_path = default_meta_path(out_path)
    else:
        meta_out_path = None

    effective = {
        "mode": mode,
        "start_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "as_of_input": as_of_input,
        "end_date_source": end_source,
        "rebalance_frequency": settings.get("rebalance_frequency"),
        "lookback_days": lookback_days,
        "min_window_days": min_window_days,
        "top_quantile": top_quantile,
        "min_turnover": min_turnover,
        "out": str(out_path),
        "latest_out": str(latest_out_path) if latest_out_path else None,
        "append_date": append_outputs,
        "write_meta": write_meta,
        "meta_out": str(meta_out_path) if meta_out_path else None,
    }

    print("Effective settings:")
    print(yaml.safe_dump(effective, sort_keys=False).strip())

    buffer_days = max(lookback_days * 3, 90)
    extended_start = (start_date - pd.Timedelta(days=buffer_days)).strftime("%Y%m%d")
    trade_dates = coerce_trading_dates(
        rqdatac.get_trading_dates(extended_start, end_date.strftime("%Y%m%d"), market=market)
    )
    if not trade_dates:
        raise SystemExit("No trading dates returned for the requested range.")

    trade_dates_in_range = [d for d in trade_dates if start_date <= d <= end_date]
    if not trade_dates_in_range:
        raise SystemExit("No trading dates in the requested range.")

    if mode == "as_of":
        rebalance_dates = [trade_dates_in_range[-1]]
    else:
        rebalance_dates = get_rebalance_dates(trade_dates_in_range, settings.get("rebalance_frequency"))
    if not rebalance_dates:
        raise SystemExit("No rebalance dates computed for the requested range.")

    membership = fetch_southbound_membership(rqdatac, rebalance_dates)
    all_symbols = sorted({sym for symbols in membership.values() for sym in symbols})
    if not all_symbols:
        raise SystemExit("No eligible HK Connect symbols returned.")

    turnover = prepare_turnover_table(
        rqdatac,
        all_symbols,
        trade_dates[0].strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
    )

    trade_index = {date.normalize(): idx for idx, date in enumerate(trade_dates)}
    results = []
    for reb_date in rebalance_dates:
        reb_date = reb_date.normalize()
        idx = trade_index.get(reb_date)
        if idx is None or idx == 0:
            continue
        window_end = idx - 1
        window_start = max(0, window_end - lookback_days + 1)
        window_dates = trade_dates[window_start : window_end + 1]
        window_data = turnover.reindex(window_dates)
        liq = window_data.median(axis=0, skipna=True)
        valid_counts = window_data.notna().sum(axis=0)
        liq = liq[valid_counts >= min(min_window_days, lookback_days)]
        if min_turnover > 0:
            liq = liq[liq >= min_turnover]

        eligible = membership.get(reb_date, set())
        if eligible:
            liq = liq[liq.index.isin(eligible)]
        if liq.empty:
            continue

        threshold = liq.quantile(top_quantile)
        selected = liq[liq >= threshold].sort_values(ascending=False)
        for order_book_id, metric in selected.items():
            ts_code = normalize_hk_symbol(order_book_id)
            results.append(
                {
                    "trade_date": reb_date.strftime("%Y%m%d"),
                    "ts_code": ts_code,
                    "stock_ticker": ts_code,
                    "liq_metric": float(metric),
                    "selected": 1,
                }
            )

    if not results:
        raise SystemExit("No symbols selected; check date range or parameters.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(results)
    output_df.to_csv(out_path, index=False)
    print(f"Wrote universe rows: {len(results)} -> {out_path}")

    if latest_out_path:
        latest_date = max(row["trade_date"] for row in results)
        latest_symbols = sorted({row["ts_code"] for row in results if row["trade_date"] == latest_date})
        latest_out_path.parent.mkdir(parents=True, exist_ok=True)
        latest_out_path.write_text("\n".join(latest_symbols), encoding="utf-8")
        print(f"Wrote latest symbols ({latest_date}) -> {latest_out_path}")

    if write_meta and meta_out_path:
        counts = (
            output_df.groupby("trade_date")["ts_code"].nunique().sort_index()
            if not output_df.empty
            else pd.Series(dtype=int)
        )
        meta = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "as_of_input": as_of_input,
            "as_of_date": end_date.strftime("%Y%m%d") if mode in {"daily", "as_of"} else None,
            "end_date_source": end_source,
            "rebalance_frequency": settings.get("rebalance_frequency"),
            "lookback_days": lookback_days,
            "min_window_days": min_window_days,
            "min_turnover": min_turnover,
            "top_quantile": top_quantile,
            "buffer_days": buffer_days,
            "rebalance_dates": [d.strftime("%Y%m%d") for d in rebalance_dates],
            "rebalance_counts": [
                {"trade_date": str(date), "count": int(count)} for date, count in counts.items()
            ],
            "count_summary": {
                "min": int(counts.min()) if not counts.empty else 0,
                "max": int(counts.max()) if not counts.empty else 0,
                "mean": float(counts.mean()) if not counts.empty else 0.0,
                "median": float(counts.median()) if not counts.empty else 0.0,
            },
            "total_rows": int(len(output_df)),
            "unique_symbols": int(output_df["ts_code"].nunique()),
            "out": str(out_path),
            "latest_out": str(latest_out_path) if latest_out_path else None,
            "meta_out": str(meta_out_path),
            "liquidity_metric": "median(total_turnover)",
            "source": "rqdatac.hk.get_southbound_eligible_secs",
        }
        meta_out_path.parent.mkdir(parents=True, exist_ok=True)
        meta_out_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        print(f"Wrote meta -> {meta_out_path}")


if __name__ == "__main__":
    main()
