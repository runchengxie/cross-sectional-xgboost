"""pipeline.py - Cross-sectional factor mining with XGBoost regression (multi-market).
Usage:
    $ csxgb run
    $ csxgb run --config config/default.yml
    $ csxgb run --config cn
    # provider-specific auth may be required (e.g. TUSHARE_TOKEN/TUSHARE_TOKEN_2)
"""
import argparse
import hashlib
import json
import logging
import sys
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
# Workaround for pandas_ta NaN import issue
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
import pyarrow  # ensures parquet support
from dotenv import load_dotenv
import yaml
from xgboost import XGBRegressor
import warnings

from .config_utils import resolve_pipeline_config
from .data_interface import DataInterface
from .data_providers import normalize_market
from .dataset import DatasetSchema, build_dataset
from .execution import build_execution_model, describe_execution_model, BpsCostModel
from .metrics import (
    daily_ic_series,
    summarize_ic,
    quantile_returns,
    estimate_turnover,
    summarize_active_returns,
    regression_error_metrics,
    hit_rate,
    topk_positive_ratio,
    assign_daily_quantile_bucket,
    bucket_ic_summary,
)
from .transform import apply_cross_sectional_transform
from .split import build_sample_weight, time_series_cv_ic
from .backtest import backtest_topk, summarize_period_returns
from .portfolio import build_positions_by_rebalance
from .rebalance import estimate_rebalance_gap, get_rebalance_dates

warnings.filterwarnings("ignore")

logger = logging.getLogger("csxgb")


def _normalize_date_token(value: object | None, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in {"today", "t", "now"}:
        return "today"
    if lowered in {"t-1", "yesterday"}:
        return "t-1"
    if lowered in {"last_trading_day", "last_completed_trading_day"}:
        return lowered
    return text


def _resolve_last_trading_date(
    as_of: pd.Timestamp,
    market: str,
    *,
    include_today: bool,
) -> Optional[pd.Timestamp]:
    try:
        import rqdatac
    except ImportError:
        return None

    as_of = pd.to_datetime(as_of).normalize()
    lookbacks = [366, 365 * 5]
    for days in lookbacks:
        start = (as_of - pd.Timedelta(days=days)).strftime("%Y%m%d")
        end = as_of.strftime("%Y%m%d")
        try:
            dates = rqdatac.get_trading_dates(start, end, market=market)
        except Exception:
            continue
        if not dates:
            continue
        candidates = [d.normalize() for d in pd.to_datetime(dates)]
        if include_today:
            candidates = [d for d in candidates if d <= as_of]
        else:
            candidates = [d for d in candidates if d < as_of]
        if candidates:
            return max(candidates)
    return None


def _resolve_date_token(
    value: object | None,
    default: str = "today",
    *,
    market: Optional[str] = None,
    provider: Optional[str] = None,
) -> pd.Timestamp:
    token = _normalize_date_token(value, default)
    today = pd.Timestamp.now().normalize()
    if token == "today":
        return today
    if token in {"last_trading_day", "last_completed_trading_day"}:
        include_today = token == "last_trading_day"
        if provider == "rqdata" and market:
            resolved = _resolve_last_trading_date(today, market, include_today=include_today)
            if resolved is not None:
                return resolved
        logger.warning(
            "Token '%s' requested but trading calendar unavailable; falling back to calendar day.",
            token,
        )
        return today if include_today else today - pd.Timedelta(days=1)
    if token == "t-1":
        return today - pd.Timedelta(days=1)
    text = str(token).strip()
    compact = text.replace("-", "")
    if compact.isdigit() and len(compact) == 8:
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"Invalid date token: {value}")
    return pd.Timestamp(parsed).normalize()


def build_benchmark_series(
    benchmark_df: Optional[pd.DataFrame],
    price_col: str,
    period_info: list[dict],
) -> tuple[pd.Series, list[dict]]:
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(dtype=float, name="benchmark_return"), []
    bench_prices = benchmark_df.set_index("trade_date")[price_col]
    bench_returns = []
    bench_index = []
    bench_periods: list[dict] = []
    for info in period_info:
        entry_date = info["entry_date"]
        exit_date = info["exit_date"]
        if entry_date not in bench_prices.index or exit_date not in bench_prices.index:
            continue
        bench_returns.append(bench_prices.loc[exit_date] / bench_prices.loc[entry_date] - 1.0)
        bench_index.append(exit_date)
        bench_periods.append(info)
    if not bench_returns:
        return pd.Series(dtype=float, name="benchmark_return"), []
    return pd.Series(bench_returns, index=bench_index, name="benchmark_return"), bench_periods


def build_walk_forward_windows(
    all_dates: np.ndarray,
    test_size: float,
    n_windows: int,
    step_size: Optional[float],
    gap_days: int,
    anchor_end: bool,
) -> list[dict]:
    n_dates = len(all_dates)
    if n_dates == 0:
        return []
    if test_size <= 0:
        return []
    test_len = int(test_size) if test_size >= 1 else int(n_dates * test_size)
    test_len = max(1, test_len)
    step = step_size
    if step is None:
        step = test_len
    elif 0 < float(step) < 1:
        step = int(n_dates * float(step))
    step = max(1, int(step))

    if anchor_end:
        first_test_start = n_dates - test_len - step * (n_windows - 1)
    else:
        first_test_start = int(n_dates * (1 - test_size))
    windows = []
    for idx in range(n_windows):
        test_start = first_test_start + idx * step
        test_end = test_start + test_len
        if test_start < 0 or test_end > n_dates:
            continue
        train_end = max(0, test_start - gap_days)
        train_dates = all_dates[:train_end]
        test_dates = all_dates[test_start:test_end]
        if len(train_dates) == 0 or len(test_dates) == 0:
            continue
        windows.append(
            {
                "window": idx + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_dates": train_dates,
                "test_dates": test_dates,
            }
        )
    return windows


def _normalize_window_months(value: object | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        items = []
        for entry in value:
            if entry is None:
                continue
            try:
                num = int(entry)
            except (TypeError, ValueError):
                continue
            if num > 0:
                items.append(num)
        return sorted(set(items))
    return default


def _estimate_obs_per_year(series: pd.Series) -> float:
    if series is None or series.empty:
        return np.nan
    if not isinstance(series.index, pd.DatetimeIndex):
        return np.nan
    start = series.index.min()
    end = series.index.max()
    if start is pd.NaT or end is pd.NaT:
        return np.nan
    days = float((end - start).days)
    if days <= 0:
        return np.nan
    return float(series.shape[0] / (days / 365.25))


def _latest_rolling_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, float] | None:
    if frame is None or frame.empty:
        return None
    valid = frame.dropna(subset=columns, how="any")
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return {col: float(last[col]) for col in columns}


def _compute_rolling_ic(
    ic_series: pd.Series, window_months: list[int]
) -> tuple[dict[str, pd.DataFrame], float]:
    results: dict[str, pd.DataFrame] = {}
    if ic_series is None or ic_series.empty:
        return results, np.nan
    obs_per_year = _estimate_obs_per_year(ic_series)
    if not np.isfinite(obs_per_year) or obs_per_year <= 0:
        return results, np.nan
    for months in window_months:
        window_obs = int(round(obs_per_year * months / 12))
        if window_obs < 2:
            continue
        rolling = ic_series.rolling(window_obs, min_periods=window_obs)
        mean = rolling.mean()
        std = rolling.std(ddof=0)
        ir = mean / std
        frame = pd.DataFrame({"ic_mean": mean, "ic_std": std, "ic_ir": ir})
        results[f"{months}m"] = frame
    return results, float(obs_per_year)


def _compute_rolling_sharpe(
    returns: pd.Series, window_months: list[int], periods_per_year: float
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    if returns is None or returns.empty:
        return results
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return results
    for months in window_months:
        window_obs = int(round(periods_per_year * months / 12))
        if window_obs < 2:
            continue
        rolling = returns.rolling(window_obs, min_periods=window_obs)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        sharpe = mean / std * np.sqrt(periods_per_year)
        frame = pd.DataFrame({"mean": mean, "std": std, "sharpe": sharpe})
        results[f"{months}m"] = frame
    return results


def _normalize_bucket_schemes(raw_schemes: object | None) -> list[dict]:
    schemes: list[dict] = []
    if raw_schemes is None:
        return schemes
    if isinstance(raw_schemes, dict):
        raw_items = raw_schemes.get("schemes") or []
    else:
        raw_items = raw_schemes
    if isinstance(raw_items, (str, int, float)):
        raw_items = [raw_items]
    if not isinstance(raw_items, (list, tuple)):
        return schemes
    for item in raw_items:
        if isinstance(item, str):
            col = item.strip()
            if not col:
                continue
            schemes.append({"name": col, "column": col, "type": "category", "n_bins": 0})
            continue
        if not isinstance(item, dict):
            continue
        col = item.get("column") or item.get("col")
        if not col:
            continue
        name = item.get("name") or col
        bucket_type = str(item.get("type", "category")).strip().lower()
        n_bins = item.get("n_bins", item.get("bins", 3))
        try:
            n_bins = int(n_bins) if n_bins is not None else 0
        except (TypeError, ValueError):
            n_bins = 0
        schemes.append(
            {
                "name": str(name),
                "column": str(col),
                "type": bucket_type,
                "n_bins": n_bins,
            }
        )
    return schemes

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging") if isinstance(cfg, dict) else None
    log_cfg = log_cfg if isinstance(log_cfg, dict) else {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    log_file = log_cfg.get("file")
    level = getattr(logging, level_name, logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def config_hash(cfg: dict) -> str:
    dumped = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()[:8]


def _atomic_write(path: Path, write_fn) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def save_series(series: pd.Series, path: Path, value_name: Optional[str] = None) -> None:
    if series is None or series.empty:
        return
    name = value_name or series.name or "value"
    out = series.rename(name).reset_index()

    def _write(tmp_path: Path) -> None:
        out.to_csv(tmp_path, index=False)

    _atomic_write(path, _write)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return

    def _write(tmp_path: Path) -> None:
        frame.to_csv(tmp_path, index=False)

    _atomic_write(path, _write)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return

    def _write(tmp_path: Path) -> None:
        frame.to_parquet(tmp_path)

    _atomic_write(path, _write)


def save_json(payload: dict, path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, default=str)

    _atomic_write(path, _write)


def _coerce_yyyymmdd(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    formatted = parsed.dt.strftime("%Y%m%d")
    return formatted.where(parsed.notna(), text)


def _annotate_positions_window(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "rebalance_date" in out.columns:
        rebalance_compact = _coerce_yyyymmdd(out["rebalance_date"])
        out["rebalance_date"] = rebalance_compact
        out["signal_asof"] = rebalance_compact
    if "entry_date" in out.columns:
        entry_compact = _coerce_yyyymmdd(out["entry_date"])
        out["entry_date"] = entry_compact
        entry_dt = pd.to_datetime(entry_compact, format="%Y%m%d", errors="coerce")
        unique_entries = sorted(entry_dt.dropna().unique())
        next_map = {
            unique_entries[idx]: unique_entries[idx + 1]
            for idx in range(len(unique_entries) - 1)
        }
        next_entry = entry_dt.map(next_map)
        next_entry_str = next_entry.dt.strftime("%Y%m%d").where(next_entry.notna(), "")
        out["next_entry_date"] = next_entry_str
        holding_window = out["entry_date"].astype(str) + " -> " + out["next_entry_date"]
        holding_window = holding_window.where(out["next_entry_date"].astype(str) != "", out["entry_date"])
        out["holding_window"] = holding_window
    return out


def _build_rebalance_diff(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "entry_date" not in frame.columns:
        return pd.DataFrame()
    entry_compact = _coerce_yyyymmdd(frame["entry_date"])
    entry_dt = pd.to_datetime(entry_compact, format="%Y%m%d", errors="coerce")
    unique_entries = sorted(entry_dt.dropna().unique())
    if len(unique_entries) < 2:
        return pd.DataFrame()
    latest_entry = unique_entries[-1]
    prev_entry = unique_entries[-2]
    current = frame[entry_dt == latest_entry].copy()
    previous = frame[entry_dt == prev_entry].copy()

    for df in (current, previous):
        if "side" not in df.columns:
            df["side"] = "long"
        if "weight" not in df.columns:
            df["weight"] = np.nan
        if "signal" not in df.columns:
            df["signal"] = np.nan
        if "rank" not in df.columns:
            df["rank"] = np.nan

    current = current[["ts_code", "side", "weight", "signal", "rank"]].rename(
        columns={
            "weight": "weight",
            "signal": "signal",
            "rank": "rank",
        }
    )
    previous = previous[["ts_code", "side", "weight", "signal", "rank"]].rename(
        columns={
            "weight": "weight_prev",
            "signal": "signal_prev",
            "rank": "rank_prev",
        }
    )

    merged = current.merge(
        previous, on=["ts_code", "side"], how="outer", indicator=True
    )
    merged["weight"] = merged["weight"].fillna(0.0)
    merged["weight_prev"] = merged["weight_prev"].fillna(0.0)
    merged["weight_delta"] = merged["weight"] - merged["weight_prev"]
    merged["change"] = merged["_merge"].astype(str).map(
        {"left_only": "added", "right_only": "removed", "both": "changed"}
    )
    merged.loc[
        (merged["_merge"] == "both") & (merged["weight_delta"].abs() < 1e-12),
        "change",
    ] = "unchanged"
    merged = merged[merged["change"] != "unchanged"].copy()
    merged["entry_date"] = latest_entry.strftime("%Y%m%d")
    merged["entry_date_prev"] = prev_entry.strftime("%Y%m%d")
    merged.drop(columns=["_merge"], inplace=True)
    merged.sort_values(["change", "side", "ts_code"], inplace=True)
    return merged


def normalize_symbol_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_symbols_file(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Symbols file not found: {path}")
    symbols = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            symbols.append(text)
    return symbols


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def normalize_universe_symbol(symbol: str, market: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return text
    if normalize_market(market) == "hk":
        upper = text.upper()
        if upper.endswith(".XHKG"):
            upper = upper[:-5]
        if upper.endswith(".HK"):
            upper = upper[:-3]
        if upper.isdigit():
            upper = upper.zfill(5)
        return f"{upper}.HK"
    return text


def load_universe_by_date(path: Path, market: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Universe-by-date file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"Universe-by-date file is empty: {path}")
    columns = {col.lower(): col for col in df.columns}
    date_col = columns.get("trade_date") or columns.get("date") or columns.get("rebalance_date")
    symbol_col = columns.get("ts_code") or columns.get("symbol") or columns.get("order_book_id")
    if not date_col or not symbol_col:
        sys.exit("Universe-by-date file must include date + symbol columns.")

    df = df.rename(columns={date_col: "trade_date", symbol_col: "ts_code"})
    selected_col = (
        columns.get("selected")
        or columns.get("selected_bool")
        or columns.get("selected_flag")
        or columns.get("is_selected")
    )
    if selected_col and selected_col in df.columns:
        df = df[df[selected_col].map(coerce_bool)].copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"].notna()].copy()
    df["trade_date"] = df["trade_date"].dt.normalize()
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df["ts_code"] = df["ts_code"].apply(lambda s: normalize_universe_symbol(s, market))
    df = df[df["ts_code"] != ""].copy()
    df = df.drop_duplicates(subset=["trade_date", "ts_code"])
    return df[["trade_date", "ts_code"]].copy()


def apply_universe_by_date(data: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return data
    rebalance_dates = np.array(sorted(universe["trade_date"].unique()))
    if rebalance_dates.size == 0:
        return data
    trade_dates = np.array(sorted(data["trade_date"].unique()))
    if trade_dates.size == 0:
        return data
    idx = np.searchsorted(rebalance_dates, trade_dates, side="right") - 1
    valid_mask = idx >= 0
    if not np.any(valid_mask):
        return data.iloc[0:0].copy()
    date_map = pd.DataFrame(
        {
            "trade_date": trade_dates[valid_mask],
            "rebalance_date": rebalance_dates[idx[valid_mask]],
        }
    )
    universe_map = universe.rename(columns={"trade_date": "rebalance_date"})
    data = data.merge(date_map, on="trade_date", how="inner")
    data = data.merge(
        universe_map[["rebalance_date", "ts_code"]],
        on=["rebalance_date", "ts_code"],
        how="inner",
    )
    return data.drop(columns=["rebalance_date"])


def parse_feature_windows(features: list[str], prefix: str, suffix: str = "") -> list[int]:
    windows = set()
    for feat in features:
        if not feat.startswith(prefix):
            continue
        if suffix and not feat.endswith(suffix):
            continue
        end = len(feat) - len(suffix) if suffix else len(feat)
        value = feat[len(prefix):end]
        if value.isdigit():
            windows.add(int(value))
    return sorted(windows)



def run(config_ref: str | Path | None = None) -> None:
    resolved = resolve_pipeline_config(config_ref)
    config = resolved.data
    config_label = resolved.label
    config_path = resolved.path
    config_source = resolved.source
    setup_logging(config)

    data_cfg = config.get("data", {})
    MARKET = normalize_market(config.get("market") or data_cfg.get("market"))
    universe_cfg = config.get("universe", {})
    label_cfg = config.get("label", {})
    features_cfg = config.get("features", {})
    fundamentals_cfg = config.get("fundamentals", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("eval", {})
    backtest_cfg = config.get("backtest", {})
    live_cfg = config.get("live", {})
    if not isinstance(live_cfg, dict):
        live_cfg = {}

    load_dotenv()
    CACHE_DIR = Path(data_cfg.get("cache_dir", "cache"))
    data_interface = DataInterface(MARKET, data_cfg, cache_dir=CACHE_DIR, logger=logger)
    provider = data_interface.provider

    DEFAULT_SYMBOLS_BY_MARKET = {
        "cn": [
            "600519.SH",
            "601318.SH",
            "600036.SH",
            "000858.SZ",
            "000333.SZ",
            "600887.SH",
            "600276.SH",
            "601888.SH",
            "000001.SZ",
            "002415.SZ",
        ],
        "hk": [
            "00700.HK",
            "00005.HK",
            "00941.HK",
            "00001.HK",
            "00388.HK",
        ],
        "us": [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
        ],
    }

    DEFAULT_SYMBOLS = DEFAULT_SYMBOLS_BY_MARKET.get(MARKET, DEFAULT_SYMBOLS_BY_MARKET["cn"])

    UNIVERSE_MODE = str(universe_cfg.get("mode", "auto")).strip().lower()
    if UNIVERSE_MODE not in {"auto", "pit", "static"}:
        sys.exit("universe.mode must be one of: auto, pit, static.")
    REQUIRE_BY_DATE = bool(universe_cfg.get("require_by_date", False))

    symbols = normalize_symbol_list(universe_cfg.get("symbols"))
    symbols_file = universe_cfg.get("symbols_file")
    by_date_file = universe_cfg.get("by_date_file")
    universe_by_date = None
    universe_mode_effective = UNIVERSE_MODE

    if not symbols and symbols_file:
        symbols = load_symbols_file(Path(symbols_file))

    if by_date_file:
        universe_by_date = load_universe_by_date(Path(by_date_file), MARKET)
        symbols_from_universe = sorted(universe_by_date["ts_code"].unique().tolist())
        if symbols:
            symbols = sorted(set(symbols) | set(symbols_from_universe))
        else:
            symbols = symbols_from_universe
        universe_mode_effective = "pit"
        if UNIVERSE_MODE == "static":
            logger.warning("universe.mode=static but by_date_file provided; using PIT universe.")
    else:
        if REQUIRE_BY_DATE or UNIVERSE_MODE == "pit":
            sys.exit("universe.by_date_file is required when universe.mode=pit or require_by_date=true.")
        universe_mode_effective = "static"
        if UNIVERSE_MODE == "auto":
            logger.warning(
                "Universe-by-date not provided; using static symbols (survivorship bias). "
                "Set universe.mode=static to acknowledge or provide by_date_file for PIT."
            )

    if not symbols:
        symbols = DEFAULT_SYMBOLS

    if not symbols:
        sys.exit("No symbols configured.")

    end_date_cfg = data_cfg.get("end_date", "today")
    end_date = _resolve_date_token(end_date_cfg, default="today", market=MARKET, provider=provider)

    start_date_cfg = data_cfg.get("start_date")
    if start_date_cfg:
        start_date = datetime.strptime(str(start_date_cfg), "%Y%m%d")
    else:
        start_years = float(data_cfg.get("start_years", 5))
        start_date = end_date - timedelta(days=int(start_years * 365))

    START_DATE = start_date.strftime("%Y%m%d")
    END_DATE = end_date.strftime("%Y%m%d")

    PRICE_COL = data_cfg.get("price_col", "close")

    LABEL_HORIZON_DAYS = int(label_cfg.get("horizon_days", 5))
    LABEL_SHIFT_DAYS = int(label_cfg.get("shift_days", 0))
    LABEL_HORIZON_MODE = str(label_cfg.get("horizon_mode", "fixed")).strip().lower()
    if LABEL_HORIZON_MODE not in {"fixed", "next_rebalance"}:
        sys.exit("label.horizon_mode must be one of: fixed, next_rebalance.")
    LABEL_REBALANCE_FREQUENCY = label_cfg.get("rebalance_frequency", eval_cfg.get("rebalance_frequency", "M"))
    TARGET = label_cfg.get("target_col", "future_return")
    WINSORIZE_PCT = label_cfg.get("winsorize_pct")
    if WINSORIZE_PCT is not None:
        WINSORIZE_PCT = float(WINSORIZE_PCT)
        if not 0 < WINSORIZE_PCT < 0.5:
            sys.exit("winsorize_pct must be between 0 and 0.5.")

    TEST_SIZE = float(eval_cfg.get("test_size", 0.2))
    N_SPLITS = int(eval_cfg.get("n_splits", 5))
    N_QUANTILES = int(eval_cfg.get("n_quantiles", 5))
    TOP_K = int(eval_cfg.get("top_k", 20))
    REBALANCE_FREQUENCY = eval_cfg.get("rebalance_frequency", "W")
    TRANSACTION_COST_BPS = float(eval_cfg.get("transaction_cost_bps", 10))
    EVAL_BUFFER_EXIT = int(eval_cfg.get("buffer_exit", backtest_cfg.get("buffer_exit", 0) if isinstance(backtest_cfg, dict) else 0))
    EVAL_BUFFER_ENTRY = int(eval_cfg.get("buffer_entry", backtest_cfg.get("buffer_entry", 0) if isinstance(backtest_cfg, dict) else 0))
    SIGNAL_DIRECTION_MODE = str(eval_cfg.get("signal_direction_mode", "fixed")).strip().lower()
    if SIGNAL_DIRECTION_MODE not in {"fixed", "train_ic", "cv_ic"}:
        sys.exit("eval.signal_direction_mode must be one of: fixed, train_ic, cv_ic.")
    SIGNAL_DIRECTION_RAW = eval_cfg.get("signal_direction", 1.0)
    SIGNAL_DIRECTION = float(SIGNAL_DIRECTION_RAW) if SIGNAL_DIRECTION_RAW is not None else 1.0
    if SIGNAL_DIRECTION == 0:
        sys.exit("eval.signal_direction cannot be 0.")
    MIN_ABS_IC_TO_FLIP_RAW = eval_cfg.get("min_abs_ic_to_flip", 0.0)
    MIN_ABS_IC_TO_FLIP = float(MIN_ABS_IC_TO_FLIP_RAW) if MIN_ABS_IC_TO_FLIP_RAW is not None else 0.0
    if MIN_ABS_IC_TO_FLIP < 0:
        sys.exit("eval.min_abs_ic_to_flip must be >= 0.")
    EMBARGO_DAYS_RAW = eval_cfg.get("embargo_days")
    EMBARGO_DAYS_CFG = int(EMBARGO_DAYS_RAW) if EMBARGO_DAYS_RAW is not None else 0
    PURGE_DAYS_RAW = eval_cfg.get("purge_days")
    PURGE_DAYS_CFG = int(PURGE_DAYS_RAW) if PURGE_DAYS_RAW is not None else None
    PURGE_STEPS = None
    EMBARGO_STEPS = None
    EFFECTIVE_GAP_STEPS = None
    REPORT_TRAIN_IC = bool(eval_cfg.get("report_train_ic", True))
    SAMPLE_ON_REBALANCE_DATES = bool(eval_cfg.get("sample_on_rebalance_dates", False))
    rolling_cfg = eval_cfg.get("rolling") if isinstance(eval_cfg, dict) else None
    if isinstance(rolling_cfg, dict):
        rolling_enabled = bool(rolling_cfg.get("enabled", True))
        if rolling_enabled:
            ROLLING_WINDOWS_MONTHS = _normalize_window_months(
                rolling_cfg.get("windows_months"), [6, 12]
            )
        else:
            ROLLING_WINDOWS_MONTHS = []
    else:
        ROLLING_WINDOWS_MONTHS = _normalize_window_months(rolling_cfg, [6, 12])

    bucket_ic_cfg = eval_cfg.get("bucket_ic") if isinstance(eval_cfg, dict) else None
    BUCKET_IC_ENABLED = False
    BUCKET_IC_METHOD = "spearman"
    BUCKET_IC_MIN_COUNT = 0
    BUCKET_IC_SCHEMES = []
    if isinstance(bucket_ic_cfg, dict):
        BUCKET_IC_ENABLED = bool(bucket_ic_cfg.get("enabled", False))
        BUCKET_IC_METHOD = str(bucket_ic_cfg.get("method", "spearman")).strip().lower()
        BUCKET_IC_MIN_COUNT = int(bucket_ic_cfg.get("min_count", 0) or 0)
        BUCKET_IC_SCHEMES = _normalize_bucket_schemes(bucket_ic_cfg.get("schemes"))
    elif bucket_ic_cfg is not None:
        BUCKET_IC_ENABLED = bool(bucket_ic_cfg)
    if BUCKET_IC_METHOD not in {"spearman", "pearson"}:
        sys.exit("eval.bucket_ic.method must be one of: spearman, pearson.")
    if BUCKET_IC_ENABLED and not BUCKET_IC_SCHEMES:
        logger.warning("eval.bucket_ic.enabled=true but no schemes configured.")
    perm_cfg = eval_cfg.get("permutation_test") or {}
    if isinstance(perm_cfg, dict):
        PERM_TEST_ENABLED = bool(perm_cfg.get("enabled", False))
        PERM_TEST_RUNS = int(perm_cfg.get("n_runs", 1))
        PERM_TEST_SEED = perm_cfg.get("seed")
    else:
        PERM_TEST_ENABLED = bool(perm_cfg)
        PERM_TEST_RUNS = 1
        PERM_TEST_SEED = None
    if PERM_TEST_SEED is not None:
        PERM_TEST_SEED = int(PERM_TEST_SEED)
    if PERM_TEST_RUNS < 1:
        PERM_TEST_ENABLED = False

    wf_cfg = eval_cfg.get("walk_forward") or {}
    if isinstance(wf_cfg, dict):
        WF_ENABLED = bool(wf_cfg.get("enabled", False))
        WF_N_WINDOWS = int(wf_cfg.get("n_windows", 3))
        WF_TEST_SIZE = wf_cfg.get("test_size", TEST_SIZE)
        WF_STEP_SIZE = wf_cfg.get("step_size")
        WF_ANCHOR_END = bool(wf_cfg.get("anchor_end", True))
        WF_BACKTEST_ENABLED = bool(wf_cfg.get("backtest_enabled", backtest_cfg.get("enabled", True)))
        wf_perm_cfg = wf_cfg.get("permutation_test")
        if isinstance(wf_perm_cfg, dict):
            WF_PERM_TEST_ENABLED = bool(wf_perm_cfg.get("enabled", False))
            WF_PERM_TEST_RUNS = int(wf_perm_cfg.get("n_runs", PERM_TEST_RUNS))
            WF_PERM_TEST_SEED = wf_perm_cfg.get("seed", PERM_TEST_SEED)
        elif wf_perm_cfg is None:
            WF_PERM_TEST_ENABLED = False
            WF_PERM_TEST_RUNS = PERM_TEST_RUNS
            WF_PERM_TEST_SEED = PERM_TEST_SEED
        else:
            WF_PERM_TEST_ENABLED = bool(wf_perm_cfg)
            WF_PERM_TEST_RUNS = PERM_TEST_RUNS
            WF_PERM_TEST_SEED = PERM_TEST_SEED
    else:
        WF_ENABLED = bool(wf_cfg)
        WF_N_WINDOWS = 3
        WF_TEST_SIZE = TEST_SIZE
        WF_STEP_SIZE = None
        WF_ANCHOR_END = True
        WF_BACKTEST_ENABLED = bool(backtest_cfg.get("enabled", True))
        WF_PERM_TEST_ENABLED = False
        WF_PERM_TEST_RUNS = PERM_TEST_RUNS
        WF_PERM_TEST_SEED = PERM_TEST_SEED
    if WF_PERM_TEST_SEED is not None:
        WF_PERM_TEST_SEED = int(WF_PERM_TEST_SEED)
    if WF_PERM_TEST_RUNS < 1:
        WF_PERM_TEST_ENABLED = False

    final_oos_cfg = eval_cfg.get("final_oos")
    FINAL_OOS_SIZE_RAW = None
    if isinstance(final_oos_cfg, dict):
        FINAL_OOS_SIZE_RAW = final_oos_cfg.get("size")
        FINAL_OOS_ENABLED = bool(final_oos_cfg.get("enabled", False) or FINAL_OOS_SIZE_RAW)
    elif final_oos_cfg is None:
        FINAL_OOS_ENABLED = False
    else:
        FINAL_OOS_SIZE_RAW = final_oos_cfg
        FINAL_OOS_ENABLED = bool(final_oos_cfg)

    SAVE_ARTIFACTS = bool(eval_cfg.get("save_artifacts", True))
    SAVE_DATASET = bool(eval_cfg.get("save_dataset", False))
    OUTPUT_DIR = eval_cfg.get("output_dir", "out/runs")
    RUN_NAME = eval_cfg.get("run_name")
    if SAVE_DATASET and not SAVE_ARTIFACTS:
        raise SystemExit("eval.save_dataset=true requires eval.save_artifacts=true.")

    MIN_SYMBOLS_PER_DATE = int(universe_cfg.get("min_symbols_per_date", N_QUANTILES))
    MIN_LISTED_DAYS = int(universe_cfg.get("min_listed_days", 0))
    MIN_TURNOVER = float(universe_cfg.get("min_turnover", 0))
    DROP_ST = bool(universe_cfg.get("drop_st", False))
    DROP_SUSPENDED = bool(universe_cfg.get("drop_suspended", True))
    SUSPENDED_POLICY = str(universe_cfg.get("suspended_policy", "mark")).strip().lower()
    if SUSPENDED_POLICY not in {"mark", "filter"}:
        sys.exit("universe.suspended_policy must be one of: mark, filter.")
    if MIN_SYMBOLS_PER_DATE < N_QUANTILES:
        MIN_SYMBOLS_PER_DATE = N_QUANTILES

    fundamentals_cfg = fundamentals_cfg if isinstance(fundamentals_cfg, dict) else {}
    FUNDAMENTALS_ENABLED = bool(fundamentals_cfg.get("enabled", False))
    FUNDAMENTALS_SOURCE = str(fundamentals_cfg.get("source", "provider")).strip().lower()
    if FUNDAMENTALS_SOURCE not in {"provider", "file"}:
        sys.exit("fundamentals.source must be one of: provider, file.")
    FUNDAMENTALS_FILE = fundamentals_cfg.get("file")
    FUNDAMENTALS_FEATURES = normalize_symbol_list(fundamentals_cfg.get("features"))
    FUNDAMENTALS_AUTO_ADD = bool(fundamentals_cfg.get("auto_add_features", True))
    FUNDAMENTALS_ALLOW_MISSING = bool(fundamentals_cfg.get("allow_missing_features", False))
    FUNDAMENTALS_FFILL = bool(fundamentals_cfg.get("ffill", True))
    FUNDAMENTALS_FFILL_LIMIT = fundamentals_cfg.get("ffill_limit")
    if FUNDAMENTALS_FFILL_LIMIT is not None:
        FUNDAMENTALS_FFILL_LIMIT = int(FUNDAMENTALS_FFILL_LIMIT)
    FUNDAMENTALS_LOG_MCAP = bool(fundamentals_cfg.get("log_market_cap", False))
    FUNDAMENTALS_MCAP_COL = str(fundamentals_cfg.get("market_cap_col", "market_cap")).strip()
    FUNDAMENTALS_LOG_MCAP_COL = str(fundamentals_cfg.get("log_market_cap_col", "log_mcap")).strip()
    FUNDAMENTALS_REQUIRED = bool(fundamentals_cfg.get("required", False))

    feature_list = features_cfg.get("list") or []
    FEATURES = normalize_symbol_list(feature_list) if feature_list else [
        "sma_20",
        "sma_5_diff",
        "sma_10_diff",
        "sma_20_diff",
        "rsi_14",
        "macd_hist",
        "volume_sma5_ratio",
        "vol",
    ]
    if FUNDAMENTALS_ENABLED and FUNDAMENTALS_AUTO_ADD and FUNDAMENTALS_FEATURES:
        FEATURES = list(dict.fromkeys(FEATURES + FUNDAMENTALS_FEATURES))
    feature_params = features_cfg.get("params", {})
    cs_cfg = features_cfg.get("cross_sectional") or {}
    CS_METHOD = str(cs_cfg.get("method", "none")).strip().lower() if isinstance(cs_cfg, dict) else "none"
    CS_WINSORIZE_PCT = cs_cfg.get("winsorize_pct") if isinstance(cs_cfg, dict) else None
    if CS_WINSORIZE_PCT is not None:
        CS_WINSORIZE_PCT = float(CS_WINSORIZE_PCT)
        if not 0 < CS_WINSORIZE_PCT < 0.5:
            sys.exit("features.cross_sectional.winsorize_pct must be between 0 and 0.5.")
    if CS_METHOD not in {"none", "zscore", "rank"}:
        sys.exit("features.cross_sectional.method must be one of: none, zscore, rank.")

    XGB_PARAMS = model_cfg.get("params") or {}
    if not XGB_PARAMS:
        XGB_PARAMS = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
        )
    SAMPLE_WEIGHT_MODE = str(model_cfg.get("sample_weight_mode", "none")).strip().lower()
    if SAMPLE_WEIGHT_MODE in {"", "none", "null"}:
        SAMPLE_WEIGHT_MODE = "none"
    if SAMPLE_WEIGHT_MODE in {"date"}:
        SAMPLE_WEIGHT_MODE = "date_equal"
    if SAMPLE_WEIGHT_MODE not in {"none", "date_equal"}:
        sys.exit("model.sample_weight_mode must be one of: none, date_equal.")

    BACKTEST_ENABLED = bool(backtest_cfg.get("enabled", True))
    BACKTEST_TOP_K = int(backtest_cfg.get("top_k", TOP_K))
    BACKTEST_REBALANCE_FREQUENCY = backtest_cfg.get("rebalance_frequency", REBALANCE_FREQUENCY)
    BACKTEST_COST_BPS = float(backtest_cfg.get("transaction_cost_bps", TRANSACTION_COST_BPS))
    BACKTEST_TRADING_DAYS_PER_YEAR = int(backtest_cfg.get("trading_days_per_year", 252))
    BACKTEST_BENCHMARK = backtest_cfg.get("benchmark_symbol")
    BACKTEST_LONG_ONLY = bool(backtest_cfg.get("long_only", True))
    BACKTEST_BUFFER_EXIT = int(backtest_cfg.get("buffer_exit", 0))
    BACKTEST_BUFFER_ENTRY = int(backtest_cfg.get("buffer_entry", 0))
    BACKTEST_SIGNAL_DIRECTION_RAW = backtest_cfg.get("signal_direction")
    if BACKTEST_SIGNAL_DIRECTION_RAW is not None:
        BACKTEST_SIGNAL_DIRECTION_RAW = float(BACKTEST_SIGNAL_DIRECTION_RAW)
        if BACKTEST_SIGNAL_DIRECTION_RAW == 0:
            sys.exit("backtest.signal_direction cannot be 0.")
    BACKTEST_SHORT_K = backtest_cfg.get("short_k")
    if BACKTEST_SHORT_K is not None:
        BACKTEST_SHORT_K = int(BACKTEST_SHORT_K)
    BACKTEST_EXIT_MODE = str(backtest_cfg.get("exit_mode", "rebalance")).strip().lower()
    if BACKTEST_EXIT_MODE not in {"rebalance", "label_horizon"}:
        sys.exit("backtest.exit_mode must be one of: rebalance, label_horizon.")
    BACKTEST_EXIT_HORIZON_DAYS = backtest_cfg.get("exit_horizon_days")
    BACKTEST_EXIT_PRICE_POLICY = str(backtest_cfg.get("exit_price_policy", "strict")).strip().lower()
    if BACKTEST_EXIT_PRICE_POLICY not in {"strict", "ffill", "delay"}:
        sys.exit("backtest.exit_price_policy must be one of: strict, ffill, delay.")
    BACKTEST_EXIT_FALLBACK_POLICY = str(
        backtest_cfg.get("exit_fallback_policy", "ffill")
    ).strip().lower()
    if BACKTEST_EXIT_FALLBACK_POLICY not in {"ffill", "none"}:
        sys.exit("backtest.exit_fallback_policy must be one of: ffill, none.")
    execution_cfg = backtest_cfg.get("execution") if isinstance(backtest_cfg, dict) else None
    execution_model = build_execution_model(
        execution_cfg,
        default_cost_bps=BACKTEST_COST_BPS,
        default_exit_price_policy=BACKTEST_EXIT_PRICE_POLICY,
        default_exit_fallback_policy=BACKTEST_EXIT_FALLBACK_POLICY,
    )
    BACKTEST_EXIT_PRICE_POLICY = execution_model.exit_policy.price_policy
    BACKTEST_EXIT_FALLBACK_POLICY = execution_model.exit_policy.fallback_policy
    BACKTEST_COST_BPS_EFFECTIVE = BACKTEST_COST_BPS
    BACKTEST_COST_BPS_REPORT = None
    if isinstance(execution_model.cost_model, BpsCostModel):
        BACKTEST_COST_BPS_EFFECTIVE = float(execution_model.cost_model.bps)
        BACKTEST_COST_BPS_REPORT = BACKTEST_COST_BPS_EFFECTIVE
    BACKTEST_TRADABLE_COL = backtest_cfg.get("tradable_col", "is_tradable")
    if BACKTEST_TRADABLE_COL is not None:
        BACKTEST_TRADABLE_COL = str(BACKTEST_TRADABLE_COL).strip() or None
    if BACKTEST_EXIT_MODE == "label_horizon":
        if BACKTEST_EXIT_HORIZON_DAYS is None:
            BACKTEST_EXIT_HORIZON_DAYS = LABEL_HORIZON_DAYS
        BACKTEST_EXIT_HORIZON_DAYS = int(BACKTEST_EXIT_HORIZON_DAYS)

    LIVE_ENABLED = bool(live_cfg.get("enabled", False))
    LIVE_AS_OF = live_cfg.get("as_of", "t-1")
    LIVE_TRAIN_MODE = str(live_cfg.get("train_mode", "full")).strip().lower()
    if LIVE_TRAIN_MODE not in {"full", "train"}:
        sys.exit("live.train_mode must be one of: full, train.")
    if LIVE_ENABLED and not SAVE_ARTIFACTS:
        raise SystemExit(
            "live.enabled=true requires eval.save_artifacts=true to persist holdings."
        )

    run_name = str(RUN_NAME or config_label)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = config_hash(config)
    run_dir = Path(OUTPUT_DIR) / f"{run_name}_{run_stamp}_{run_hash}"
    if SAVE_ARTIFACTS:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Artifacts will be saved to %s", run_dir)
    # -----------------------------------------------------------------------------
    # 2. Data download
    # -----------------------------------------------------------------------------
    benchmark_symbol = str(BACKTEST_BENCHMARK).strip() if BACKTEST_BENCHMARK else None
    symbols_for_data = symbols[:]
    if benchmark_symbol and benchmark_symbol not in symbols_for_data:
        symbols_for_data.append(benchmark_symbol)

    frames = []
    def fetch_daily_with_retry(symbol: str) -> pd.DataFrame:
        return data_interface.fetch_daily(symbol, START_DATE, END_DATE)

    for symbol in symbols_for_data:
        logger.info("Fetching daily data for %s (%s) ...", symbol, MARKET)
        try:
            data = fetch_daily_with_retry(symbol)
        except Exception as exc:
            logger.warning("Skipping %s after retries (%s).", symbol, exc)
            data = pd.DataFrame()
        if not data.empty:
            frames.append(data)

    if not frames:
        sys.exit("No data returned - check symbols and date range.")

    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df.sort_values(["ts_code", "trade_date"], inplace=True)

    benchmark_df = None
    if benchmark_symbol:
        if benchmark_symbol in symbols:
            logger.info("Benchmark symbol %s removed from modeling universe.", benchmark_symbol)
        benchmark_df = df[df["ts_code"] == benchmark_symbol].copy()
        df = df[df["ts_code"] != benchmark_symbol].copy()

    basic_df = None
    if DROP_ST or MIN_LISTED_DAYS > 0:
        try:
            if MARKET != "cn" and DROP_ST:
                logger.info("drop_st is CN-specific; attempting basic data for market '%s'.", MARKET)
            basic_df = data_interface.load_basic(symbols_for_data)
        except Exception as exc:
            logger.warning("Basic data load failed (%s); skipping ST/listed filters.", exc)
            basic_df = None

    if DROP_ST and basic_df is not None and "name" in basic_df.columns:
        st_codes = basic_df[
            basic_df["name"].str.contains("ST", case=False, na=False)
        ]["ts_code"]
        df = df[~df["ts_code"].isin(st_codes)].copy()

    if MIN_LISTED_DAYS > 0 and basic_df is not None and "list_date" in basic_df.columns:
        list_dates = basic_df.copy()
        list_dates["list_date"] = pd.to_datetime(list_dates["list_date"], format="%Y%m%d", errors="coerce")
        list_date_map = list_dates.set_index("ts_code")["list_date"].to_dict()
        df["list_date"] = df["ts_code"].map(list_date_map)
        df = df[df["list_date"].notna()].copy()
        df = df[df["trade_date"] >= df["list_date"] + pd.Timedelta(days=MIN_LISTED_DAYS)].copy()

    df["is_tradable"] = True
    if DROP_SUSPENDED:
        if "amount" in df.columns:
            tradable_mask = (df["vol"] > 0) & (df["amount"] > 0)
        else:
            tradable_mask = df["vol"] > 0
        tradable_mask = tradable_mask.fillna(False)
        df["is_tradable"] = tradable_mask
        if SUSPENDED_POLICY == "filter":
            df = df[df["is_tradable"]].copy()

    if MIN_TURNOVER > 0 and "amount" in df.columns:
        df = df[df["amount"] >= MIN_TURNOVER].copy()

    fundamentals_cols: list[str] = []
    if FUNDAMENTALS_ENABLED:
        if FUNDAMENTALS_SOURCE == "provider" and provider != "tushare":
            message = "Fundamentals provider mode currently supports only Tushare; use source=file instead."
            if FUNDAMENTALS_REQUIRED:
                sys.exit(message)
            logger.warning("%s Fundamentals disabled.", message)
            FUNDAMENTALS_ENABLED = False
        if FUNDAMENTALS_SOURCE == "provider":
            endpoint_name = fundamentals_cfg.get("endpoint") or data_cfg.get("fundamentals_endpoint")
            if not endpoint_name:
                message = "fundamentals.endpoint is required for provider mode."
                if FUNDAMENTALS_REQUIRED:
                    sys.exit(message)
                logger.warning("%s Fundamentals disabled.", message)
                FUNDAMENTALS_ENABLED = False
        if FUNDAMENTALS_SOURCE == "file" and not FUNDAMENTALS_FILE:
            message = "fundamentals.file is required when fundamentals.source=file."
            if FUNDAMENTALS_REQUIRED:
                sys.exit(message)
            logger.warning("%s Fundamentals disabled.", message)
            FUNDAMENTALS_ENABLED = False

    if not FUNDAMENTALS_ENABLED and FUNDAMENTALS_AUTO_ADD and FUNDAMENTALS_FEATURES:
        FEATURES = [feat for feat in FEATURES if feat not in FUNDAMENTALS_FEATURES]

    if FUNDAMENTALS_ENABLED:
        fundamentals_frames = []
        if FUNDAMENTALS_SOURCE == "file":
            fund_path = Path(FUNDAMENTALS_FILE)
            if not fund_path.exists():
                message = f"Fundamentals file not found: {fund_path}"
                if FUNDAMENTALS_REQUIRED:
                    sys.exit(message)
                logger.warning("%s Fundamentals disabled.", message)
                FUNDAMENTALS_ENABLED = False
            else:
                if fund_path.suffix.lower() in {".parquet", ".pq"}:
                    fundamentals_frames.append(pd.read_parquet(fund_path))
                else:
                    fundamentals_frames.append(pd.read_csv(fund_path))
        else:
            fund_cache_dir = Path(
                fundamentals_cfg.get("cache_dir", data_cfg.get("cache_dir", "cache"))
            )
            fund_cache_dir.mkdir(exist_ok=True)

            def fetch_fundamentals_with_retry(symbol: str) -> pd.DataFrame:
                return data_interface.fetch_fundamentals(
                    symbol,
                    START_DATE,
                    END_DATE,
                    fundamentals_cfg,
                    cache_dir=fund_cache_dir,
                )

            for symbol in symbols_for_data:
                logger.info("Fetching fundamentals for %s (%s) ...", symbol, MARKET)
                try:
                    fdata = fetch_fundamentals_with_retry(symbol)
                except Exception as exc:
                    logger.warning("Skipping fundamentals for %s after retries (%s).", symbol, exc)
                    fdata = pd.DataFrame()
                if fdata is not None and not fdata.empty:
                    fundamentals_frames.append(fdata)

        if FUNDAMENTALS_ENABLED and fundamentals_frames:
            fund_df = pd.concat(fundamentals_frames, ignore_index=True)
            column_map = fundamentals_cfg.get("column_map") or {}
            if column_map:
                rename_map = {
                    source: standard
                    for standard, source in column_map.items()
                    if source in fund_df.columns and standard != source
                }
                if rename_map:
                    fund_df = fund_df.rename(columns=rename_map)
            if "trade_date" not in fund_df.columns:
                if "date" in fund_df.columns:
                    fund_df = fund_df.rename(columns={"date": "trade_date"})
            if "ts_code" not in fund_df.columns:
                if "symbol" in fund_df.columns:
                    fund_df = fund_df.rename(columns={"symbol": "ts_code"})
            if "trade_date" not in fund_df.columns or "ts_code" not in fund_df.columns:
                sys.exit("Fundamentals data must include trade_date and ts_code columns.")
            fund_df["trade_date"] = pd.to_datetime(fund_df["trade_date"], errors="coerce")
            fund_df = fund_df[fund_df["trade_date"].notna()].copy()
            fund_df["trade_date"] = fund_df["trade_date"].dt.normalize()
            fund_df["ts_code"] = fund_df["ts_code"].astype(str).str.strip()
            fund_df = fund_df.drop_duplicates(subset=["trade_date", "ts_code"])
            fundamentals_cols = [
                col for col in fund_df.columns if col not in {"trade_date", "ts_code"}
            ]
            df = df.merge(fund_df, on=["trade_date", "ts_code"], how="left")
            if FUNDAMENTALS_FFILL and fundamentals_cols:
                df.sort_values(["ts_code", "trade_date"], inplace=True)
                df[fundamentals_cols] = df.groupby("ts_code")[fundamentals_cols].ffill(
                    limit=FUNDAMENTALS_FFILL_LIMIT
                )
            if FUNDAMENTALS_LOG_MCAP and FUNDAMENTALS_MCAP_COL in df.columns:
                df[FUNDAMENTALS_LOG_MCAP_COL] = np.where(
                    df[FUNDAMENTALS_MCAP_COL] > 0,
                    np.log(df[FUNDAMENTALS_MCAP_COL]),
                    np.nan,
                )
                if FUNDAMENTALS_AUTO_ADD and FUNDAMENTALS_LOG_MCAP_COL not in FEATURES:
                    FEATURES = list(dict.fromkeys(FEATURES + [FUNDAMENTALS_LOG_MCAP_COL]))
            logger.info(
                "Merged fundamentals: %s rows, %s columns.",
                len(fund_df),
                len(fundamentals_cols),
            )
        elif FUNDAMENTALS_ENABLED:
            logger.warning("Fundamentals enabled but no data was loaded.")

    label_next_rebalance_map = None
    label_horizon_gap = None
    if LABEL_HORIZON_MODE == "next_rebalance":
        label_trade_dates = sorted(df["trade_date"].unique())
        label_rebalance_dates = get_rebalance_dates(label_trade_dates, LABEL_REBALANCE_FREQUENCY)
        if len(label_rebalance_dates) < 2:
            logger.warning(
                "label.horizon_mode=next_rebalance but insufficient rebalance dates; "
                "falling back to fixed horizon_days."
            )
            LABEL_HORIZON_MODE = "fixed"
        else:
            rebalance_array = np.array(label_rebalance_dates)
            trade_array = np.array(label_trade_dates)
            idx = np.searchsorted(rebalance_array, trade_array, side="right")
            next_dates = [
                rebalance_array[i] if i < len(rebalance_array) else pd.NaT
                for i in idx
            ]
            label_next_rebalance_map = dict(zip(label_trade_dates, next_dates))
            label_horizon_gap = estimate_rebalance_gap(label_trade_dates, label_rebalance_dates)
            if np.isfinite(label_horizon_gap):
                logger.info(
                    "Label horizon set to next rebalance date (median gap %.1f days).",
                    label_horizon_gap,
                )

    # -----------------------------------------------------------------------------
    # 3. Feature engineering (per symbol) + label
    # -----------------------------------------------------------------------------
    logger.info("Engineering features ...")

    if PRICE_COL not in df.columns:
        sys.exit(f"Price column '{PRICE_COL}' not found in data.")
    if not FEATURES:
        sys.exit("Feature list is empty.")


    def add_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trade_date").copy()
        needed = set(FEATURES)

        sma_windows = set(parse_feature_windows(FEATURES, "sma_"))
        sma_windows.update(parse_feature_windows(FEATURES, "sma_", "_diff"))
        if not sma_windows:
            sma_windows = set(feature_params.get("sma_windows", []))
        for win in sorted(sma_windows):
            group[f"sma_{win}"] = ta.sma(group["close"], length=win)
            if f"sma_{win}_diff" in needed:
                group[f"sma_{win}_diff"] = group[f"sma_{win}"].pct_change()

        rsi_lengths = set(parse_feature_windows(FEATURES, "rsi_"))
        if not rsi_lengths:
            rsi_cfg = feature_params.get("rsi")
            if isinstance(rsi_cfg, list):
                rsi_lengths.update(int(x) for x in rsi_cfg)
            elif rsi_cfg:
                rsi_lengths.add(int(rsi_cfg))
        for length in sorted(rsi_lengths):
            group[f"rsi_{length}"] = ta.rsi(group["close"], length=length)

        if "macd_hist" in needed:
            macd_cfg = feature_params.get("macd", [12, 26, 9])
            macd_fast, macd_slow, macd_signal = macd_cfg
            macd = ta.macd(group["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            col_name = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
            if macd is not None and col_name in macd.columns:
                group["macd_hist"] = macd[col_name]
            else:
                group["macd_hist"] = np.nan

        volume_windows = set(parse_feature_windows(FEATURES, "volume_sma", "_ratio"))
        if not volume_windows:
            vol_cfg = feature_params.get("volume_sma_windows", [])
            if isinstance(vol_cfg, list):
                volume_windows.update(int(x) for x in vol_cfg)
            elif vol_cfg:
                volume_windows.add(int(vol_cfg))
        for win in sorted(volume_windows):
            volume_sma = ta.sma(group["vol"], length=win)
            if volume_sma is None:
                volume_sma = group["vol"].rolling(window=win).mean()
            group[f"volume_sma{win}"] = volume_sma
            if f"volume_sma{win}_ratio" in needed:
                group[f"volume_sma{win}_ratio"] = group["vol"] / group[f"volume_sma{win}"]

        if LABEL_SHIFT_DAYS > 0:
            shifted_price = group[PRICE_COL].shift(-LABEL_SHIFT_DAYS)
        else:
            shifted_price = group[PRICE_COL]
        entry_price = shifted_price
        if LABEL_HORIZON_MODE == "next_rebalance" and label_next_rebalance_map is not None:
            exit_base = group["trade_date"].map(label_next_rebalance_map)
            shifted_by_date = pd.Series(shifted_price.values, index=group["trade_date"])
            exit_price = exit_base.map(shifted_by_date)
        else:
            exit_price = shifted_price.shift(-LABEL_HORIZON_DAYS)
        group[TARGET] = exit_price / entry_price - 1.0

        return group


    df = df.groupby("ts_code", group_keys=False).apply(add_features)

    missing_features = [feat for feat in FEATURES if feat not in df.columns]
    if missing_features:
        if FUNDAMENTALS_ALLOW_MISSING:
            logger.warning("Dropping missing features: %s", missing_features)
            FEATURES = [feat for feat in FEATURES if feat in df.columns]
        else:
            sys.exit(f"Missing features after engineering: {missing_features}")

    # Keep only the necessary columns; drop NaNs in features for live snapshot support
    meta_cols = ["is_tradable"] if "is_tradable" in df.columns else []
    eval_extra_df = None
    bucket_cols = []
    if BUCKET_IC_ENABLED and BUCKET_IC_SCHEMES:
        bucket_cols = list(dict.fromkeys([scheme["column"] for scheme in BUCKET_IC_SCHEMES]))
        missing_bucket_cols = [col for col in bucket_cols if col not in df.columns]
        if missing_bucket_cols:
            logger.warning("Bucket IC columns missing in data: %s", missing_bucket_cols)
        bucket_cols = [col for col in bucket_cols if col in df.columns]
        if bucket_cols:
            eval_extra_df = df[["trade_date", "ts_code"] + bucket_cols].copy()

    cols = ["trade_date", "ts_code", PRICE_COL] + FEATURES + meta_cols + [TARGET]
    cols = list(dict.fromkeys(cols))
    df = df[cols].copy()

    required_cols = [PRICE_COL] + FEATURES
    df_features = df.dropna(subset=required_cols).reset_index(drop=True)

    if universe_by_date is not None:
        before_rows = len(df_features)
        df_features = apply_universe_by_date(df_features, universe_by_date)
        after_rows = len(df_features)
        logger.info("Applied universe-by-date filter: %s -> %s rows", before_rows, after_rows)
        if df_features.empty:
            sys.exit("Universe-by-date filter removed all rows.")

    if WINSORIZE_PCT:
        def _winsorize(group: pd.DataFrame) -> pd.DataFrame:
            lower = group[TARGET].quantile(WINSORIZE_PCT)
            upper = group[TARGET].quantile(1 - WINSORIZE_PCT)
            group[TARGET] = group[TARGET].clip(lower, upper)
            return group

        df_features = df_features.groupby("trade_date", group_keys=False).apply(_winsorize)

    if CS_METHOD != "none":
        df_features = apply_cross_sectional_transform(
            df_features, FEATURES, CS_METHOD, CS_WINSORIZE_PCT
        )

    dataset_schema = DatasetSchema(
        date_col="trade_date",
        instrument_col="ts_code",
        price_col=PRICE_COL,
        label_col=TARGET,
        tradable_col="is_tradable" if "is_tradable" in df_features.columns else None,
        feature_cols=FEATURES,
    )
    dataset = build_dataset(df_features, dataset_schema)
    df_features = dataset.frame
    df_full = df_features.dropna().reset_index(drop=True)
    if eval_extra_df is not None and not eval_extra_df.empty:
        eval_extra_df = eval_extra_df.drop_duplicates(subset=["trade_date", "ts_code"])
        df_full = df_full.merge(eval_extra_df, on=["trade_date", "ts_code"], how="left")
    all_dates_full = np.array(sorted(df_full["trade_date"].unique()))
    rebalance_dates_all = None
    if SAMPLE_ON_REBALANCE_DATES:
        rebalance_dates_all = get_rebalance_dates(all_dates_full, REBALANCE_FREQUENCY)
        df_model_all = df_full[df_full["trade_date"].isin(rebalance_dates_all)].copy()
    else:
        df_model_all = df_full

    # Drop dates with too few symbols for evaluation (model sample only)
    date_counts = df_model_all.groupby("trade_date")["ts_code"].nunique()
    valid_dates = date_counts[date_counts >= MIN_SYMBOLS_PER_DATE].index
    dropped_date_counts = date_counts[date_counts < MIN_SYMBOLS_PER_DATE].sort_index()
    if len(valid_dates) != len(date_counts):
        df_model_all = df_model_all[df_model_all["trade_date"].isin(valid_dates)].copy()
    if not dropped_date_counts.empty:
        logger.info(
            "Dropped %s dates with < %s symbols (min=%s, max=%s).",
            len(dropped_date_counts),
            MIN_SYMBOLS_PER_DATE,
            int(dropped_date_counts.min()),
            int(dropped_date_counts.max()),
        )
    valid_dates_set = set(pd.to_datetime(valid_dates))

    def _resolve_holdout_len(value: object | None, n_dates: int) -> int:
        if value is None:
            return 0
        if n_dates <= 0:
            return 0
        try:
            size = float(value)
        except (TypeError, ValueError):
            raise SystemExit("eval.final_oos.size must be a number.")
        if size <= 0:
            return 0
        if size < 1:
            return max(1, int(np.floor(n_dates * size)))
        return int(size)

    df_model = df_model_all
    df_model_oos = pd.DataFrame()
    final_oos_dates = np.array([], dtype="datetime64[ns]")
    final_oos_len = 0
    final_oos_start = None
    final_oos_end = None
    if FINAL_OOS_ENABLED and FINAL_OOS_SIZE_RAW is None:
        FINAL_OOS_SIZE_RAW = TEST_SIZE
        logger.info(
            "final_oos.enabled=true but size not set; using eval.test_size=%s.",
            TEST_SIZE,
        )
    if FINAL_OOS_ENABLED:
        all_dates_model_full = np.array(sorted(df_model_all["trade_date"].unique()))
        final_oos_len = _resolve_holdout_len(FINAL_OOS_SIZE_RAW, len(all_dates_model_full))
        if final_oos_len <= 0:
            FINAL_OOS_ENABLED = False
        elif final_oos_len >= len(all_dates_model_full):
            sys.exit("eval.final_oos.size leaves no in-sample dates.")
        else:
            final_oos_dates = all_dates_model_full[-final_oos_len:]
            final_oos_start = pd.to_datetime(final_oos_dates[0])
            final_oos_end = pd.to_datetime(final_oos_dates[-1])
            df_model_oos = df_model_all[df_model_all["trade_date"].isin(final_oos_dates)].copy()
            df_model = df_model_all[~df_model_all["trade_date"].isin(final_oos_dates)].copy()
            logger.info(
                "Final OOS holdout enabled: %s dates (%s -> %s).",
                final_oos_len,
                final_oos_start.strftime("%Y-%m-%d"),
                final_oos_end.strftime("%Y-%m-%d"),
            )

    rebalance_gap_days = None
    if SAMPLE_ON_REBALANCE_DATES:
        sample_dates = sorted(df_model["trade_date"].unique())
        if len(sample_dates) >= 2:
            rebalance_gap_days = estimate_rebalance_gap(all_dates_full, sample_dates)
            if np.isfinite(rebalance_gap_days):
                logger.info(
                    "Sample-on-rebalance enabled: median gap %.1f trade days.",
                    rebalance_gap_days,
                )

    # -----------------------------------------------------------------------------
    # 4. Train-test split (time-series by date)
    # -----------------------------------------------------------------------------
    logger.info("Splitting train/test by date ...")
    label_horizon_effective = LABEL_HORIZON_DAYS
    if LABEL_HORIZON_MODE == "next_rebalance" and label_horizon_gap is not None:
        if np.isfinite(label_horizon_gap):
            label_horizon_effective = int(round(label_horizon_gap))
    if PURGE_DAYS_CFG is None:
        purge_days = int(label_horizon_effective + LABEL_SHIFT_DAYS)
    else:
        purge_days = int(PURGE_DAYS_CFG)
    embargo_days = int(EMBARGO_DAYS_CFG)

    def _days_to_steps(days: int, gap_days: Optional[float]) -> int:
        if days <= 0:
            return 0
        if gap_days is None or not np.isfinite(gap_days) or gap_days <= 0:
            return int(days)
        return max(0, int(math.ceil(days / gap_days)))

    if SAMPLE_ON_REBALANCE_DATES:
        PURGE_STEPS = _days_to_steps(purge_days, rebalance_gap_days)
        EMBARGO_STEPS = _days_to_steps(embargo_days, rebalance_gap_days)
        if rebalance_gap_days is not None and np.isfinite(rebalance_gap_days):
            logger.info(
                "Converted embargo/purge from days to rebalance steps: "
                "embargo=%s->%s, purge=%s->%s (gap≈%.1f days).",
                embargo_days,
                EMBARGO_STEPS,
                purge_days,
                PURGE_STEPS,
                rebalance_gap_days,
            )
        else:
            logger.warning(
                "Sample-on-rebalance enabled but rebalance gap could not be estimated; "
                "using raw embargo/purge values as steps."
            )
    else:
        PURGE_STEPS = purge_days
        EMBARGO_STEPS = embargo_days

    EFFECTIVE_GAP_STEPS = max(EMBARGO_STEPS, PURGE_STEPS)

    all_dates = np.array(sorted(df_model["trade_date"].unique()))
    if len(all_dates) < 10:
        sys.exit("Not enough dates for a meaningful split.")

    split_idx = int(len(all_dates) * (1 - TEST_SIZE))
    train_end = split_idx
    if EFFECTIVE_GAP_STEPS > 0:
        train_end = max(0, split_idx - EFFECTIVE_GAP_STEPS)
    train_dates = all_dates[:train_end]
    test_dates = all_dates[split_idx:]

    train_df = df_model[df_model["trade_date"].isin(train_dates)].copy()
    test_df = df_model[df_model["trade_date"].isin(test_dates)].copy()

    if train_df.empty or test_df.empty:
        sys.exit("Not enough dates for train/test after embargo.")

    logger.info(
        "Train/test split: train_dates=%s, test_dates=%s, purge_steps=%s, embargo_steps=%s.",
        len(train_dates),
        len(test_dates),
        PURGE_STEPS,
        EMBARGO_STEPS,
    )

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # -----------------------------------------------------------------------------
    # 5. Cross-validation on dates (IC metric)
    # -----------------------------------------------------------------------------
    logger.info("Time-series cross-validation (IC) ...")


    def permute_target_within_date(
        data: pd.DataFrame,
        target_col: str,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        def _permute(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group[target_col] = rng.permutation(group[target_col].values)
            return group

        return data.groupby("trade_date", group_keys=False, sort=False).apply(_permute)


    def permutation_test_ic(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_runs: int,
        seed: Optional[int],
        signal_direction: float,
    ) -> list[float]:
        scores = []
        for idx in range(n_runs):
            run_seed = None if seed is None else seed + idx
            rng = np.random.default_rng(run_seed)
            perm_train = permute_target_within_date(train_data, TARGET, rng)

            perm_model = XGBRegressor(**XGB_PARAMS)
            perm_weights = build_sample_weight(perm_train, SAMPLE_WEIGHT_MODE)
            if perm_weights is not None:
                perm_model.fit(perm_train[FEATURES], perm_train[TARGET], sample_weight=perm_weights)
            else:
                perm_model.fit(perm_train[FEATURES], perm_train[TARGET])

            perm_test = test_data.copy()
            perm_test["pred"] = perm_model.predict(perm_test[FEATURES])
            if signal_direction != 1.0:
                perm_test["pred"] = perm_test["pred"] * signal_direction

            ic_values = daily_ic_series(perm_test, TARGET, "pred")
            scores.append(float(ic_values.mean()) if not ic_values.empty else np.nan)
        return scores


    def evaluate_window(window_meta: dict) -> dict:
        window_id = int(window_meta["window"])
        train_dates = window_meta["train_dates"]
        test_dates = window_meta["test_dates"]
        train_df_w = df_model[df_model["trade_date"].isin(train_dates)].copy()
        test_df_w = df_model[df_model["trade_date"].isin(test_dates)].copy()
        result = {
            "window": window_id,
            "train_start": pd.to_datetime(window_meta["train_start"]).strftime("%Y-%m-%d"),
            "train_end": pd.to_datetime(window_meta["train_end"]).strftime("%Y-%m-%d"),
            "test_start": pd.to_datetime(window_meta["test_start"]).strftime("%Y-%m-%d"),
            "test_end": pd.to_datetime(window_meta["test_end"]).strftime("%Y-%m-%d"),
            "status": "ok",
        }
        if train_df_w.empty or test_df_w.empty:
            result["status"] = "insufficient_data"
            return result

        direction = SIGNAL_DIRECTION
        cv_stats = None
        if SIGNAL_DIRECTION_MODE == "cv_ic":
            cv_scores_w = time_series_cv_ic(
                train_df_w,
                FEATURES,
                TARGET,
                N_SPLITS,
                EMBARGO_STEPS,
                PURGE_STEPS,
                XGB_PARAMS,
                1.0,
                sample_weight_mode=SAMPLE_WEIGHT_MODE,
            )
            if cv_scores_w:
                cv_mean = float(np.nanmean(cv_scores_w))
                cv_std = float(np.nanstd(cv_scores_w))
                if np.isfinite(cv_mean) and cv_mean != 0 and abs(cv_mean) >= MIN_ABS_IC_TO_FLIP:
                    direction = float(np.sign(cv_mean))
                cv_stats = {
                    "mean": cv_mean,
                    "std": cv_std,
                    "scores": [float(score) for score in cv_scores_w],
                }

        model_w = XGBRegressor(**XGB_PARAMS)
        train_weights_w = build_sample_weight(train_df_w, SAMPLE_WEIGHT_MODE)
        if train_weights_w is not None:
            model_w.fit(train_df_w[FEATURES], train_df_w[TARGET], sample_weight=train_weights_w)
        else:
            model_w.fit(train_df_w[FEATURES], train_df_w[TARGET])

        train_eval = train_df_w.copy()
        train_eval["pred"] = model_w.predict(train_eval[FEATURES])
        train_ic_raw_stats = None
        if SIGNAL_DIRECTION_MODE == "train_ic":
            train_ic_raw = daily_ic_series(train_eval, TARGET, "pred")
            train_ic_raw_stats = summarize_ic(train_ic_raw)
            raw_mean = train_ic_raw_stats.get("mean", np.nan)
            if np.isfinite(raw_mean) and raw_mean != 0:
                direction = float(np.sign(raw_mean))
            else:
                direction = 1.0

        train_signal_col = "pred"
        if direction != 1.0:
            train_eval["signal"] = train_eval["pred"] * direction
            train_signal_col = "signal"

        train_ic_stats = {}
        if REPORT_TRAIN_IC:
            train_ic_stats = summarize_ic(daily_ic_series(train_eval, TARGET, train_signal_col))

        test_eval = test_df_w.copy()
        test_eval["pred"] = model_w.predict(test_eval[FEATURES])
        signal_col_w = "pred"
        if direction != 1.0:
            test_eval["signal"] = test_eval["pred"] * direction
            signal_col_w = "signal"

        ic_stats_w = summarize_ic(daily_ic_series(test_eval, TARGET, signal_col_w))
        pearson_ic_stats_w = summarize_ic(
            daily_ic_series(test_eval, TARGET, signal_col_w, method="pearson")
        )
        error_metrics_w = regression_error_metrics(test_eval[TARGET], test_eval[signal_col_w])
        hit_rate_w = hit_rate(test_eval[TARGET], test_eval[signal_col_w])

        perm_stats_w = None
        if WF_PERM_TEST_ENABLED:
            perm_scores = permutation_test_ic(
                train_df_w,
                test_df_w,
                WF_PERM_TEST_RUNS,
                WF_PERM_TEST_SEED,
                direction,
            )
            if perm_scores:
                perm_stats_w = {
                    "mean": float(np.nanmean(perm_scores)),
                    "std": float(np.nanstd(perm_scores)),
                    "scores": [float(score) for score in perm_scores],
                    "runs": int(len(perm_scores)),
                }

        trade_dates_sorted = sorted(test_eval["trade_date"].unique())
        rebalance_dates_w = get_rebalance_dates(trade_dates_sorted, REBALANCE_FREQUENCY)
        if valid_dates_set:
            rebalance_dates_w = [d for d in rebalance_dates_w if d in valid_dates_set]
        if SAMPLE_ON_REBALANCE_DATES:
            test_dates_set = set(pd.to_datetime(test_dates))
            rebalance_dates_w = [d for d in rebalance_dates_w if d in test_dates_set]
        eval_df_w = test_eval[test_eval["trade_date"].isin(rebalance_dates_w)].copy()

        quantile_ts_w = quantile_returns(eval_df_w, signal_col_w, TARGET, N_QUANTILES)
        quantile_mean_w = quantile_ts_w.mean() if not quantile_ts_w.empty else pd.Series(dtype=float)
        long_short_w = (
            float(quantile_mean_w.iloc[-1] - quantile_mean_w.iloc[0])
            if not quantile_mean_w.empty
            else None
        )

        k_w = min(TOP_K, eval_df_w["ts_code"].nunique())
        turnover_series_w = estimate_turnover(
            eval_df_w,
            signal_col_w,
            k_w,
            rebalance_dates_w,
            buffer_exit=EVAL_BUFFER_EXIT,
            buffer_entry=EVAL_BUFFER_ENTRY,
        )
        turnover_mean_w = (
            float(turnover_series_w.mean()) if not turnover_series_w.empty else None
        )

        topk_positive_w = topk_positive_ratio(eval_df_w, signal_col_w, TARGET, k_w)

        bt_stats_w = None
        bt_benchmark_stats_w = None
        bt_active_stats_w = None
        if WF_BACKTEST_ENABLED:
            bt_direction = direction if BACKTEST_SIGNAL_DIRECTION_RAW is None else BACKTEST_SIGNAL_DIRECTION_RAW
            bt_pred_col = "pred"
            test_start = pd.to_datetime(window_meta["test_start"])
            test_end = pd.to_datetime(window_meta["test_end"])
            test_full_w = df_full[
                (df_full["trade_date"] >= test_start) & (df_full["trade_date"] <= test_end)
            ].copy()
            if test_full_w.empty:
                bt_result_w = None
            else:
                test_full_w["pred"] = model_w.predict(test_full_w[FEATURES])
                if bt_direction != 1.0:
                    test_full_w["signal_bt"] = test_full_w["pred"] * bt_direction
                    bt_pred_col = "signal_bt"
                bt_rebalance = get_rebalance_dates(
                    sorted(test_full_w["trade_date"].unique()), BACKTEST_REBALANCE_FREQUENCY
                )
                if valid_dates_set:
                    bt_rebalance = [d for d in bt_rebalance if d in valid_dates_set]
                try:
                    bt_result_w = backtest_topk(
                        test_full_w,
                        pred_col=bt_pred_col,
                        price_col=PRICE_COL,
                        rebalance_dates=bt_rebalance,
                        top_k=BACKTEST_TOP_K,
                        shift_days=LABEL_SHIFT_DAYS,
                        cost_bps=BACKTEST_COST_BPS_EFFECTIVE,
                        trading_days_per_year=BACKTEST_TRADING_DAYS_PER_YEAR,
                        exit_mode=BACKTEST_EXIT_MODE,
                        exit_horizon_days=BACKTEST_EXIT_HORIZON_DAYS,
                        long_only=BACKTEST_LONG_ONLY,
                        short_k=BACKTEST_SHORT_K,
                        buffer_exit=BACKTEST_BUFFER_EXIT,
                        buffer_entry=BACKTEST_BUFFER_ENTRY,
                        tradable_col=BACKTEST_TRADABLE_COL if BACKTEST_TRADABLE_COL in test_full_w.columns else None,
                        exit_price_policy=BACKTEST_EXIT_PRICE_POLICY,
                        exit_fallback_policy=BACKTEST_EXIT_FALLBACK_POLICY,
                        execution=execution_model,
                    )
                except ValueError:
                    bt_result_w = None
            if bt_result_w is not None:
                bt_stats_w, bt_net_w, _, _, bt_periods_w = bt_result_w
                if benchmark_df is not None and not benchmark_df.empty:
                    bench_series_w, bench_periods_w = build_benchmark_series(
                        benchmark_df, PRICE_COL, bt_periods_w
                    )
                    if not bench_series_w.empty:
                        bt_benchmark_stats_w = summarize_period_returns(
                            bench_series_w, bench_periods_w, BACKTEST_TRADING_DAYS_PER_YEAR
                        )
                        periods_per_year = bt_stats_w.get("periods_per_year", np.nan)
                        bt_active_stats_w, _ = summarize_active_returns(
                            bt_net_w, bench_series_w, periods_per_year
                        )

        result.update(
            {
                "signal_direction": direction,
                "signal_direction_mode": SIGNAL_DIRECTION_MODE,
                "cv_ic": cv_stats,
                "train_ic": train_ic_stats if REPORT_TRAIN_IC else None,
                "train_ic_raw": train_ic_raw_stats,
                "test_ic": ic_stats_w,
                "test_pearson_ic": pearson_ic_stats_w,
                "error_metrics": error_metrics_w,
                "hit_rate": hit_rate_w,
                "topk_positive_ratio": topk_positive_w,
                "long_short": long_short_w,
                "turnover_mean": turnover_mean_w,
                "backtest": {
                    "stats": bt_stats_w,
                    "benchmark": bt_benchmark_stats_w,
                    "active": bt_active_stats_w,
                }
                if WF_BACKTEST_ENABLED
                else None,
                "permutation_test": perm_stats_w,
            }
        )
        return result

    cv_scores_raw = time_series_cv_ic(
        train_df,
        FEATURES,
        TARGET,
        N_SPLITS,
        EMBARGO_STEPS,
        PURGE_STEPS,
        XGB_PARAMS,
        1.0,
        sample_weight_mode=SAMPLE_WEIGHT_MODE,
    )
    if cv_scores_raw:
        logger.info(
            "CV IC (raw): mean=%.4f, std=%.4f", np.nanmean(cv_scores_raw), np.nanstd(cv_scores_raw)
        )
        logger.info("CV fold ICs (raw): %s", [f"{s:.4f}" for s in cv_scores_raw])
    else:
        logger.info("CV IC not available - insufficient data after embargo/purge.")

    cv_scores_adj = None
    if SIGNAL_DIRECTION_MODE == "cv_ic" and cv_scores_raw:
        cv_mean = float(np.nanmean(cv_scores_raw))
        if np.isfinite(cv_mean) and cv_mean != 0 and abs(cv_mean) >= MIN_ABS_IC_TO_FLIP:
            SIGNAL_DIRECTION = float(np.sign(cv_mean))
            logger.info("Signal direction set from CV IC: %s", SIGNAL_DIRECTION)
        else:
            logger.info(
                "CV IC mean below threshold (|mean| < %.4f); keeping signal direction: %s",
                MIN_ABS_IC_TO_FLIP,
                SIGNAL_DIRECTION,
            )

    # -----------------------------------------------------------------------------
    # 6. Fit final model
    # -----------------------------------------------------------------------------
    logger.info("Fitting XGBoost regressor ...")
    model = XGBRegressor(**XGB_PARAMS)
    train_weights = build_sample_weight(train_df, SAMPLE_WEIGHT_MODE)
    if train_weights is not None:
        model.fit(X_train, y_train, sample_weight=train_weights)
    else:
        model.fit(X_train, y_train)

    # -----------------------------------------------------------------------------
    # 7. Evaluation (cross-sectional factor style)
    # -----------------------------------------------------------------------------
    logger.info("Evaluating model on train/test sets ...")

    test_start = pd.to_datetime(test_dates[0])
    test_end = pd.to_datetime(test_dates[-1])
    test_df_full = df_full[
        (df_full["trade_date"] >= test_start) & (df_full["trade_date"] <= test_end)
    ].copy()
    if test_df_full.empty:
        sys.exit("Not enough test data after applying the split window.")

    train_eval_df = train_df.copy()
    train_eval_df["pred"] = model.predict(train_eval_df[FEATURES])
    train_ic_raw_stats = {}
    if SIGNAL_DIRECTION_MODE == "train_ic":
        train_ic_raw_series = daily_ic_series(train_eval_df, TARGET, "pred")
        train_ic_raw_stats = summarize_ic(train_ic_raw_series)
        raw_mean = train_ic_raw_stats.get("mean", np.nan)
        if np.isfinite(raw_mean) and raw_mean != 0:
            SIGNAL_DIRECTION = float(np.sign(raw_mean))
        else:
            SIGNAL_DIRECTION = 1.0
        logger.info("Signal direction set from Train IC: %s", SIGNAL_DIRECTION)

    train_signal_col = "pred"
    if SIGNAL_DIRECTION != 1.0:
        train_eval_df["signal"] = train_eval_df["pred"] * SIGNAL_DIRECTION
        train_signal_col = "signal"

    if cv_scores_raw:
        cv_scores_adj = [float(score) * SIGNAL_DIRECTION for score in cv_scores_raw]
        if SIGNAL_DIRECTION != 1.0:
            logger.info(
                "CV IC (adj): mean=%.4f, std=%.4f",
                np.nanmean(cv_scores_adj),
                np.nanstd(cv_scores_adj),
            )
            logger.info("CV fold ICs (adj): %s", [f"{s:.4f}" for s in cv_scores_adj])

    train_ic_series = pd.Series(dtype=float, name="ic")
    train_ic_stats = {}
    train_pearson_ic_series = pd.Series(dtype=float, name="ic_pearson")
    train_pearson_ic_stats = {}
    if REPORT_TRAIN_IC:
        train_ic_series = daily_ic_series(train_eval_df, TARGET, train_signal_col)
        train_ic_stats = summarize_ic(train_ic_series)
        logger.info(
            "Train Daily IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
            train_ic_stats["mean"],
            train_ic_stats["std"],
            train_ic_stats["ir"],
            train_ic_stats["t_stat"],
            train_ic_stats["p_value"],
            train_ic_stats["n"],
        )
        train_pearson_ic_series = daily_ic_series(
            train_eval_df, TARGET, train_signal_col, method="pearson"
        )
        train_pearson_ic_stats = summarize_ic(train_pearson_ic_series)
        logger.info(
            "Train Daily Pearson IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
            train_pearson_ic_stats["mean"],
            train_pearson_ic_stats["std"],
            train_pearson_ic_stats["ir"],
            train_pearson_ic_stats["t_stat"],
            train_pearson_ic_stats["p_value"],
            train_pearson_ic_stats["n"],
        )

    positions_by_rebalance_live = None
    live_as_of = None
    live_positions_ready = False
    if LIVE_ENABLED:
        live_as_of = _resolve_date_token(
            LIVE_AS_OF, default="t-1", market=MARKET, provider=provider
        )
        df_live = df_features[df_features["trade_date"] <= live_as_of].copy()
        if df_live.empty:
            logger.warning("Live snapshot skipped: no data on or before %s.", live_as_of.date())
        else:
            df_live_labeled = df_live[df_live[TARGET].notna()].copy()
            if df_live_labeled.empty:
                logger.warning("Live snapshot skipped: no labeled data on or before %s.", live_as_of.date())
            else:
                live_model = model
                if LIVE_TRAIN_MODE == "full":
                    live_model = XGBRegressor(**XGB_PARAMS)
                    live_weights = build_sample_weight(df_live_labeled, SAMPLE_WEIGHT_MODE)
                    if live_weights is not None:
                        live_model.fit(
                            df_live_labeled[FEATURES],
                            df_live_labeled[TARGET],
                            sample_weight=live_weights,
                        )
                    else:
                        live_model.fit(df_live_labeled[FEATURES], df_live_labeled[TARGET])

                df_live["pred"] = live_model.predict(df_live[FEATURES])
                live_pred_col = "pred"
                if SIGNAL_DIRECTION != 1.0:
                    df_live["signal"] = df_live["pred"] * SIGNAL_DIRECTION
                    live_pred_col = "signal"

                live_dates = sorted(df_live["trade_date"].unique())
                live_rebalance = get_rebalance_dates(live_dates, BACKTEST_REBALANCE_FREQUENCY)
                live_counts = df_live.groupby("trade_date")["ts_code"].nunique()
                live_valid_dates = set(
                    live_counts[live_counts >= MIN_SYMBOLS_PER_DATE].index
                )
                live_rebalance = [d for d in live_rebalance if d in live_valid_dates]

                positions_by_rebalance_live = build_positions_by_rebalance(
                    df_live,
                    pred_col=live_pred_col,
                    price_col=PRICE_COL,
                    rebalance_dates=live_rebalance,
                    top_k=BACKTEST_TOP_K,
                    shift_days=LABEL_SHIFT_DAYS,
                    buffer_exit=BACKTEST_BUFFER_EXIT,
                    buffer_entry=BACKTEST_BUFFER_ENTRY,
                    long_only=BACKTEST_LONG_ONLY,
                    short_k=BACKTEST_SHORT_K,
                    tradable_col=BACKTEST_TRADABLE_COL if BACKTEST_TRADABLE_COL in df_live.columns else None,
                )

                if positions_by_rebalance_live is None or positions_by_rebalance_live.empty:
                    logger.warning("Live snapshot skipped: no positions generated.")
                else:
                    live_positions_ready = True
                    entry_dates_live = pd.to_datetime(
                        positions_by_rebalance_live["entry_date"], errors="coerce"
                    )
                    if entry_dates_live.notna().any():
                        latest_entry = entry_dates_live.max()
                        holdings_count = int((entry_dates_live == latest_entry).sum())
                        logger.info(
                            "Live snapshot ready: as_of=%s, entry_date=%s, holdings=%s",
                            live_as_of.strftime("%Y-%m-%d"),
                            latest_entry.strftime("%Y-%m-%d"),
                            holdings_count,
                        )

    if LIVE_ENABLED and not BACKTEST_ENABLED and not live_positions_ready:
        raise SystemExit(
            "live.enabled=true but no live positions were generated; "
            "refusing to fall back to backtest holdings."
        )

    def evaluate_period(
        label: str,
        model_eval: XGBRegressor,
        test_df_full: pd.DataFrame,
        test_dates: np.ndarray,
        *,
        run_perm_test: bool,
        perm_train_df: Optional[pd.DataFrame] = None,
        perm_test_df: Optional[pd.DataFrame] = None,
        allow_live_fallback: bool = True,
    ) -> dict:
        label_prefix = f"[{label}] " if label else ""
        default_series = pd.Series(dtype=float)
        default_frame = pd.DataFrame()
        result = {
            "ic_series": default_series,
            "ic_stats": {},
            "pearson_ic_series": default_series,
            "pearson_ic_stats": {},
            "error_metrics": {},
            "hit_rate": {},
            "topk_positive_ratio": {},
            "bucket_ic": [],
            "quantile_ts": default_frame,
            "quantile_mean": default_series,
            "turnover_series": default_series,
            "positions_by_rebalance": None,
            "bt_stats": None,
            "bt_net_series": pd.Series(dtype=float, name="net_return"),
            "bt_gross_series": pd.Series(dtype=float, name="gross_return"),
            "bt_turnover_series": pd.Series(dtype=float, name="turnover"),
            "bt_benchmark_series": pd.Series(dtype=float, name="benchmark_return"),
            "bt_active_series": pd.Series(dtype=float, name="active_return"),
            "bt_benchmark_stats": None,
            "bt_active_stats": None,
            "bt_periods": [],
            "perm_stats": None,
        }
        if test_df_full is None or test_df_full.empty:
            logger.info("%sEvaluation skipped: no data.", label_prefix)
            return result

        eval_df_full = test_df_full.copy()
        eval_df_full["pred"] = model_eval.predict(eval_df_full[FEATURES])
        if SAMPLE_ON_REBALANCE_DATES:
            test_eval_df = eval_df_full[eval_df_full["trade_date"].isin(test_dates)].copy()
        else:
            test_eval_df = eval_df_full
        signal_col = "pred"
        if SIGNAL_DIRECTION != 1.0:
            eval_df_full["signal"] = eval_df_full["pred"] * SIGNAL_DIRECTION
            if SAMPLE_ON_REBALANCE_DATES:
                test_eval_df["signal"] = test_eval_df["pred"] * SIGNAL_DIRECTION
            signal_col = "signal"
            logger.info("%sSignal direction applied to ranking: %s", label_prefix, SIGNAL_DIRECTION)

        ic_series = daily_ic_series(test_eval_df, TARGET, signal_col)
        ic_stats = summarize_ic(ic_series)
        logger.info(
            "%sDaily IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
            label_prefix,
            ic_stats["mean"],
            ic_stats["std"],
            ic_stats["ir"],
            ic_stats["t_stat"],
            ic_stats["p_value"],
            ic_stats["n"],
        )
        result["ic_series"] = ic_series
        result["ic_stats"] = ic_stats

        pearson_ic_series = daily_ic_series(test_eval_df, TARGET, signal_col, method="pearson")
        pearson_ic_stats = summarize_ic(pearson_ic_series)
        logger.info(
            "%sDaily Pearson IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
            label_prefix,
            pearson_ic_stats["mean"],
            pearson_ic_stats["std"],
            pearson_ic_stats["ir"],
            pearson_ic_stats["t_stat"],
            pearson_ic_stats["p_value"],
            pearson_ic_stats["n"],
        )
        result["pearson_ic_series"] = pearson_ic_series
        result["pearson_ic_stats"] = pearson_ic_stats

        error_metrics = regression_error_metrics(test_eval_df[TARGET], test_eval_df[signal_col])
        result["error_metrics"] = error_metrics
        if error_metrics and error_metrics.get("n", 0) > 0:
            logger.info(
                "%sError metrics: MAE=%.6f, RMSE=%.6f, R2=%.4f (n=%s)",
                label_prefix,
                error_metrics.get("mae", np.nan),
                error_metrics.get("rmse", np.nan),
                error_metrics.get("r2", np.nan),
                error_metrics.get("n", 0),
            )

        hit_stats = hit_rate(test_eval_df[TARGET], test_eval_df[signal_col])
        result["hit_rate"] = hit_stats
        if hit_stats and hit_stats.get("n", 0) > 0:
            logger.info(
                "%sHit rate: %.2f%% (n=%s)",
                label_prefix,
                hit_stats.get("hit_rate", np.nan) * 100,
                hit_stats.get("n", 0),
            )

        if run_perm_test:
            if perm_train_df is None or perm_test_df is None:
                raise SystemExit("Permutation test requested but data was not provided.")
            logger.info("%sPermutation test (shuffle train labels within date) ...", label_prefix)
            perm_scores = permutation_test_ic(
                perm_train_df,
                perm_test_df,
                PERM_TEST_RUNS,
                PERM_TEST_SEED,
                SIGNAL_DIRECTION,
            )
            if perm_scores:
                perm_mean = np.nanmean(perm_scores)
                perm_std = np.nanstd(perm_scores)
                logger.info(
                    "%sPermutation IC: mean=%.4f, std=%.4f, runs=%s",
                    label_prefix,
                    perm_mean,
                    perm_std,
                    len(perm_scores),
                )
                logger.info("%sPermutation ICs: %s", label_prefix, [f"{s:.4f}" for s in perm_scores])
                result["perm_stats"] = {
                    "mean": float(perm_mean),
                    "std": float(perm_std),
                    "scores": [float(score) for score in perm_scores],
                    "runs": int(len(perm_scores)),
                }

        trade_dates_sorted_full = sorted(eval_df_full["trade_date"].unique())
        rebalance_dates_full = get_rebalance_dates(trade_dates_sorted_full, REBALANCE_FREQUENCY)
        rebalance_gap = estimate_rebalance_gap(trade_dates_sorted_full, rebalance_dates_full)
        if (
            BACKTEST_EXIT_MODE == "rebalance"
            and np.isfinite(rebalance_gap)
            and LABEL_HORIZON_MODE == "fixed"
        ):
            gap_diff = abs(rebalance_gap - label_horizon_effective)
            if gap_diff >= max(3.0, rebalance_gap * 0.25):
                logger.warning(
                    "%sLabel horizon (%s days) differs from rebalance gap (median %.1f days).",
                    label_prefix,
                    label_horizon_effective,
                    rebalance_gap,
                )
        rebalance_dates_eval = rebalance_dates_full
        if valid_dates_set:
            rebalance_dates_eval = [d for d in rebalance_dates_eval if d in valid_dates_set]
        if SAMPLE_ON_REBALANCE_DATES:
            test_dates_set = set(pd.to_datetime(test_dates))
            rebalance_dates_eval = [d for d in rebalance_dates_eval if d in test_dates_set]
        eval_df = test_eval_df[test_eval_df["trade_date"].isin(rebalance_dates_eval)].copy()

        quantile_ts = quantile_returns(eval_df, signal_col, TARGET, N_QUANTILES)
        quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
        result["quantile_ts"] = quantile_ts
        result["quantile_mean"] = quantile_mean
        if not quantile_mean.empty:
            for q_idx, value in quantile_mean.items():
                logger.info("%sQ%s mean return: %.4f%%", label_prefix, int(q_idx) + 1, value * 100)
            long_short = quantile_mean.iloc[-1] - quantile_mean.iloc[0]
            logger.info(
                "%sLong-short (Q%s-Q1): %.4f%%", label_prefix, N_QUANTILES, long_short * 100
            )
        else:
            logger.info("%sQuantile returns not available - insufficient symbols per date.", label_prefix)

        k = min(TOP_K, eval_df["ts_code"].nunique()) if not eval_df.empty else 0
        turnover_series = estimate_turnover(
            eval_df,
            signal_col,
            k,
            rebalance_dates_eval,
            buffer_exit=EVAL_BUFFER_EXIT,
            buffer_entry=EVAL_BUFFER_ENTRY,
        )
        result["turnover_series"] = turnover_series
        if not turnover_series.empty:
            turnover = turnover_series.mean()
            cost_drag = 2 * (TRANSACTION_COST_BPS / 10000.0) * turnover
            logger.info(
                "%sTop-%s turnover per rebalance: %.2f%% (n=%s)",
                label_prefix,
                k,
                turnover * 100,
                len(turnover_series),
            )
            logger.info(
                "%sApprox cost drag per rebalance: %.2f%% at %s bps per side",
                label_prefix,
                cost_drag * 100,
                TRANSACTION_COST_BPS,
            )

        topk_stats = topk_positive_ratio(eval_df, signal_col, TARGET, k)
        result["topk_positive_ratio"] = topk_stats
        if topk_stats and topk_stats.get("n_dates", 0) > 0:
            logger.info(
                "%sTop-%s positive ratio: %.2f%% (n=%s)",
                label_prefix,
                k,
                topk_stats.get("topk_positive_ratio", np.nan) * 100,
                topk_stats.get("n_dates", 0),
            )

        if BUCKET_IC_ENABLED and BUCKET_IC_SCHEMES:
            bucket_frames = []
            for scheme in BUCKET_IC_SCHEMES:
                col = scheme["column"]
                if col not in test_eval_df.columns:
                    continue
                bucket_type = str(scheme.get("type", "category")).strip().lower()
                if bucket_type not in {"category", "quantile"}:
                    bucket_type = "category"
                data_for_bucket = test_eval_df.copy()
                bucket_col = col
                if bucket_type == "quantile":
                    n_bins = int(scheme.get("n_bins") or 0)
                    if n_bins < 2:
                        continue
                    bucket_col = f"bucket_{scheme['name']}"
                    data_for_bucket[bucket_col] = assign_daily_quantile_bucket(
                        data_for_bucket, col, n_bins
                    )
                summary_df = bucket_ic_summary(
                    data_for_bucket,
                    TARGET,
                    signal_col,
                    bucket_col,
                    method=BUCKET_IC_METHOD,
                    min_count=BUCKET_IC_MIN_COUNT,
                )
                if not summary_df.empty:
                    summary_df.insert(0, "scheme", scheme["name"])
                    summary_df.insert(1, "type", bucket_type)
                    if bucket_type == "quantile":
                        summary_df.insert(2, "n_bins", int(scheme.get("n_bins") or 0))
                    summary_df["method"] = BUCKET_IC_METHOD
                    bucket_frames.append(summary_df)
            if bucket_frames:
                bucket_df = pd.concat(bucket_frames, ignore_index=True)
                result["bucket_ic"] = bucket_df.to_dict(orient="records")

        bt_rebalance = get_rebalance_dates(
            sorted(eval_df_full["trade_date"].unique()), BACKTEST_REBALANCE_FREQUENCY
        )
        if valid_dates_set:
            bt_rebalance = [d for d in bt_rebalance if d in valid_dates_set]
        bt_pred_col = signal_col
        if BACKTEST_SIGNAL_DIRECTION != SIGNAL_DIRECTION:
            eval_df_full["signal_bt"] = eval_df_full["pred"] * BACKTEST_SIGNAL_DIRECTION
            bt_pred_col = "signal_bt"

        positions_by_rebalance = None
        if BACKTEST_ENABLED or not LIVE_ENABLED or not allow_live_fallback:
            positions_by_rebalance = build_positions_by_rebalance(
                eval_df_full,
                pred_col=bt_pred_col,
                price_col=PRICE_COL,
                rebalance_dates=bt_rebalance,
                top_k=BACKTEST_TOP_K,
                shift_days=LABEL_SHIFT_DAYS,
                buffer_exit=BACKTEST_BUFFER_EXIT,
                buffer_entry=BACKTEST_BUFFER_ENTRY,
                long_only=BACKTEST_LONG_ONLY,
                short_k=BACKTEST_SHORT_K,
                tradable_col=BACKTEST_TRADABLE_COL
                if BACKTEST_TRADABLE_COL in eval_df_full.columns
                else None,
            )
        if allow_live_fallback and LIVE_ENABLED and not BACKTEST_ENABLED:
            positions_by_rebalance = positions_by_rebalance_live
        result["positions_by_rebalance"] = positions_by_rebalance

        bt_result = None
        bt_attempted = False
        if BACKTEST_ENABLED:
            bt_attempted = True
            try:
                bt_result = backtest_topk(
                    eval_df_full,
                    pred_col=bt_pred_col,
                    price_col=PRICE_COL,
                    rebalance_dates=bt_rebalance,
                    top_k=BACKTEST_TOP_K,
                    shift_days=LABEL_SHIFT_DAYS,
                    cost_bps=BACKTEST_COST_BPS_EFFECTIVE,
                    trading_days_per_year=BACKTEST_TRADING_DAYS_PER_YEAR,
                    exit_mode=BACKTEST_EXIT_MODE,
                    exit_horizon_days=BACKTEST_EXIT_HORIZON_DAYS,
                    long_only=BACKTEST_LONG_ONLY,
                    short_k=BACKTEST_SHORT_K,
                    buffer_exit=BACKTEST_BUFFER_EXIT,
                    buffer_entry=BACKTEST_BUFFER_ENTRY,
                    tradable_col=BACKTEST_TRADABLE_COL
                    if BACKTEST_TRADABLE_COL in eval_df_full.columns
                    else None,
                    exit_price_policy=BACKTEST_EXIT_PRICE_POLICY,
                    exit_fallback_policy=BACKTEST_EXIT_FALLBACK_POLICY,
                    execution=execution_model,
                )
            except ValueError as exc:
                logger.warning("%sBacktest skipped: %s", label_prefix, exc)
                bt_result = None

        if bt_attempted:
            if bt_result is None:
                logger.info("%sBacktest not available - insufficient data.", label_prefix)
            else:
                stats, net_series, gross_series, bt_turnover_series, period_info = bt_result
                result["bt_stats"] = stats
                result["bt_net_series"] = net_series
                result["bt_gross_series"] = gross_series
                result["bt_turnover_series"] = bt_turnover_series
                result["bt_periods"] = period_info
                mode_text = "long-only" if BACKTEST_LONG_ONLY else "long-short"
                logger.info(
                    "%sBacktest (%s, top-K, exit_mode=%s):",
                    label_prefix,
                    mode_text,
                    BACKTEST_EXIT_MODE,
                )
                logger.info("%s  periods: %s", label_prefix, stats["periods"])
                logger.info("%s  total return: %.2f%%", label_prefix, stats["total_return"] * 100)
                logger.info("%s  ann return: %.2f%%", label_prefix, stats["ann_return"] * 100)
                logger.info("%s  ann vol: %.2f%%", label_prefix, stats["ann_vol"] * 100)
                logger.info("%s  sharpe: %.2f", label_prefix, stats["sharpe"])
                logger.info("%s  max drawdown: %.2f%%", label_prefix, stats["max_drawdown"] * 100)
                if not np.isnan(stats["avg_turnover"]):
                    logger.info("%s  avg turnover: %.2f%%", label_prefix, stats["avg_turnover"] * 100)
                    logger.info(
                        "%s  avg cost drag: %.2f%%",
                        label_prefix,
                        stats["avg_cost_drag"] * 100,
                    )

                if benchmark_df is not None and not benchmark_df.empty:
                    bench_series, bench_periods = build_benchmark_series(
                        benchmark_df, PRICE_COL, period_info
                    )
                    if not bench_series.empty:
                        result["bt_benchmark_series"] = bench_series
                        bt_benchmark_stats = summarize_period_returns(
                            bench_series, bench_periods, BACKTEST_TRADING_DAYS_PER_YEAR
                        )
                        result["bt_benchmark_stats"] = bt_benchmark_stats
                        logger.info(
                            "%s  benchmark total return: %.2f%%",
                            label_prefix,
                            bt_benchmark_stats["total_return"] * 100,
                        )
                        periods_per_year = stats.get("periods_per_year", np.nan)
                        bt_active_stats, bt_active_series = summarize_active_returns(
                            net_series, bench_series, periods_per_year
                        )
                        result["bt_active_stats"] = bt_active_stats
                        result["bt_active_series"] = bt_active_series
                        if bt_active_stats and bt_active_stats.get("n", 0) > 0:
                            logger.info(
                                "%s  active total return: %.2f%%",
                                label_prefix,
                                bt_active_stats["active_total_return"] * 100,
                            )
                            if np.isfinite(bt_active_stats.get("information_ratio", np.nan)):
                                logger.info(
                                    "%s  information ratio: %.2f",
                                    label_prefix,
                                    bt_active_stats["information_ratio"],
                                )
                            if np.isfinite(bt_active_stats.get("beta", np.nan)):
                                logger.info("%s  beta: %.2f", label_prefix, bt_active_stats["beta"])
                            if np.isfinite(bt_active_stats.get("alpha", np.nan)):
                                logger.info(
                                    "%s  alpha (ann): %.2f%%",
                                    label_prefix,
                                    bt_active_stats["alpha"] * 100,
                                )

        return result

    BACKTEST_SIGNAL_DIRECTION = (
        SIGNAL_DIRECTION if BACKTEST_SIGNAL_DIRECTION_RAW is None else BACKTEST_SIGNAL_DIRECTION_RAW
    )

    eval_main = evaluate_period(
        "Test",
        model,
        test_df_full,
        test_dates,
        run_perm_test=PERM_TEST_ENABLED,
        perm_train_df=train_df,
        perm_test_df=test_df,
        allow_live_fallback=True,
    )

    ic_series = eval_main["ic_series"]
    ic_stats = eval_main["ic_stats"]
    pearson_ic_series = eval_main["pearson_ic_series"]
    pearson_ic_stats = eval_main["pearson_ic_stats"]
    error_metrics = eval_main["error_metrics"]
    hit_rate_stats = eval_main["hit_rate"]
    topk_positive_stats = eval_main["topk_positive_ratio"]
    bucket_ic_records = eval_main["bucket_ic"]
    quantile_ts = eval_main["quantile_ts"]
    quantile_mean = eval_main["quantile_mean"]
    turnover_series = eval_main["turnover_series"]
    positions_by_rebalance = eval_main["positions_by_rebalance"]
    bt_stats = eval_main["bt_stats"]
    bt_net_series = eval_main["bt_net_series"]
    bt_gross_series = eval_main["bt_gross_series"]
    bt_turnover_series = eval_main["bt_turnover_series"]
    bt_benchmark_series = eval_main["bt_benchmark_series"]
    bt_active_series = eval_main["bt_active_series"]
    bt_benchmark_stats = eval_main["bt_benchmark_stats"]
    bt_active_stats = eval_main["bt_active_stats"]
    bt_periods = eval_main["bt_periods"]
    perm_stats = eval_main["perm_stats"]

    rolling_ic_results, rolling_ic_obs_per_year = _compute_rolling_ic(
        ic_series, ROLLING_WINDOWS_MONTHS
    )
    rolling_ic_latest = {
        label: _latest_rolling_stats(frame, ["ic_mean", "ic_ir"])
        for label, frame in rolling_ic_results.items()
    }
    rolling_sharpe_results = {}
    rolling_sharpe_latest = {}
    if bt_stats is not None and not bt_net_series.empty:
        periods_per_year = bt_stats.get("periods_per_year", np.nan)
        rolling_sharpe_results = _compute_rolling_sharpe(
            bt_net_series, ROLLING_WINDOWS_MONTHS, periods_per_year
        )
        rolling_sharpe_latest = {
            label: _latest_rolling_stats(frame, ["mean", "std", "sharpe"])
            for label, frame in rolling_sharpe_results.items()
        }

    if positions_by_rebalance is not None and not positions_by_rebalance.empty:
        positions_by_rebalance = _annotate_positions_window(positions_by_rebalance)
    if positions_by_rebalance_live is not None and not positions_by_rebalance_live.empty:
        positions_by_rebalance_live = _annotate_positions_window(positions_by_rebalance_live)
    positions_by_rebalance_path: Optional[Path] = None
    positions_current_path: Optional[Path] = None
    positions_by_rebalance_live_path: Optional[Path] = None
    positions_current_live_path: Optional[Path] = None
    positions_diff_path: Optional[Path] = None
    positions_diff_live_path: Optional[Path] = None

    cv_stats_raw = None
    cv_stats = None
    if cv_scores_raw:
        cv_stats_raw = {
            "mean": float(np.nanmean(cv_scores_raw)),
            "std": float(np.nanstd(cv_scores_raw)),
            "scores": [float(score) for score in cv_scores_raw],
        }
        if cv_scores_adj is None:
            cv_scores_adj = [float(score) * SIGNAL_DIRECTION for score in cv_scores_raw]
        cv_stats = {
            "mean": float(np.nanmean(cv_scores_adj)),
            "std": float(np.nanstd(cv_scores_adj)),
            "scores": [float(score) for score in cv_scores_adj],
        }

    walk_forward_results: list[dict] = []
    if WF_ENABLED:
        try:
            wf_test_size = float(WF_TEST_SIZE)
        except (TypeError, ValueError):
            wf_test_size = TEST_SIZE
        windows = build_walk_forward_windows(
            all_dates,
            wf_test_size,
            WF_N_WINDOWS,
            WF_STEP_SIZE,
            EFFECTIVE_GAP_STEPS,
            WF_ANCHOR_END,
        )
        if not windows:
            logger.info("Walk-forward evaluation skipped: insufficient windows.")
        else:
            logger.info("Walk-forward evaluation: %s windows.", len(windows))
            for window_meta in windows:
                walk_forward_results.append(evaluate_window(window_meta))

    final_oos_eval = None
    ic_series_oos = pd.Series(dtype=float, name="ic")
    ic_stats_oos = {}
    pearson_ic_series_oos = pd.Series(dtype=float, name="ic_pearson")
    pearson_ic_stats_oos = {}
    error_metrics_oos = {}
    hit_rate_stats_oos = {}
    topk_positive_stats_oos = {}
    bucket_ic_records_oos = []
    quantile_ts_oos = pd.DataFrame()
    quantile_mean_oos = pd.Series(dtype=float)
    turnover_series_oos = pd.Series(dtype=float, name="turnover")
    positions_by_rebalance_oos = None
    bt_stats_oos = None
    bt_net_series_oos = pd.Series(dtype=float, name="net_return")
    bt_gross_series_oos = pd.Series(dtype=float, name="gross_return")
    bt_turnover_series_oos = pd.Series(dtype=float, name="turnover")
    bt_benchmark_series_oos = pd.Series(dtype=float, name="benchmark_return")
    bt_active_series_oos = pd.Series(dtype=float, name="active_return")
    bt_benchmark_stats_oos = None
    bt_active_stats_oos = None
    bt_periods_oos: list[dict] = []
    rolling_ic_oos_results: dict[str, pd.DataFrame] = {}
    rolling_ic_oos_obs_per_year = np.nan
    rolling_ic_latest_oos: dict[str, dict | None] = {}
    rolling_sharpe_oos_results: dict[str, pd.DataFrame] = {}
    rolling_sharpe_latest_oos: dict[str, dict | None] = {}
    positions_by_rebalance_oos_path: Optional[Path] = None
    positions_current_oos_path: Optional[Path] = None
    positions_diff_oos_path: Optional[Path] = None
    if FINAL_OOS_ENABLED and final_oos_dates.size > 0:
        oos_start = pd.to_datetime(final_oos_dates[0])
        oos_end = pd.to_datetime(final_oos_dates[-1])
        oos_df_full = df_full[
            (df_full["trade_date"] >= oos_start) & (df_full["trade_date"] <= oos_end)
        ].copy()
        if oos_df_full.empty:
            logger.info(
                "Final OOS evaluation skipped: no data between %s and %s.",
                oos_start.date(),
                oos_end.date(),
            )
        else:
            logger.info("Fitting final model on all in-sample data for OOS evaluation ...")
            final_model = XGBRegressor(**XGB_PARAMS)
            final_weights = build_sample_weight(df_model, SAMPLE_WEIGHT_MODE)
            if final_weights is not None:
                final_model.fit(
                    df_model[FEATURES], df_model[TARGET], sample_weight=final_weights
                )
            else:
                final_model.fit(df_model[FEATURES], df_model[TARGET])
            final_oos_eval = evaluate_period(
                "Final OOS",
                final_model,
                oos_df_full,
                final_oos_dates,
                run_perm_test=False,
                allow_live_fallback=False,
            )
            ic_series_oos = final_oos_eval["ic_series"]
            ic_stats_oos = final_oos_eval["ic_stats"]
            pearson_ic_series_oos = final_oos_eval["pearson_ic_series"]
            pearson_ic_stats_oos = final_oos_eval["pearson_ic_stats"]
            error_metrics_oos = final_oos_eval["error_metrics"]
            hit_rate_stats_oos = final_oos_eval["hit_rate"]
            topk_positive_stats_oos = final_oos_eval["topk_positive_ratio"]
            bucket_ic_records_oos = final_oos_eval["bucket_ic"]
            quantile_ts_oos = final_oos_eval["quantile_ts"]
            quantile_mean_oos = final_oos_eval["quantile_mean"]
            turnover_series_oos = final_oos_eval["turnover_series"]
            positions_by_rebalance_oos = final_oos_eval["positions_by_rebalance"]
            bt_stats_oos = final_oos_eval["bt_stats"]
            bt_net_series_oos = final_oos_eval["bt_net_series"]
            bt_gross_series_oos = final_oos_eval["bt_gross_series"]
            bt_turnover_series_oos = final_oos_eval["bt_turnover_series"]
            bt_benchmark_series_oos = final_oos_eval["bt_benchmark_series"]
            bt_active_series_oos = final_oos_eval["bt_active_series"]
            bt_benchmark_stats_oos = final_oos_eval["bt_benchmark_stats"]
            bt_active_stats_oos = final_oos_eval["bt_active_stats"]
            bt_periods_oos = final_oos_eval["bt_periods"]
            if positions_by_rebalance_oos is not None and not positions_by_rebalance_oos.empty:
                positions_by_rebalance_oos = _annotate_positions_window(positions_by_rebalance_oos)

    if final_oos_eval is not None:
        rolling_ic_oos_results, rolling_ic_oos_obs_per_year = _compute_rolling_ic(
            ic_series_oos, ROLLING_WINDOWS_MONTHS
        )
        rolling_ic_latest_oos = {
            label: _latest_rolling_stats(frame, ["ic_mean", "ic_ir"])
            for label, frame in rolling_ic_oos_results.items()
        }
        if bt_stats_oos is not None and not bt_net_series_oos.empty:
            periods_per_year_oos = bt_stats_oos.get("periods_per_year", np.nan)
            rolling_sharpe_oos_results = _compute_rolling_sharpe(
                bt_net_series_oos, ROLLING_WINDOWS_MONTHS, periods_per_year_oos
            )
            rolling_sharpe_latest_oos = {
                label: _latest_rolling_stats(frame, ["mean", "std", "sharpe"])
                for label, frame in rolling_sharpe_oos_results.items()
            }

    # Feature importance
    logger.info("Feature importance:")
    importance_df = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    for _, row in importance_df.iterrows():
        logger.info("  %-20s: %.4f", row["feature"], row["importance"])

    # Persist artifacts
    dataset_path: Optional[Path] = None
    if SAVE_ARTIFACTS:
        rolling_ic_files: dict[str, str] = {}
        rolling_sharpe_files: dict[str, str] = {}
        rolling_ic_oos_files: dict[str, str] = {}
        rolling_sharpe_oos_files: dict[str, str] = {}
        bucket_ic_path: Optional[Path] = None
        bucket_ic_oos_path: Optional[Path] = None
        if SAVE_DATASET:
            dataset_path = run_dir / "dataset.parquet"
            save_parquet(dataset.as_multiindex(), dataset_path)
        save_frame(importance_df, run_dir / "feature_importance.csv")
        save_series(ic_series, run_dir / "ic_test.csv", value_name="ic")
        save_series(pearson_ic_series, run_dir / "ic_pearson_test.csv", value_name="ic")
        if REPORT_TRAIN_IC:
            save_series(train_ic_series, run_dir / "ic_train.csv", value_name="ic")
            save_series(train_pearson_ic_series, run_dir / "ic_pearson_train.csv", value_name="ic")
        if not quantile_ts.empty:
            quantile_out = quantile_ts.reset_index()
            save_frame(quantile_out, run_dir / "quantile_returns.csv")
        save_series(turnover_series, run_dir / "turnover_eval.csv", value_name="turnover")
        if bucket_ic_records:
            bucket_ic_path = run_dir / "bucket_ic.csv"
            save_frame(pd.DataFrame(bucket_ic_records), bucket_ic_path)
        if rolling_ic_results:
            for label, frame in rolling_ic_results.items():
                if frame.empty:
                    continue
                out = frame.copy()
                out.index.name = "trade_date"
                path = run_dir / f"ic_rolling_{label}.csv"
                save_frame(out.reset_index(), path)
                rolling_ic_files[label] = str(path)
        if rolling_sharpe_results:
            for label, frame in rolling_sharpe_results.items():
                if frame.empty:
                    continue
                out = frame.copy()
                out.index.name = "trade_date"
                path = run_dir / f"backtest_rolling_sharpe_{label}.csv"
                save_frame(out.reset_index(), path)
                rolling_sharpe_files[label] = str(path)
        if final_oos_eval is not None:
            save_series(ic_series_oos, run_dir / "ic_oos.csv", value_name="ic")
            save_series(
                pearson_ic_series_oos, run_dir / "ic_pearson_oos.csv", value_name="ic"
            )
            if not quantile_ts_oos.empty:
                quantile_oos_out = quantile_ts_oos.reset_index()
                save_frame(quantile_oos_out, run_dir / "quantile_returns_oos.csv")
            save_series(
                turnover_series_oos,
                run_dir / "turnover_eval_oos.csv",
                value_name="turnover",
            )
            if bucket_ic_records_oos:
                bucket_ic_oos_path = run_dir / "bucket_ic_oos.csv"
                save_frame(pd.DataFrame(bucket_ic_records_oos), bucket_ic_oos_path)
            if rolling_ic_oos_results:
                for label, frame in rolling_ic_oos_results.items():
                    if frame.empty:
                        continue
                    out = frame.copy()
                    out.index.name = "trade_date"
                    path = run_dir / f"ic_rolling_{label}_oos.csv"
                    save_frame(out.reset_index(), path)
                    rolling_ic_oos_files[label] = str(path)
            if rolling_sharpe_oos_results:
                for label, frame in rolling_sharpe_oos_results.items():
                    if frame.empty:
                        continue
                    out = frame.copy()
                    out.index.name = "trade_date"
                    path = run_dir / f"backtest_rolling_sharpe_{label}_oos.csv"
                    save_frame(out.reset_index(), path)
                    rolling_sharpe_oos_files[label] = str(path)
        if not dropped_date_counts.empty:
            dropped_df = dropped_date_counts.rename("symbol_count").reset_index()
            save_frame(dropped_df, run_dir / "dropped_dates.csv")
        if bt_stats is not None:
            save_series(bt_net_series, run_dir / "backtest_net.csv", value_name="net_return")
            save_series(bt_gross_series, run_dir / "backtest_gross.csv", value_name="gross_return")
            save_series(bt_turnover_series, run_dir / "backtest_turnover.csv", value_name="turnover")
            if not bt_benchmark_series.empty:
                save_series(
                    bt_benchmark_series, run_dir / "backtest_benchmark.csv", value_name="benchmark_return"
                )
            if not bt_active_series.empty:
                save_series(bt_active_series, run_dir / "backtest_active.csv", value_name="active_return")
            if bt_periods:
                save_frame(pd.DataFrame(bt_periods), run_dir / "backtest_periods.csv")
        if bt_stats_oos is not None:
            save_series(bt_net_series_oos, run_dir / "backtest_net_oos.csv", value_name="net_return")
            save_series(
                bt_gross_series_oos, run_dir / "backtest_gross_oos.csv", value_name="gross_return"
            )
            save_series(
                bt_turnover_series_oos, run_dir / "backtest_turnover_oos.csv", value_name="turnover"
            )
            if not bt_benchmark_series_oos.empty:
                save_series(
                    bt_benchmark_series_oos,
                    run_dir / "backtest_benchmark_oos.csv",
                    value_name="benchmark_return",
                )
            if not bt_active_series_oos.empty:
                save_series(
                    bt_active_series_oos,
                    run_dir / "backtest_active_oos.csv",
                    value_name="active_return",
                )
            if bt_periods_oos:
                save_frame(pd.DataFrame(bt_periods_oos), run_dir / "backtest_periods_oos.csv")

        if (
            positions_by_rebalance is not None
            and not positions_by_rebalance.empty
            and (BACKTEST_ENABLED or not LIVE_ENABLED)
        ):
            positions_by_rebalance_path = run_dir / "positions_by_rebalance.csv"
            save_frame(positions_by_rebalance, positions_by_rebalance_path)
            entry_dates = pd.to_datetime(positions_by_rebalance["entry_date"], errors="coerce")
            if entry_dates.notna().any():
                latest_entry = entry_dates.max()
                positions_current = positions_by_rebalance[entry_dates == latest_entry].copy()
                if not positions_current.empty:
                    positions_current_path = run_dir / "positions_current.csv"
                    save_frame(positions_current, positions_current_path)
        if positions_by_rebalance_oos is not None and not positions_by_rebalance_oos.empty:
            positions_by_rebalance_oos_path = run_dir / "positions_by_rebalance_oos.csv"
            save_frame(positions_by_rebalance_oos, positions_by_rebalance_oos_path)
            oos_entry_dates = pd.to_datetime(
                positions_by_rebalance_oos["entry_date"], errors="coerce"
            )
            if oos_entry_dates.notna().any():
                oos_latest_entry = oos_entry_dates.max()
                positions_current_oos = positions_by_rebalance_oos[
                    oos_entry_dates == oos_latest_entry
                ].copy()
                if not positions_current_oos.empty:
                    positions_current_oos_path = run_dir / "positions_current_oos.csv"
                    save_frame(positions_current_oos, positions_current_oos_path)

        if (
            LIVE_ENABLED
            and positions_by_rebalance_live is not None
            and not positions_by_rebalance_live.empty
        ):
            positions_by_rebalance_live_path = run_dir / "positions_by_rebalance_live.csv"
            save_frame(positions_by_rebalance_live, positions_by_rebalance_live_path)
            live_entry_dates = pd.to_datetime(
                positions_by_rebalance_live["entry_date"], errors="coerce"
            )
            if live_entry_dates.notna().any():
                live_latest_entry = live_entry_dates.max()
                positions_current_live = positions_by_rebalance_live[
                    live_entry_dates == live_latest_entry
                ].copy()
                if not positions_current_live.empty:
                    positions_current_live_path = run_dir / "positions_current_live.csv"
                    save_frame(positions_current_live, positions_current_live_path)

        if (
            positions_by_rebalance is not None
            and not positions_by_rebalance.empty
            and (BACKTEST_ENABLED or not LIVE_ENABLED)
        ):
            diff_frame = _build_rebalance_diff(positions_by_rebalance)
            if not diff_frame.empty:
                positions_diff_path = run_dir / "rebalance_diff.csv"
                save_frame(diff_frame, positions_diff_path)
        if positions_by_rebalance_oos is not None and not positions_by_rebalance_oos.empty:
            diff_oos = _build_rebalance_diff(positions_by_rebalance_oos)
            if not diff_oos.empty:
                positions_diff_oos_path = run_dir / "rebalance_diff_oos.csv"
                save_frame(diff_oos, positions_diff_oos_path)

        if LIVE_ENABLED and positions_by_rebalance_live is not None and not positions_by_rebalance_live.empty:
            diff_live = _build_rebalance_diff(positions_by_rebalance_live)
            if not diff_live.empty:
                positions_diff_live_path = run_dir / "rebalance_diff_live.csv"
                save_frame(diff_live, positions_diff_live_path)

        if perm_stats and perm_stats.get("scores"):
            save_frame(
                pd.DataFrame({"ic": perm_stats["scores"]}),
                run_dir / "permutation_test.csv",
            )

        if walk_forward_results:
            save_frame(pd.DataFrame(walk_forward_results), run_dir / "walk_forward_summary.csv")

        live_positions_file = None
        live_current_file = None
        if LIVE_ENABLED:
            if positions_by_rebalance_live_path is not None:
                live_positions_file = positions_by_rebalance_live_path
            if positions_current_live_path is not None:
                live_current_file = positions_current_live_path

        summary = {
            "run": {
                "name": run_name,
                "timestamp": run_stamp,
                "config_hash": run_hash,
                "config_path": str(config_path) if config_path else None,
                "config_source": config_source,
                "output_dir": str(run_dir),
            },
            "data": {
                "market": MARKET,
                "provider": provider,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "symbols": len(symbols),
                "rows": len(df_full),
                "rows_model": len(df_model_all),
                "rows_model_in_sample": len(df_model),
                "rows_model_oos": len(df_model_oos) if FINAL_OOS_ENABLED else 0,
                "min_symbols_per_date": MIN_SYMBOLS_PER_DATE,
                "dropped_dates": int(dropped_date_counts.shape[0]),
            },
            "dataset": {
                "schema": dataset.schema.to_dict() if dataset is not None else None,
                "rows": int(len(dataset.frame)) if dataset is not None else 0,
                "file": str(dataset_path) if dataset_path else None,
                "index": [dataset.schema.date_col, dataset.schema.instrument_col]
                if dataset is not None
                else None,
            },
            "universe": {
                "mode": universe_mode_effective,
                "by_date_file": str(by_date_file) if by_date_file else None,
                "require_by_date": REQUIRE_BY_DATE,
                "drop_suspended": DROP_SUSPENDED,
                "suspended_policy": SUSPENDED_POLICY,
            },
            "label": {
                "horizon_days": LABEL_HORIZON_DAYS,
                "horizon_days_effective": label_horizon_effective,
                "horizon_mode": LABEL_HORIZON_MODE,
                "rebalance_frequency": LABEL_REBALANCE_FREQUENCY,
                "shift_days": LABEL_SHIFT_DAYS,
                "winsorize_pct": WINSORIZE_PCT,
            },
            "split": {
                "train_dates": len(train_dates),
                "test_dates": len(test_dates),
                "purge_days": purge_days,
                "embargo_days": embargo_days,
                "purge_steps": PURGE_STEPS,
                "embargo_steps": EMBARGO_STEPS,
                "rebalance_gap_days": float(rebalance_gap_days)
                if SAMPLE_ON_REBALANCE_DATES
                and rebalance_gap_days is not None
                and np.isfinite(rebalance_gap_days)
                else None,
            },
            "eval": {
                "ic": ic_stats,
                "pearson_ic": pearson_ic_stats,
                "train_ic": train_ic_stats if REPORT_TRAIN_IC else None,
                "train_ic_raw": train_ic_raw_stats if train_ic_raw_stats else None,
                "train_pearson_ic": train_pearson_ic_stats if REPORT_TRAIN_IC else None,
                "cv_ic": cv_stats,
                "cv_ic_raw": cv_stats_raw,
                "signal_direction": SIGNAL_DIRECTION,
                "signal_direction_mode": SIGNAL_DIRECTION_MODE,
                "error_metrics": error_metrics,
                "hit_rate": hit_rate_stats,
                "topk_positive_ratio": topk_positive_stats,
                "bucket_ic": bucket_ic_records,
                "bucket_ic_file": str(bucket_ic_path) if bucket_ic_path else None,
                "rolling_ic": {
                    "windows_months": ROLLING_WINDOWS_MONTHS,
                    "obs_per_year": rolling_ic_obs_per_year,
                    "latest": rolling_ic_latest,
                    "series_files": rolling_ic_files,
                },
                "quantile_mean": quantile_mean.to_dict() if not quantile_mean.empty else {},
                "long_short": float(quantile_mean.iloc[-1] - quantile_mean.iloc[0])
                if not quantile_mean.empty
                else None,
                "turnover_mean": float(turnover_series.mean()) if not turnover_series.empty else None,
                "turnover_count": int(turnover_series.shape[0]),
                "buffer_exit": EVAL_BUFFER_EXIT,
                "buffer_entry": EVAL_BUFFER_ENTRY,
                "sample_on_rebalance_dates": SAMPLE_ON_REBALANCE_DATES,
                "permutation_test": perm_stats,
            },
            "backtest": {
                "enabled": BACKTEST_ENABLED,
                "exit_mode": BACKTEST_EXIT_MODE,
                "exit_price_policy": BACKTEST_EXIT_PRICE_POLICY,
                "exit_fallback_policy": BACKTEST_EXIT_FALLBACK_POLICY,
                "buffer_exit": BACKTEST_BUFFER_EXIT,
                "buffer_entry": BACKTEST_BUFFER_ENTRY,
                "mode": "long_only" if BACKTEST_LONG_ONLY else "long_short",
                "top_k": BACKTEST_TOP_K,
                "short_k": BACKTEST_SHORT_K,
                "rebalance_frequency": BACKTEST_REBALANCE_FREQUENCY,
                "signal_direction": BACKTEST_SIGNAL_DIRECTION,
                "benchmark_symbol": benchmark_symbol,
                "transaction_cost_bps": BACKTEST_COST_BPS_REPORT,
                "execution": describe_execution_model(execution_model),
                "stats": bt_stats,
                "benchmark": bt_benchmark_stats,
                "active": bt_active_stats,
                "rolling_sharpe": {
                    "windows_months": ROLLING_WINDOWS_MONTHS,
                    "latest": rolling_sharpe_latest,
                    "series_files": rolling_sharpe_files,
                },
            },
            "final_oos": {
                "enabled": FINAL_OOS_ENABLED,
                "size": FINAL_OOS_SIZE_RAW,
                "dates": int(final_oos_len) if FINAL_OOS_ENABLED else 0,
                "start": final_oos_start.strftime("%Y%m%d") if final_oos_start else None,
                "end": final_oos_end.strftime("%Y%m%d") if final_oos_end else None,
                "ic": ic_stats_oos if final_oos_eval is not None else None,
                "pearson_ic": pearson_ic_stats_oos if final_oos_eval is not None else None,
                "error_metrics": error_metrics_oos if final_oos_eval is not None else None,
                "hit_rate": hit_rate_stats_oos if final_oos_eval is not None else None,
                "topk_positive_ratio": topk_positive_stats_oos if final_oos_eval is not None else None,
                "bucket_ic": bucket_ic_records_oos if final_oos_eval is not None else None,
                "bucket_ic_file": str(bucket_ic_oos_path) if bucket_ic_oos_path else None,
                "rolling_ic": {
                    "windows_months": ROLLING_WINDOWS_MONTHS,
                    "obs_per_year": rolling_ic_oos_obs_per_year,
                    "latest": rolling_ic_latest_oos,
                    "series_files": rolling_ic_oos_files,
                }
                if final_oos_eval is not None
                else None,
                "quantile_mean": quantile_mean_oos.to_dict()
                if final_oos_eval is not None and not quantile_mean_oos.empty
                else {},
                "long_short": float(quantile_mean_oos.iloc[-1] - quantile_mean_oos.iloc[0])
                if final_oos_eval is not None and not quantile_mean_oos.empty
                else None,
                "turnover_mean": float(turnover_series_oos.mean())
                if final_oos_eval is not None and not turnover_series_oos.empty
                else None,
                "turnover_count": int(turnover_series_oos.shape[0])
                if final_oos_eval is not None
                else 0,
                "backtest": {
                    "stats": bt_stats_oos,
                    "benchmark": bt_benchmark_stats_oos,
                    "active": bt_active_stats_oos,
                    "rolling_sharpe": {
                        "windows_months": ROLLING_WINDOWS_MONTHS,
                        "latest": rolling_sharpe_latest_oos,
                        "series_files": rolling_sharpe_oos_files,
                    },
                }
                if final_oos_eval is not None
                else None,
                "positions": {
                    "by_rebalance_file": str(positions_by_rebalance_oos_path)
                    if positions_by_rebalance_oos_path
                    else None,
                    "current_file": str(positions_current_oos_path)
                    if positions_current_oos_path
                    else None,
                    "diff_file": str(positions_diff_oos_path) if positions_diff_oos_path else None,
                }
                if final_oos_eval is not None
                else None,
            },
            "positions": {
                "by_rebalance_file": str(positions_by_rebalance_path)
                if positions_by_rebalance_path
                else None,
                "current_file": str(positions_current_path) if positions_current_path else None,
                "diff_file": str(positions_diff_path) if positions_diff_path else None,
                "shift_days": LABEL_SHIFT_DAYS,
                "buffer_exit": BACKTEST_BUFFER_EXIT,
                "buffer_entry": BACKTEST_BUFFER_ENTRY,
                "window_fields": {
                    "signal_asof": "signal_asof",
                    "entry_date": "entry_date",
                    "next_entry_date": "next_entry_date",
                    "holding_window": "holding_window",
                },
            },
            "live": {
                "enabled": LIVE_ENABLED,
                "as_of": live_as_of.strftime("%Y%m%d") if LIVE_ENABLED and live_as_of else None,
                "train_mode": LIVE_TRAIN_MODE if LIVE_ENABLED else None,
                "positions_file": str(live_positions_file) if live_positions_file else None,
                "current_file": str(live_current_file) if live_current_file else None,
                "diff_file": str(positions_diff_live_path) if positions_diff_live_path else None,
            },
            "fundamentals": {
                "enabled": FUNDAMENTALS_ENABLED,
                "source": FUNDAMENTALS_SOURCE if FUNDAMENTALS_ENABLED else None,
                "file": str(FUNDAMENTALS_FILE) if FUNDAMENTALS_FILE else None,
                "features": FUNDAMENTALS_FEATURES,
                "log_market_cap": FUNDAMENTALS_LOG_MCAP,
                "market_cap_col": FUNDAMENTALS_MCAP_COL,
            },
            "walk_forward": {
                "enabled": WF_ENABLED,
                "n_windows": WF_N_WINDOWS,
                "test_size": WF_TEST_SIZE,
                "step_size": WF_STEP_SIZE,
                "anchor_end": WF_ANCHOR_END,
                "results": walk_forward_results,
            },
        }
        save_json(summary, run_dir / "summary.json")
        with (run_dir / "config.used.yml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        if LIVE_ENABLED:
            latest_payload = {
                "run_dir": str(run_dir),
                "run_name": run_name,
                "timestamp": run_stamp,
                "config_hash": run_hash,
                "summary_file": str(run_dir / "summary.json"),
                "as_of": summary.get("live", {}).get("as_of"),
                "positions_file": summary.get("live", {}).get("positions_file"),
                "current_file": summary.get("live", {}).get("current_file"),
                "diff_file": summary.get("live", {}).get("diff_file"),
            }
            save_json(latest_payload, run_dir.parent / "latest.json")

    # Optional: save the model
    # from joblib import dump; dump(model, "xgb_factor_model.joblib")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional XGBoost pipeline")
    parser.add_argument(
        "--config",
        help="Path to YAML config or built-in name (default/cn/hk/us).",
    )
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
