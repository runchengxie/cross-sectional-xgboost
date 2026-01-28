"""main.py - Cross-sectional factor mining with XGBoost regression (multi-market).
Usage:
    $ python main.py --config config/config.yml
    # provider-specific auth may be required (e.g. TUSHARE_TOKEN/TUSHARE_TOKEN_2)
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
# Workaround for pandas_ta NaN import issue
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
import tushare as ts
import pyarrow  # ensures parquet support
from dotenv import load_dotenv
import yaml
from xgboost import XGBRegressor
import warnings

from data_providers import fetch_daily, load_basic, normalize_market, resolve_provider
from csxgb.metrics import daily_ic_series, summarize_ic, quantile_returns, estimate_turnover
from csxgb.transform import apply_cross_sectional_transform
from csxgb.split import time_series_cv_ic
from csxgb.backtest import backtest_topk

warnings.filterwarnings("ignore")

logger = logging.getLogger("csxgb")

# -----------------------------------------------------------------------------
# rqdatac workaround
# -----------------------------------------------------------------------------
def _patch_rqdatac_adjust_price_readonly() -> None:
    """Ensure rqdatac's in-place adjust doesn't choke on read-only arrays."""
    try:
        import rqdatac.services.detail.adjust_price as adjust_price
    except Exception as exc:  # pragma: no cover - defensive import
        logger.debug("rqdatac adjust_price import failed: %s", exc)
        return
    if getattr(adjust_price, "_csxgb_readonly_patch", False):
        return

    original = adjust_price.adjust_price_multi_df

    def wrapped(df, order_book_ids, how, obid_slice_map, market):
        r_map_fields = {
            f: i
            for i, f in enumerate(df.columns)
            if f in adjust_price.FIELDS_NEED_TO_ADJUST
        }
        if not r_map_fields:
            return
        pre = how in ("pre", "pre_volume")
        volume_adjust_by_ex_factor = how in ("pre_volume", "post_volume")
        ex_factors = adjust_price.get_ex_factor_for(order_book_ids, market)
        volume_adjust_factors = {}
        if "volume" in r_map_fields:
            if not volume_adjust_by_ex_factor:
                volume_adjust_factors = adjust_price.get_split_factor_for(order_book_ids, market)
            else:
                volume_adjust_factors = ex_factors

        data = df.to_numpy(copy=True)
        try:
            data.setflags(write=True)
        except Exception:
            pass
        timestamps_level = df.index.get_level_values(1)
        for order_book_id, slice_ in obid_slice_map.items():
            if order_book_id not in order_book_ids:
                continue
            timestamps = timestamps_level[slice_]

            def calculate_factor(factors_map, order_book_id):
                factors = factors_map.get(order_book_id, None)
                if factors is not None:
                    factor = np.take(
                        factors.values,
                        factors.index.searchsorted(timestamps, side="right") - 1,
                    )
                    if pre:
                        factor /= factors.iloc[-1]
                    return factor

            factor = calculate_factor(ex_factors, order_book_id)
            if factor is None:
                continue

            if not volume_adjust_by_ex_factor:
                factor_volume = calculate_factor(volume_adjust_factors, order_book_id)
            else:
                factor_volume = factor

            for f, j in r_map_fields.items():
                if f in adjust_price.PRICE_FIELDS:
                    data[slice_, j] *= factor
                elif factor_volume is not None:
                    data[slice_, j] *= 1 / factor_volume

        df.iloc[:, :] = data

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    adjust_price._csxgb_original_adjust_price_multi_df = original
    adjust_price.adjust_price_multi_df = wrapped
    adjust_price._csxgb_readonly_patch = True
    logger.warning(
        "Applied rqdatac read-only adjust_price patch (DataFrame copy on demand)."
    )

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        sys.exit("Config root must be a mapping.")
    return cfg


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


def save_series(series: pd.Series, path: Path, value_name: Optional[str] = None) -> None:
    if series is None or series.empty:
        return
    name = value_name or series.name or "value"
    out = series.rename(name).reset_index()
    out.to_csv(path, index=False)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return
    frame.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=str)


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


parser = argparse.ArgumentParser(description="Cross-sectional XGBoost pipeline")
parser.add_argument("--config", default="config/config.yml", help="Path to YAML config")
args = parser.parse_args()

config = load_config(Path(args.config))
setup_logging(config)

data_cfg = config.get("data", {})
MARKET = normalize_market(config.get("market") or data_cfg.get("market"))
universe_cfg = config.get("universe", {})
label_cfg = config.get("label", {})
features_cfg = config.get("features", {})
model_cfg = config.get("model", {})
eval_cfg = config.get("eval", {})
backtest_cfg = config.get("backtest", {})

load_dotenv()
provider = resolve_provider(data_cfg)
data_client = None
tushare_tokens: list[str] = []
tushare_token_idx = 0
if provider == "tushare":
    raw_tokens = [
        os.getenv("TUSHARE_TOKEN"),
        os.getenv("TUSHARE_TOKEN_2"),
        os.getenv("TUSHARE_API_KEY"),  # legacy alias
    ]
    for token in raw_tokens:
        if token and token not in tushare_tokens:
            tushare_tokens.append(token)
    if not tushare_tokens:
        sys.exit(
            "Please set TUSHARE_TOKEN (or TUSHARE_TOKEN_2 / legacy TUSHARE_API_KEY) first."
        )

    def make_tushare_client(token: str):
        try:
            return ts.pro_api(token)
        except TypeError:
            ts.set_token(token)
            return ts.pro_api()

    data_client = make_tushare_client(tushare_tokens[tushare_token_idx])
elif provider == "rqdata":
    try:
        import rqdatac
    except ImportError as exc:
        sys.exit(f"rqdatac is required for provider='rqdata' ({exc}).")
    rq_cfg = data_cfg.get("rqdata") or {}
    init_kwargs = {}
    if isinstance(rq_cfg, dict) and isinstance(rq_cfg.get("init"), dict):
        init_kwargs.update(rq_cfg.get("init"))
    # Allow .env / environment-based credentials as a fallback.
    env_username = os.getenv("RQDATA_USERNAME") or os.getenv("RQDATA_USER")
    env_password = os.getenv("RQDATA_PASSWORD")
    if env_username and "username" not in init_kwargs:
        init_kwargs["username"] = env_username
    if env_password and "password" not in init_kwargs:
        init_kwargs["password"] = env_password
    try:
        rqdatac.init(**init_kwargs)
    except Exception as exc:
        sys.exit(f"rqdatac.init failed: {exc}")
    _patch_rqdatac_adjust_price_readonly()
    data_client = rqdatac
elif provider == "eodhd":
    eod_cfg = data_cfg.get("eodhd") or {}
    api_token = (
        (eod_cfg.get("api_token") if isinstance(eod_cfg, dict) else None)
        or os.getenv("EODHD_API_TOKEN")
        or os.getenv("EODHD_API_KEY")
    )
    if not api_token:
        sys.exit("Please set EODHD_API_TOKEN (or data.eodhd.api_token) first.")
    data_client = {"api_token": api_token}
    if isinstance(eod_cfg, dict):
        if eod_cfg.get("base_url"):
            data_client["base_url"] = eod_cfg.get("base_url")
        if eod_cfg.get("exchange"):
            data_client["exchange"] = eod_cfg.get("exchange")
        if eod_cfg.get("timeout"):
            data_client["timeout"] = eod_cfg.get("timeout")
else:
    sys.exit(f"Unsupported data.provider '{provider}'.")

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

symbols = normalize_symbol_list(universe_cfg.get("symbols"))
symbols_file = universe_cfg.get("symbols_file")
by_date_file = universe_cfg.get("by_date_file")
universe_by_date = None

if not symbols and symbols_file:
    symbols = load_symbols_file(Path(symbols_file))

if by_date_file:
    universe_by_date = load_universe_by_date(Path(by_date_file), MARKET)
    symbols_from_universe = sorted(universe_by_date["ts_code"].unique().tolist())
    if symbols:
        symbols = sorted(set(symbols) | set(symbols_from_universe))
    else:
        symbols = symbols_from_universe

if not symbols:
    symbols = DEFAULT_SYMBOLS

if not symbols:
    sys.exit("No symbols configured.")

end_date_cfg = data_cfg.get("end_date", "today")
if not end_date_cfg or str(end_date_cfg).lower() in {"today", "now"}:
    end_date = datetime.now()
else:
    end_date = datetime.strptime(str(end_date_cfg), "%Y%m%d")

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
SIGNAL_DIRECTION = float(eval_cfg.get("signal_direction", 1.0))
if SIGNAL_DIRECTION == 0:
    sys.exit("eval.signal_direction cannot be 0.")
EMBARGO_DAYS = eval_cfg.get("embargo_days")
EMBARGO_DAYS = int(EMBARGO_DAYS) if EMBARGO_DAYS is not None else 0
PURGE_DAYS = eval_cfg.get("purge_days")
if PURGE_DAYS is None:
    PURGE_DAYS = LABEL_HORIZON_DAYS + LABEL_SHIFT_DAYS
PURGE_DAYS = int(PURGE_DAYS)
EFFECTIVE_GAP_DAYS = max(EMBARGO_DAYS, PURGE_DAYS)
REPORT_TRAIN_IC = bool(eval_cfg.get("report_train_ic", True))
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

SAVE_ARTIFACTS = bool(eval_cfg.get("save_artifacts", True))
OUTPUT_DIR = eval_cfg.get("output_dir", "out/runs")
RUN_NAME = eval_cfg.get("run_name")

MIN_SYMBOLS_PER_DATE = int(universe_cfg.get("min_symbols_per_date", N_QUANTILES))
MIN_LISTED_DAYS = int(universe_cfg.get("min_listed_days", 0))
MIN_TURNOVER = float(universe_cfg.get("min_turnover", 0))
DROP_ST = bool(universe_cfg.get("drop_st", False))
DROP_SUSPENDED = bool(universe_cfg.get("drop_suspended", True))
if MIN_SYMBOLS_PER_DATE < N_QUANTILES:
    MIN_SYMBOLS_PER_DATE = N_QUANTILES

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

BACKTEST_ENABLED = bool(backtest_cfg.get("enabled", True))
BACKTEST_TOP_K = int(backtest_cfg.get("top_k", TOP_K))
BACKTEST_REBALANCE_FREQUENCY = backtest_cfg.get("rebalance_frequency", REBALANCE_FREQUENCY)
BACKTEST_COST_BPS = float(backtest_cfg.get("transaction_cost_bps", TRANSACTION_COST_BPS))
BACKTEST_TRADING_DAYS_PER_YEAR = int(backtest_cfg.get("trading_days_per_year", 252))
BACKTEST_BENCHMARK = backtest_cfg.get("benchmark_symbol")
BACKTEST_LONG_ONLY = bool(backtest_cfg.get("long_only", True))
BACKTEST_SIGNAL_DIRECTION = float(backtest_cfg.get("signal_direction", SIGNAL_DIRECTION))
if BACKTEST_SIGNAL_DIRECTION == 0:
    sys.exit("backtest.signal_direction cannot be 0.")
BACKTEST_EXIT_MODE = str(backtest_cfg.get("exit_mode", "rebalance")).strip().lower()
if BACKTEST_EXIT_MODE not in {"rebalance", "label_horizon"}:
    sys.exit("backtest.exit_mode must be one of: rebalance, label_horizon.")
BACKTEST_EXIT_HORIZON_DAYS = backtest_cfg.get("exit_horizon_days")
if BACKTEST_EXIT_MODE == "label_horizon":
    if BACKTEST_EXIT_HORIZON_DAYS is None:
        BACKTEST_EXIT_HORIZON_DAYS = LABEL_HORIZON_DAYS
    BACKTEST_EXIT_HORIZON_DAYS = int(BACKTEST_EXIT_HORIZON_DAYS)

run_name = str(RUN_NAME or Path(args.config).stem)
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_hash = config_hash(config)
run_dir = Path(OUTPUT_DIR) / f"{run_name}_{run_stamp}_{run_hash}"
if SAVE_ARTIFACTS:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts will be saved to %s", run_dir)
# -----------------------------------------------------------------------------
# 2. Data download
# -----------------------------------------------------------------------------
CACHE_DIR = Path(data_cfg.get("cache_dir", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

retry_cfg = data_cfg.get("retry") if isinstance(data_cfg, dict) else None
retry_cfg = retry_cfg if isinstance(retry_cfg, dict) else {}
MAX_ATTEMPTS = int(retry_cfg.get("max_attempts", 1))
MAX_ATTEMPTS = max(1, MAX_ATTEMPTS)
BACKOFF_SECONDS = float(retry_cfg.get("backoff_seconds", 0.5))
MAX_BACKOFF_SECONDS = float(retry_cfg.get("max_backoff_seconds", 5.0))
ROTATE_TOKENS = bool(retry_cfg.get("rotate_tokens", True))

benchmark_symbol = str(BACKTEST_BENCHMARK).strip() if BACKTEST_BENCHMARK else None
symbols_for_data = symbols[:]
if benchmark_symbol and benchmark_symbol not in symbols_for_data:
    symbols_for_data.append(benchmark_symbol)

frames = []
def fetch_daily_with_retry(symbol: str) -> pd.DataFrame:
    global data_client, tushare_token_idx
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetch_daily(
                MARKET,
                symbol,
                START_DATE,
                END_DATE,
                CACHE_DIR,
                data_client,
                data_cfg,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Daily data load failed for %s (attempt %s/%s): %s",
                symbol,
                attempt,
                MAX_ATTEMPTS,
                exc,
                exc_info=True,
            )
            if provider == "tushare" and ROTATE_TOKENS and len(tushare_tokens) > 1:
                tushare_token_idx = (tushare_token_idx + 1) % len(tushare_tokens)
                data_client = make_tushare_client(tushare_tokens[tushare_token_idx])
                logger.info("Switched Tushare token to index %s.", tushare_token_idx)
            if attempt < MAX_ATTEMPTS:
                sleep_for = min(BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                time.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()

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
        basic_df = load_basic(MARKET, CACHE_DIR, data_client, data_cfg, symbols_for_data)
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

if DROP_SUSPENDED:
    if "amount" in df.columns:
        df = df[(df["vol"] > 0) & (df["amount"] > 0)].copy()
    else:
        df = df[df["vol"] > 0].copy()

if MIN_TURNOVER > 0 and "amount" in df.columns:
    df = df[df["amount"] >= MIN_TURNOVER].copy()

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

    entry_price = group[PRICE_COL].shift(-LABEL_SHIFT_DAYS)
    exit_price = group[PRICE_COL].shift(-(LABEL_HORIZON_DAYS + LABEL_SHIFT_DAYS))
    group[TARGET] = exit_price / entry_price - 1.0

    return group


df = df.groupby("ts_code", group_keys=False).apply(add_features)

missing_features = [feat for feat in FEATURES if feat not in df.columns]
if missing_features:
    sys.exit(f"Missing features after engineering: {missing_features}")

# Keep only the necessary columns & drop NaNs from rolling calcs / future label
cols = ["trade_date", "ts_code", PRICE_COL] + FEATURES + [TARGET]
cols = list(dict.fromkeys(cols))
df = df[cols].dropna().reset_index(drop=True)

if universe_by_date is not None:
    before_rows = len(df)
    df = apply_universe_by_date(df, universe_by_date)
    after_rows = len(df)
    logger.info("Applied universe-by-date filter: %s -> %s rows", before_rows, after_rows)
    if df.empty:
        sys.exit("Universe-by-date filter removed all rows.")

if WINSORIZE_PCT:
    def _winsorize(group: pd.DataFrame) -> pd.DataFrame:
        lower = group[TARGET].quantile(WINSORIZE_PCT)
        upper = group[TARGET].quantile(1 - WINSORIZE_PCT)
        group[TARGET] = group[TARGET].clip(lower, upper)
        return group

    df = df.groupby("trade_date", group_keys=False).apply(_winsorize)

# Drop dates with too few symbols for evaluation
date_counts = df.groupby("trade_date")["ts_code"].nunique()
valid_dates = date_counts[date_counts >= MIN_SYMBOLS_PER_DATE].index
dropped_date_counts = date_counts[date_counts < MIN_SYMBOLS_PER_DATE].sort_index()
if len(valid_dates) != len(date_counts):
    df = df[df["trade_date"].isin(valid_dates)].copy()
if not dropped_date_counts.empty:
    logger.info(
        "Dropped %s dates with < %s symbols (min=%s, max=%s).",
        len(dropped_date_counts),
        MIN_SYMBOLS_PER_DATE,
        int(dropped_date_counts.min()),
        int(dropped_date_counts.max()),
    )

if CS_METHOD != "none":
    df = apply_cross_sectional_transform(df, FEATURES, CS_METHOD, CS_WINSORIZE_PCT)

# -----------------------------------------------------------------------------
# 4. Train-test split (time-series by date)
# -----------------------------------------------------------------------------
logger.info("Splitting train/test by date ...")
all_dates = np.array(sorted(df["trade_date"].unique()))
if len(all_dates) < 10:
    sys.exit("Not enough dates for a meaningful split.")

split_idx = int(len(all_dates) * (1 - TEST_SIZE))
train_end = split_idx
if EFFECTIVE_GAP_DAYS > 0:
    train_end = max(0, split_idx - EFFECTIVE_GAP_DAYS)
train_dates = all_dates[:train_end]
test_dates = all_dates[split_idx:]

train_df = df[df["trade_date"].isin(train_dates)].copy()
test_df = df[df["trade_date"].isin(test_dates)].copy()

if train_df.empty or test_df.empty:
    sys.exit("Not enough dates for train/test after embargo.")

logger.info(
    "Train/test split: train_dates=%s, test_dates=%s, purge_days=%s, embargo_days=%s.",
    len(train_dates),
    len(test_dates),
    PURGE_DAYS,
    EMBARGO_DAYS,
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
        perm_model.fit(perm_train[FEATURES], perm_train[TARGET])

        perm_test = test_data.copy()
        perm_test["pred"] = perm_model.predict(perm_test[FEATURES])
        if signal_direction != 1.0:
            perm_test["pred"] = perm_test["pred"] * signal_direction

        ic_values = daily_ic_series(perm_test, TARGET, "pred")
        scores.append(float(ic_values.mean()) if not ic_values.empty else np.nan)
    return scores


cv_scores = time_series_cv_ic(
    train_df,
    FEATURES,
    TARGET,
    N_SPLITS,
    EMBARGO_DAYS,
    PURGE_DAYS,
    XGB_PARAMS,
    SIGNAL_DIRECTION,
)
if cv_scores:
    logger.info("CV IC: mean=%.4f, std=%.4f", np.nanmean(cv_scores), np.nanstd(cv_scores))
    logger.info("CV fold ICs: %s", [f"{s:.4f}" for s in cv_scores])
else:
    logger.info("CV IC not available - insufficient data after embargo/purge.")

# -----------------------------------------------------------------------------
# 6. Fit final model
# -----------------------------------------------------------------------------
logger.info("Fitting XGBoost regressor ...")
model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 7. Evaluation (cross-sectional factor style)
# -----------------------------------------------------------------------------
logger.info("Evaluating model on train/test sets ...")

train_eval_df = train_df.copy()
train_eval_df["pred"] = model.predict(train_eval_df[FEATURES])
train_signal_col = "pred"
if SIGNAL_DIRECTION != 1.0:
    train_eval_df["signal"] = train_eval_df["pred"] * SIGNAL_DIRECTION
    train_signal_col = "signal"
train_ic_series = pd.Series(dtype=float, name="ic")
train_ic_stats = {}
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

test_df["pred"] = model.predict(test_df[FEATURES])
signal_col = "pred"
if SIGNAL_DIRECTION != 1.0:
    test_df["signal"] = test_df["pred"] * SIGNAL_DIRECTION
    signal_col = "signal"
    logger.info("Signal direction applied to ranking: %s", SIGNAL_DIRECTION)

# Daily IC
ic_series = daily_ic_series(test_df, TARGET, signal_col)
ic_stats = summarize_ic(ic_series)
logger.info(
    "Daily IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
    ic_stats["mean"],
    ic_stats["std"],
    ic_stats["ir"],
    ic_stats["t_stat"],
    ic_stats["p_value"],
    ic_stats["n"],
)

if PERM_TEST_ENABLED:
    logger.info("Permutation test (shuffle train labels within date) ...")
    perm_scores = permutation_test_ic(
        train_df,
        test_df,
        PERM_TEST_RUNS,
        PERM_TEST_SEED,
        SIGNAL_DIRECTION,
    )
    if perm_scores:
        perm_mean = np.nanmean(perm_scores)
        perm_std = np.nanstd(perm_scores)
        logger.info(
            "Permutation IC: mean=%.4f, std=%.4f, runs=%s",
            perm_mean,
            perm_std,
            len(perm_scores),
        )
        logger.info("Permutation ICs: %s", [f"{s:.4f}" for s in perm_scores])

# Quantile returns on rebalance dates

def get_rebalance_dates(dates, freq: str) -> list:
    if not freq or freq.upper() == "D":
        return list(dates)
    date_df = pd.DataFrame({"date": pd.to_datetime(dates)})
    date_df["period"] = date_df["date"].dt.to_period(freq)
    rebalance_dates = (
        date_df.groupby("period")["date"].max().sort_values().tolist()
    )
    return rebalance_dates


def estimate_rebalance_gap(trade_dates: list[pd.Timestamp], rebalance_dates: list[pd.Timestamp]) -> float:
    if len(rebalance_dates) < 2:
        return np.nan
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    gaps = []
    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        if start in date_to_idx and end in date_to_idx:
            gaps.append(date_to_idx[end] - date_to_idx[start])
    if not gaps:
        return np.nan
    return float(np.median(gaps))


trade_dates_sorted = sorted(test_df["trade_date"].unique())
rebalance_dates = get_rebalance_dates(trade_dates_sorted, REBALANCE_FREQUENCY)
rebalance_gap = estimate_rebalance_gap(trade_dates_sorted, rebalance_dates)
if BACKTEST_EXIT_MODE == "rebalance" and np.isfinite(rebalance_gap):
    gap_diff = abs(rebalance_gap - LABEL_HORIZON_DAYS)
    if gap_diff >= max(3.0, rebalance_gap * 0.25):
        logger.warning(
            "Label horizon (%s days) differs from rebalance gap (median %.1f days).",
            LABEL_HORIZON_DAYS,
            rebalance_gap,
        )
eval_df = test_df[test_df["trade_date"].isin(rebalance_dates)].copy()

quantile_ts = quantile_returns(eval_df, signal_col, TARGET, N_QUANTILES)
quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
if not quantile_mean.empty:
    for q_idx, value in quantile_mean.items():
        logger.info("Q%s mean return: %.4f%%", int(q_idx) + 1, value * 100)
    long_short = quantile_mean.iloc[-1] - quantile_mean.iloc[0]
    logger.info("Long-short (Q%s-Q1): %.4f%%", N_QUANTILES, long_short * 100)
else:
    logger.info("Quantile returns not available - insufficient symbols per date.")

# Turnover estimate for top-K portfolio
k = min(TOP_K, eval_df["ts_code"].nunique())
turnover_series = estimate_turnover(eval_df, signal_col, k, rebalance_dates)
if not turnover_series.empty:
    turnover = turnover_series.mean()
    cost_drag = 2 * (TRANSACTION_COST_BPS / 10000.0) * turnover
    logger.info("Top-%s turnover per rebalance: %.2f%% (n=%s)", k, turnover * 100, len(turnover_series))
    logger.info(
        "Approx cost drag per rebalance: %.2f%% at %s bps per side",
        cost_drag * 100,
        TRANSACTION_COST_BPS,
    )

bt_stats = None
bt_net_series = pd.Series(dtype=float, name="net_return")
bt_gross_series = pd.Series(dtype=float, name="gross_return")
bt_turnover_series = pd.Series(dtype=float, name="turnover")
bt_periods: list[dict] = []


if BACKTEST_ENABLED:
    if not BACKTEST_LONG_ONLY:
        logger.info("Backtest only supports long-only at the moment; set backtest.long_only: true.")
    else:
        bt_rebalance = get_rebalance_dates(sorted(test_df["trade_date"].unique()), BACKTEST_REBALANCE_FREQUENCY)
        bt_pred_col = signal_col
        if BACKTEST_SIGNAL_DIRECTION != SIGNAL_DIRECTION:
            test_df["signal_bt"] = test_df["pred"] * BACKTEST_SIGNAL_DIRECTION
            bt_pred_col = "signal_bt"
        bt_result = backtest_topk(
            test_df,
            pred_col=bt_pred_col,
            price_col=PRICE_COL,
            rebalance_dates=bt_rebalance,
            top_k=BACKTEST_TOP_K,
            shift_days=LABEL_SHIFT_DAYS,
            cost_bps=BACKTEST_COST_BPS,
            trading_days_per_year=BACKTEST_TRADING_DAYS_PER_YEAR,
            exit_mode=BACKTEST_EXIT_MODE,
            exit_horizon_days=BACKTEST_EXIT_HORIZON_DAYS,
        )

        if bt_result is None:
            logger.info("Backtest not available - insufficient data.")
        else:
            stats, net_series, gross_series, bt_turnover_series, period_info = bt_result
            bt_stats = stats
            bt_net_series = net_series
            bt_gross_series = gross_series
            bt_periods = period_info
            logger.info("Backtest (long-only, top-K, exit_mode=%s):", BACKTEST_EXIT_MODE)
            logger.info("  periods: %s", stats["periods"])
            logger.info("  total return: %.2f%%", stats["total_return"] * 100)
            logger.info("  ann return: %.2f%%", stats["ann_return"] * 100)
            logger.info("  ann vol: %.2f%%", stats["ann_vol"] * 100)
            logger.info("  sharpe: %.2f", stats["sharpe"])
            logger.info("  max drawdown: %.2f%%", stats["max_drawdown"] * 100)
            if not np.isnan(stats["avg_turnover"]):
                logger.info("  avg turnover: %.2f%%", stats["avg_turnover"] * 100)
                logger.info("  avg cost drag: %.2f%%", stats["avg_cost_drag"] * 100)

            if benchmark_df is not None and not benchmark_df.empty:
                bench_prices = benchmark_df.set_index("trade_date")[PRICE_COL]
                bench_returns = []
                bench_index = []
                for info in period_info:
                    entry_date = info["entry_date"]
                    exit_date = info["exit_date"]
                    if entry_date not in bench_prices.index or exit_date not in bench_prices.index:
                        continue
                    bench_returns.append(bench_prices.loc[exit_date] / bench_prices.loc[entry_date] - 1.0)
                    bench_index.append(exit_date)
                if bench_returns:
                    bench_series = pd.Series(bench_returns, index=bench_index, name="benchmark_return")
                    bench_nav = (1 + bench_series).cumprod()
                    bench_total = bench_nav.iloc[-1] - 1.0
                    logger.info("  benchmark total return: %.2f%%", bench_total * 100)

# Feature importance
logger.info("Feature importance:")
importance_df = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
for _, row in importance_df.iterrows():
    logger.info("  %-20s: %.4f", row["feature"], row["importance"])

# Persist artifacts
if SAVE_ARTIFACTS:
    save_frame(importance_df, run_dir / "feature_importance.csv")
    save_series(ic_series, run_dir / "ic_test.csv", value_name="ic")
    if REPORT_TRAIN_IC:
        save_series(train_ic_series, run_dir / "ic_train.csv", value_name="ic")
    if not quantile_ts.empty:
        quantile_out = quantile_ts.reset_index()
        quantile_out.to_csv(run_dir / "quantile_returns.csv", index=False)
    save_series(turnover_series, run_dir / "turnover_eval.csv", value_name="turnover")
    if not dropped_date_counts.empty:
        dropped_date_counts.rename("symbol_count").reset_index().to_csv(
            run_dir / "dropped_dates.csv", index=False
        )
    if bt_stats is not None:
        save_series(bt_net_series, run_dir / "backtest_net.csv", value_name="net_return")
        save_series(bt_gross_series, run_dir / "backtest_gross.csv", value_name="gross_return")
        save_series(bt_turnover_series, run_dir / "backtest_turnover.csv", value_name="turnover")
        if bt_periods:
            pd.DataFrame(bt_periods).to_csv(run_dir / "backtest_periods.csv", index=False)

    summary = {
        "run": {
            "name": run_name,
            "timestamp": run_stamp,
            "config_hash": run_hash,
            "config_path": str(args.config),
            "output_dir": str(run_dir),
        },
        "data": {
            "market": MARKET,
            "provider": provider,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "symbols": len(symbols),
            "rows": len(df),
            "min_symbols_per_date": MIN_SYMBOLS_PER_DATE,
            "dropped_dates": int(dropped_date_counts.shape[0]),
        },
        "label": {
            "horizon_days": LABEL_HORIZON_DAYS,
            "shift_days": LABEL_SHIFT_DAYS,
            "winsorize_pct": WINSORIZE_PCT,
        },
        "split": {
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "purge_days": PURGE_DAYS,
            "embargo_days": EMBARGO_DAYS,
        },
        "eval": {
            "ic": ic_stats,
            "train_ic": train_ic_stats if REPORT_TRAIN_IC else None,
            "quantile_mean": quantile_mean.to_dict() if not quantile_mean.empty else {},
            "long_short": float(quantile_mean.iloc[-1] - quantile_mean.iloc[0])
            if not quantile_mean.empty
            else None,
            "turnover_mean": float(turnover_series.mean()) if not turnover_series.empty else None,
            "turnover_count": int(turnover_series.shape[0]),
        },
        "backtest": {
            "enabled": BACKTEST_ENABLED,
            "exit_mode": BACKTEST_EXIT_MODE,
            "stats": bt_stats,
        },
    }
    save_json(summary, run_dir / "summary.json")
    with (run_dir / "config.used.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

# Optional: save the model
# from joblib import dump; dump(model, "xgb_factor_model.joblib")
