"""main.py - Cross-sectional factor mining with XGBoost regression (multi-market).
Usage:
    $ python main.py --config config/config.yml
    # provider-specific auth may be required (e.g. TUSHARE_API_KEY/TUSHARE_TOKEN)
"""
import argparse
import os
import sys
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
from sklearn.model_selection import TimeSeriesSplit
import warnings

from data_providers import fetch_daily, load_basic, normalize_market, resolve_provider

warnings.filterwarnings("ignore")

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
if provider == "tushare":
    TOKEN = os.getenv("TUSHARE_API_KEY") or os.getenv("TUSHARE_TOKEN")
    if not TOKEN:
        sys.exit("Please set the TUSHARE_API_KEY or TUSHARE_TOKEN environment variable first.")
    ts.set_token(TOKEN)
    data_client = ts.pro_api()
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
    data_client = rqdatac
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

symbols = normalize_symbol_list(universe_cfg.get("symbols", DEFAULT_SYMBOLS))
symbols_file = universe_cfg.get("symbols_file")
if not symbols and symbols_file:
    symbols = load_symbols_file(Path(symbols_file))
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
if EMBARGO_DAYS is None:
    EMBARGO_DAYS = LABEL_HORIZON_DAYS + LABEL_SHIFT_DAYS
EMBARGO_DAYS = int(EMBARGO_DAYS)
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
# -----------------------------------------------------------------------------
# 2. Data download
# -----------------------------------------------------------------------------
CACHE_DIR = Path(data_cfg.get("cache_dir", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

benchmark_symbol = str(BACKTEST_BENCHMARK).strip() if BACKTEST_BENCHMARK else None
symbols_for_data = symbols[:]
if benchmark_symbol and benchmark_symbol not in symbols_for_data:
    symbols_for_data.append(benchmark_symbol)

frames = []
for symbol in symbols_for_data:
    print(f"Fetching daily data for {symbol} ({MARKET}) ...")
    try:
        data = fetch_daily(
            MARKET,
            symbol,
            START_DATE,
            END_DATE,
            CACHE_DIR,
            data_client,
            data_cfg,
        )
    except Exception as exc:
        print(f"Warning: daily data load failed for {symbol} ({exc}) - skipping.")
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
        print(f"Benchmark symbol {benchmark_symbol} removed from modeling universe.")
    benchmark_df = df[df["ts_code"] == benchmark_symbol].copy()
    df = df[df["ts_code"] != benchmark_symbol].copy()

basic_df = None
if DROP_ST or MIN_LISTED_DAYS > 0:
    try:
        if MARKET != "cn" and DROP_ST:
            print(f"Note: drop_st is CN-specific; attempting basic data for market '{MARKET}'.")
        basic_df = load_basic(MARKET, CACHE_DIR, data_client, data_cfg, symbols_for_data)
    except Exception as exc:
        print(f"Warning: basic data load failed ({exc}); skipping ST/listed filters.")
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
print("Engineering features ...")

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

if WINSORIZE_PCT:
    def _winsorize(group: pd.DataFrame) -> pd.DataFrame:
        lower = group[TARGET].quantile(WINSORIZE_PCT)
        upper = group[TARGET].quantile(1 - WINSORIZE_PCT)
        group[TARGET] = group[TARGET].clip(lower, upper)
        return group

    df = df.groupby("trade_date", group_keys=False).apply(_winsorize)


def apply_cross_sectional_transform(
    data: pd.DataFrame,
    features: list[str],
    method: str,
    winsorize_pct: Optional[float],
) -> pd.DataFrame:
    if method == "none":
        return data

    def _transform(group: pd.DataFrame) -> pd.DataFrame:
        values = group[features].copy()
        if winsorize_pct:
            lower = values.quantile(winsorize_pct)
            upper = values.quantile(1 - winsorize_pct)
            values = values.clip(lower=lower, upper=upper, axis=1)
        if method == "zscore":
            mean = values.mean()
            std = values.std(ddof=0).replace(0, np.nan)
            values = (values - mean) / std
            values = values.fillna(0.0)
        elif method == "rank":
            values = values.rank(method="average", pct=True) - 0.5
        group[features] = values
        return group

    return data.groupby("trade_date", group_keys=False, sort=False).apply(_transform).reset_index(drop=True)

# Drop dates with too few symbols for evaluation
date_counts = df.groupby("trade_date")["ts_code"].nunique()
valid_dates = date_counts[date_counts >= MIN_SYMBOLS_PER_DATE].index
if len(valid_dates) != len(date_counts):
    df = df[df["trade_date"].isin(valid_dates)].copy()

if CS_METHOD != "none":
    df = apply_cross_sectional_transform(df, FEATURES, CS_METHOD, CS_WINSORIZE_PCT)

# -----------------------------------------------------------------------------
# 4. Train-test split (time-series by date)
# -----------------------------------------------------------------------------
print("Splitting train/test by date ...")
all_dates = np.array(sorted(df["trade_date"].unique()))
if len(all_dates) < 10:
    sys.exit("Not enough dates for a meaningful split.")

split_idx = int(len(all_dates) * (1 - TEST_SIZE))
train_end = split_idx
if EMBARGO_DAYS > 0:
    train_end = max(0, split_idx - EMBARGO_DAYS)
train_dates = all_dates[:train_end]
test_dates = all_dates[split_idx:]

train_df = df[df["trade_date"].isin(train_dates)].copy()
test_df = df[df["trade_date"].isin(test_dates)].copy()

if train_df.empty or test_df.empty:
    sys.exit("Not enough dates for train/test after embargo.")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# -----------------------------------------------------------------------------
# 5. Cross-validation on dates (IC metric)
# -----------------------------------------------------------------------------
print("Time-series cross-validation (IC) ...")


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return x_rank.corr(y_rank)


def daily_ic_series(data: pd.DataFrame, target_col: str, pred_col: str) -> np.ndarray:
    daily = []
    for _, group in data.groupby("trade_date"):
        if group[target_col].nunique() < 2:
            continue
        ic = spearman_corr(group[pred_col], group[target_col])
        if not np.isnan(ic):
            daily.append(ic)
    return np.array(daily)


def summarize_ic(data: pd.DataFrame, target_col: str, pred_col: str) -> tuple[np.ndarray, float, float, float]:
    ic_values = daily_ic_series(data, target_col, pred_col)
    ic_mean = np.nanmean(ic_values)
    ic_std = np.nanstd(ic_values)
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    return ic_values, ic_mean, ic_std, ic_ir


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
        scores.append(np.nanmean(ic_values))
    return scores


def time_series_cv_ic(
    data: pd.DataFrame,
    n_splits: int,
    embargo_days: int,
    signal_direction: float,
) -> list:
    dates = np.array(sorted(data["trade_date"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(dates):
        if embargo_days > 0:
            cutoff = val_idx[0] - embargo_days
            train_idx = train_idx[train_idx < cutoff]
            if len(train_idx) == 0:
                continue
        tr_dates = dates[train_idx]
        va_dates = dates[val_idx]
        tr_df = data[data["trade_date"].isin(tr_dates)]
        va_df = data[data["trade_date"].isin(va_dates)].copy()

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(tr_df[FEATURES], tr_df[TARGET])
        va_df["pred"] = model.predict(va_df[FEATURES])
        if signal_direction != 1.0:
            va_df["pred"] = va_df["pred"] * signal_direction

        ic_values = daily_ic_series(va_df, TARGET, "pred")
        scores.append(np.nanmean(ic_values))
    return scores


cv_scores = time_series_cv_ic(train_df, N_SPLITS, EMBARGO_DAYS, SIGNAL_DIRECTION)
if cv_scores:
    print(f"CV IC: mean={np.nanmean(cv_scores):.4f}, std={np.nanstd(cv_scores):.4f}")
    print(f"CV fold ICs: {[f'{s:.4f}' for s in cv_scores]}")
else:
    print("CV IC not available - insufficient data after embargo.")

# -----------------------------------------------------------------------------
# 6. Fit final model
# -----------------------------------------------------------------------------
print("Fitting XGBoost regressor ...")
model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 7. Evaluation (cross-sectional factor style)
# -----------------------------------------------------------------------------
print("Evaluating model on train/test sets ...")

train_eval_df = train_df.copy()
train_eval_df["pred"] = model.predict(train_eval_df[FEATURES])
train_signal_col = "pred"
if SIGNAL_DIRECTION != 1.0:
    train_eval_df["signal"] = train_eval_df["pred"] * SIGNAL_DIRECTION
    train_signal_col = "signal"
if REPORT_TRAIN_IC:
    train_ic_values, train_ic_mean, train_ic_std, train_ic_ir = summarize_ic(
        train_eval_df, TARGET, train_signal_col
    )
    print(
        f"Train Daily IC: mean={train_ic_mean:.4f}, std={train_ic_std:.4f}, "
        f"IR={train_ic_ir:.2f} (n={len(train_ic_values)})"
    )

test_df["pred"] = model.predict(test_df[FEATURES])
signal_col = "pred"
if SIGNAL_DIRECTION != 1.0:
    test_df["signal"] = test_df["pred"] * SIGNAL_DIRECTION
    signal_col = "signal"
    print(f"Signal direction applied to ranking: {SIGNAL_DIRECTION}")

# Daily IC
ic_values, ic_mean, ic_std, ic_ir = summarize_ic(test_df, TARGET, signal_col)
print(f"Daily IC: mean={ic_mean:.4f}, std={ic_std:.4f}, IR={ic_ir:.2f} (n={len(ic_values)})")

if PERM_TEST_ENABLED:
    print("Permutation test (shuffle train labels within date) ...")
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
        print(f"Permutation IC: mean={perm_mean:.4f}, std={perm_std:.4f}, runs={len(perm_scores)}")
        print(f"Permutation ICs: {[f'{s:.4f}' for s in perm_scores]}")

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


def quantile_returns(data: pd.DataFrame, pred_col: str, target_col: str, n_quantiles: int) -> pd.Series:
    def _add_quantile(group):
        if len(group) < n_quantiles:
            return pd.Series([np.nan] * len(group), index=group.index)
        ranks = group[pred_col].rank(method="first")
        return pd.qcut(ranks, n_quantiles, labels=False)

    data = data.copy()
    data["quantile"] = data.groupby("trade_date", group_keys=False).apply(_add_quantile)
    data = data.dropna(subset=["quantile"])  # drop dates with insufficient symbols

    q_ret = data.groupby(["trade_date", "quantile"])[target_col].mean().unstack()
    return q_ret.mean()


rebalance_dates = get_rebalance_dates(sorted(test_df["trade_date"].unique()), REBALANCE_FREQUENCY)
eval_df = test_df[test_df["trade_date"].isin(rebalance_dates)].copy()

quantile_mean = quantile_returns(eval_df, signal_col, TARGET, N_QUANTILES)
if not quantile_mean.empty:
    for q_idx, value in quantile_mean.items():
        print(f"Q{int(q_idx) + 1} mean return: {value:.4%}")
    long_short = quantile_mean.iloc[-1] - quantile_mean.iloc[0]
    print(f"Long-short (Q{N_QUANTILES}-Q1): {long_short:.4%}")
else:
    print("Quantile returns not available - insufficient symbols per date.")

# Turnover estimate for top-K portfolio

def estimate_turnover(data: pd.DataFrame, pred_col: str, k: int, freq: str):
    dates = sorted(data["trade_date"].unique())
    rebalance = get_rebalance_dates(dates, freq)
    prev = None
    turnovers = []
    for date in rebalance:
        day = data[data["trade_date"] == date]
        if len(day) < k:
            continue
        holdings = set(day.nlargest(k, pred_col)["ts_code"])
        if prev is not None:
            overlap = len(holdings & prev)
            turnovers.append(1 - overlap / k)
        prev = holdings
    return np.nanmean(turnovers), len(turnovers)


k = min(TOP_K, eval_df["ts_code"].nunique())
turnover, n_turn = estimate_turnover(eval_df, signal_col, k, REBALANCE_FREQUENCY)
if not np.isnan(turnover):
    cost_drag = 2 * (TRANSACTION_COST_BPS / 10000.0) * turnover
    print(f"Top-{k} turnover per rebalance: {turnover:.2%} (n={n_turn})")
    print(f"Approx cost drag per rebalance: {cost_drag:.2%} at {TRANSACTION_COST_BPS} bps per side")


def backtest_topk(
    data: pd.DataFrame,
    pred_col: str,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    top_k: int,
    shift_days: int,
    cost_bps: float,
    trading_days_per_year: int,
):
    trade_dates = sorted(data["trade_date"].unique())
    if len(trade_dates) < 2:
        return None
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    price_table = data.pivot(index="trade_date", columns="ts_code", values=price_col)

    net_returns = []
    gross_returns = []
    turnovers = []
    period_info = []
    prev_holdings = None

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb = rebalance_dates[i + 1]
        if reb_date not in date_to_idx or next_reb not in date_to_idx:
            continue

        entry_idx = date_to_idx[reb_date] + shift_days
        exit_idx = date_to_idx[next_reb] + shift_days
        if entry_idx >= len(trade_dates) or exit_idx >= len(trade_dates) or entry_idx >= exit_idx:
            continue

        entry_date = trade_dates[entry_idx]
        exit_date = trade_dates[exit_idx]
        day = data[data["trade_date"] == reb_date]
        if day.empty:
            continue

        k = min(top_k, len(day))
        if k <= 0:
            continue

        holdings = list(day.nlargest(k, pred_col)["ts_code"])
        entry_prices = price_table.loc[entry_date, holdings]
        exit_prices = price_table.loc[exit_date, holdings]
        valid = entry_prices.notna() & exit_prices.notna()
        if valid.sum() == 0:
            continue

        period_returns = (exit_prices[valid] / entry_prices[valid]) - 1.0
        gross = period_returns.mean()
        turnover = np.nan
        if prev_holdings is not None:
            overlap = len(set(holdings) & prev_holdings)
            turnover = 1 - overlap / k
        cost = 0.0 if prev_holdings is None else 2 * (cost_bps / 10000.0) * turnover
        net = gross - cost

        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)
        period_info.append((entry_idx, exit_idx, entry_date, exit_date))
        prev_holdings = set(holdings)

    if not net_returns:
        return None

    index = [info[3] for info in period_info]
    net_series = pd.Series(net_returns, index=index, name="net_return")
    gross_series = pd.Series(gross_returns, index=index, name="gross_return")
    turnover_series = pd.Series(turnovers, index=index, name="turnover")

    nav = (1 + net_series).cumprod()
    total_return = nav.iloc[-1] - 1.0
    max_drawdown = (nav / nav.cummax() - 1.0).min()

    entry_first, exit_last = period_info[0][0], period_info[-1][1]
    total_days = exit_last - entry_first
    if total_days > 0:
        ann_return = (1 + total_return) ** (trading_days_per_year / total_days) - 1.0
    else:
        ann_return = np.nan

    holding_lengths = [info[1] - info[0] for info in period_info]
    avg_holding = np.mean(holding_lengths) if holding_lengths else np.nan
    periods_per_year = (
        trading_days_per_year / avg_holding
        if np.isfinite(avg_holding) and avg_holding > 0
        else np.nan
    )
    period_vol = net_series.std(ddof=1)
    if np.isfinite(period_vol) and period_vol > 0 and np.isfinite(periods_per_year):
        ann_vol = period_vol * np.sqrt(periods_per_year)
        sharpe = net_series.mean() / period_vol * np.sqrt(periods_per_year)
    else:
        ann_vol = np.nan
        sharpe = np.nan

    avg_turnover = turnover_series.dropna().mean() if turnover_series.notna().any() else np.nan
    avg_cost = 2 * (cost_bps / 10000.0) * avg_turnover if not np.isnan(avg_turnover) else np.nan

    stats = {
        "periods": len(net_series),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "avg_cost_drag": avg_cost,
    }
    return stats, net_series, gross_series, period_info


if BACKTEST_ENABLED:
    if not BACKTEST_LONG_ONLY:
        print("Backtest only supports long-only at the moment; set backtest.long_only: true.")
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
        )

        if bt_result is None:
            print("Backtest not available - insufficient data.")
        else:
            stats, net_series, gross_series, period_info = bt_result
            print("Backtest (long-only, top-K):")
            print(f"  periods: {stats['periods']}")
            print(f"  total return: {stats['total_return']:.2%}")
            print(f"  ann return: {stats['ann_return']:.2%}")
            print(f"  ann vol: {stats['ann_vol']:.2%}")
            print(f"  sharpe: {stats['sharpe']:.2f}")
            print(f"  max drawdown: {stats['max_drawdown']:.2%}")
            if not np.isnan(stats["avg_turnover"]):
                print(f"  avg turnover: {stats['avg_turnover']:.2%}")
                print(f"  avg cost drag: {stats['avg_cost_drag']:.2%}")

            if benchmark_df is not None and not benchmark_df.empty:
                bench_prices = benchmark_df.set_index("trade_date")[PRICE_COL]
                bench_returns = []
                bench_index = []
                for _, _, entry_date, exit_date in period_info:
                    if entry_date not in bench_prices.index or exit_date not in bench_prices.index:
                        continue
                    bench_returns.append(bench_prices.loc[exit_date] / bench_prices.loc[entry_date] - 1.0)
                    bench_index.append(exit_date)
                if bench_returns:
                    bench_series = pd.Series(bench_returns, index=bench_index, name="benchmark_return")
                    bench_nav = (1 + bench_series).cumprod()
                    bench_total = bench_nav.iloc[-1] - 1.0
                    print(f"  benchmark total return: {bench_total:.2%}")

# Feature importance
print("Feature importance:")
importance_df = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
for _, row in importance_df.iterrows():
    print(f"  {row['feature']:<20}: {row['importance']:.4f}")

# Optional: save the model
# from joblib import dump; dump(model, "xgb_factor_model.joblib")
