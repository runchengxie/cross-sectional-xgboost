"""main.py - Cross-sectional factor mining with XGBoost regression.
Usage:
    $ python main.py  # requires the TUSHARE_API_KEY env var
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
# Workaround for pandas_ta NaN import issue
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
import tushare as ts
import pyarrow  # ensures parquet support
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
TOKEN = os.getenv("TUSHARE_API_KEY")
if not TOKEN:
    sys.exit("Please set the TUSHARE_API_KEY environment variable first.")

SYMBOLS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "UNH",
    "XOM",
]

end_date = datetime.now()
start_date = end_date - timedelta(days=5 * 365)
START_DATE = start_date.strftime("%Y%m%d")
END_DATE = end_date.strftime("%Y%m%d")

LABEL_HORIZON_DAYS = 5   # forward return horizon (e.g., 5 trading days ~ weekly)
TEST_SIZE = 0.2          # fraction of dates for hold-out test
N_SPLITS = 5             # time-series CV splits on dates
N_QUANTILES = 5          # quantile buckets for evaluation
TOP_K = 20               # top-K holdings for turnover estimate
REBALANCE_FREQUENCY = "W"  # "D", "W", "M" or None
TRANSACTION_COST_BPS = 10  # per side, rough estimate for cost drag

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

FEATURES = [
    "sma_20",
    "sma_5_diff",
    "sma_10_diff",
    "sma_20_diff",
    "rsi_14",
    "macd_hist",
    "volume_sma5_ratio",
    "vol",
]
TARGET = "future_return"

# -----------------------------------------------------------------------------
# 2. Data download
# -----------------------------------------------------------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

ts.set_token(TOKEN)
pro = ts.pro_api()


def load_symbol_data(symbol: str) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"us_daily_{symbol}_{START_DATE}_{END_DATE}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    df_symbol = pro.us_daily(ts_code=symbol, start_date=START_DATE, end_date=END_DATE)
    if df_symbol.empty:
        print(f"No data for {symbol} - skipping.")
        return df_symbol
    if "ts_code" not in df_symbol.columns:
        df_symbol["ts_code"] = symbol
    df_symbol.to_parquet(cache_file)
    return df_symbol


frames = []
for symbol in SYMBOLS:
    print(f"Fetching daily data for {symbol} ...")
    data = load_symbol_data(symbol)
    if not data.empty:
        frames.append(data)

if not frames:
    sys.exit("No data returned - check symbols and date range.")

df = pd.concat(frames, ignore_index=True)
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df.sort_values(["ts_code", "trade_date"], inplace=True)

# -----------------------------------------------------------------------------
# 3. Feature engineering (per symbol) + label
# -----------------------------------------------------------------------------
print("Engineering features ...")


def add_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("trade_date").copy()

    for win in (5, 10, 20):
        group[f"sma_{win}"] = ta.sma(group["close"], length=win)
        group[f"sma_{win}_diff"] = group[f"sma_{win}"].pct_change()

    group["rsi_14"] = ta.rsi(group["close"], length=14)

    macd = ta.macd(group["close"], fast=12, slow=26, signal=9)
    if macd is not None and "MACDh_12_26_9" in macd.columns:
        group["macd_hist"] = macd["MACDh_12_26_9"]
    else:
        group["macd_hist"] = np.nan

    volume_sma5 = ta.sma(group["vol"], length=5)
    if volume_sma5 is None:
        volume_sma5 = group["vol"].rolling(window=5).mean()
    group["volume_sma5"] = volume_sma5
    group["volume_sma5_ratio"] = group["vol"] / group["volume_sma5"]

    group[TARGET] = group["close"].shift(-LABEL_HORIZON_DAYS) / group["close"] - 1.0

    return group


df = df.groupby("ts_code", group_keys=False).apply(add_features)

# Keep only the necessary columns & drop NaNs from rolling calcs / future label
cols = ["trade_date", "ts_code"] + FEATURES + [TARGET]
df = df[cols].dropna().reset_index(drop=True)

# Drop dates with too few symbols for quantiles
date_counts = df.groupby("trade_date")["ts_code"].nunique()
valid_dates = date_counts[date_counts >= N_QUANTILES].index
if len(valid_dates) != len(date_counts):
    df = df[df["trade_date"].isin(valid_dates)].copy()

# -----------------------------------------------------------------------------
# 4. Train-test split (time-series by date)
# -----------------------------------------------------------------------------
print("Splitting train/test by date ...")
all_dates = np.array(sorted(df["trade_date"].unique()))
if len(all_dates) < 10:
    sys.exit("Not enough dates for a meaningful split.")

split_idx = int(len(all_dates) * (1 - TEST_SIZE))
train_dates = all_dates[:split_idx]
test_dates = all_dates[split_idx:]

train_df = df[df["trade_date"].isin(train_dates)].copy()
test_df = df[df["trade_date"].isin(test_dates)].copy()

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


def time_series_cv_ic(data: pd.DataFrame, n_splits: int) -> list:
    dates = np.array(sorted(data["trade_date"].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(dates):
        tr_dates = dates[train_idx]
        va_dates = dates[val_idx]
        tr_df = data[data["trade_date"].isin(tr_dates)]
        va_df = data[data["trade_date"].isin(va_dates)].copy()

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(tr_df[FEATURES], tr_df[TARGET])
        va_df["pred"] = model.predict(va_df[FEATURES])

        ic_values = daily_ic_series(va_df, TARGET, "pred")
        scores.append(np.nanmean(ic_values))
    return scores


cv_scores = time_series_cv_ic(train_df, N_SPLITS)
print(f"CV IC: mean={np.nanmean(cv_scores):.4f}, std={np.nanstd(cv_scores):.4f}")
print(f"CV fold ICs: {[f'{s:.4f}' for s in cv_scores]}")

# -----------------------------------------------------------------------------
# 6. Fit final model
# -----------------------------------------------------------------------------
print("Fitting XGBoost regressor ...")
model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 7. Evaluation (cross-sectional factor style)
# -----------------------------------------------------------------------------
print("Evaluating on test set ...")

test_df["pred"] = model.predict(test_df[FEATURES])

# Daily IC
ic_values = daily_ic_series(test_df, TARGET, "pred")
ic_mean = np.nanmean(ic_values)
ic_std = np.nanstd(ic_values)
ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

print(f"Daily IC: mean={ic_mean:.4f}, std={ic_std:.4f}, IR={ic_ir:.2f} (n={len(ic_values)})")

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

quantile_mean = quantile_returns(eval_df, "pred", TARGET, N_QUANTILES)
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
turnover, n_turn = estimate_turnover(eval_df, "pred", k, REBALANCE_FREQUENCY)
if not np.isnan(turnover):
    cost_drag = 2 * (TRANSACTION_COST_BPS / 10000.0) * turnover
    print(f"Top-{k} turnover per rebalance: {turnover:.2%} (n={n_turn})")
    print(f"Approx cost drag per rebalance: {cost_drag:.2%} at {TRANSACTION_COST_BPS} bps per side")

# Feature importance
print("Feature importance:")
importance_df = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
for _, row in importance_df.iterrows():
    print(f"  {row['feature']:<20}: {row['importance']:.4f}")

# Optional: save the model
# from joblib import dump; dump(model, "xgb_factor_model.joblib")
