"""main.py – XGBoost classification of ±0.2 % next‑day moves.
Usage:
    $ python main.py  # remember to set the TUSHARE_TOKEN env‑var first
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta # Added for date calculation

import numpy as np
# Workaround for pandas_ta NaN import issue
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta  # technical‑analysis helpers
import tushare as ts
import pyarrow # Added for Parquet
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────────
# 1. Config
# ────────────────────────────────────────────────────────────────────────────────
TOKEN = os.getenv("TUSHARE_API_KEY")  # Changed to TUSHARE_API_KEY
if not TOKEN:
    sys.exit("❌  Please set the TUSHARE_API_KEY environment variable first!")

SYMBOL = "AAPL"          # Apple Inc. stock code for US market
# Calculate start date for 5 years of data
end_date = datetime.now()
start_date_dt = end_date - timedelta(days=5*365)
START_DATE = start_date_dt.strftime("%Y%m%d")
END_DATE = end_date.strftime("%Y%m%d")

TEST_SIZE = 0.2                # 20 % of data kept for hold‑out test
UP_THRESHOLD = 0.002           # +0.2 %.

# XGBoost hyper‑params – improved for better generalization
XGB_PARAMS = dict(
    n_estimators=200,         # Reduced from 500 to prevent overfitting
    learning_rate=0.01,       # Reduced from 0.05 for more conservative learning
    max_depth=3,              # Reduced from 4 to limit tree complexity
    subsample=0.7,            # Reduced from 0.8 for more regularization
    colsample_bytree=0.7,     # Reduced from 0.8 for more regularization
    reg_alpha=1.0,            # Added L1 regularization
    reg_lambda=1.0,           # Added L2 regularization
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Data download
# ────────────────────────────────────────────────────────────────────────────────
CACHE_FILE = Path("data_cache.parquet")

ts.set_token(TOKEN)
pro = ts.pro_api()

if CACHE_FILE.exists():
    print(f"♻️  Loading data from cache: {CACHE_FILE} …")
    df = pd.read_parquet(CACHE_FILE)
else:
    print(f"📥  Fetching daily bars for {SYMBOL} …")
    df = pro.us_daily(ts_code=SYMBOL, start_date=START_DATE, end_date=END_DATE)
    if df.empty:
        sys.exit("No data returned – check symbol and date range.")
    print(f"💾  Saving data to cache: {CACHE_FILE} …")
    df.to_parquet(CACHE_FILE)

df.sort_values("trade_date", inplace=True)  # chronological order

# ────────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ────────────────────────────────────────────────────────────────────────────────
print("🛠️   Engineering features …")

# Simple Moving Averages & their day‑to‑day diffs
for win in (5, 10, 20):
    df[f"SMA{win}"] = ta.sma(df["close"], length=win)
    df[f"SMA{win}_diff"] = df[f"SMA{win}"].pct_change()

# Relative Strength Index
df["RSI_14"] = ta.rsi(df["close"], length=14)

# MACD histogram
macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["MACD_hist"] = macd["MACDh_12_26_9"]

# Volume‑based signal
volume_sma5 = ta.sma(df["vol"], length=5)
if volume_sma5 is not None:
    df["Volume_SMA5"] = volume_sma5
    df["Volume_SMA5_ratio"] = df["vol"] / df["Volume_SMA5"]
else:
    # Fallback: manual calculation
    df["Volume_SMA5"] = df["vol"].rolling(window=5).mean()
    df["Volume_SMA5_ratio"] = df["vol"] / df["Volume_SMA5"]

# Label: next‑day return
print("🏷️   Labelling targets …")
df["future_return"] = df["close"].shift(-1) / df["close"] - 1.0

df["target"] = (df["future_return"] >= UP_THRESHOLD).astype(int)

# Keep only the necessary columns & drop NaNs from rolling calcs
FEATURES = [
    "SMA20",
    "SMA5_diff",
    "SMA10_diff",
    "SMA20_diff",
    "RSI_14",
    "MACD_hist",
    "Volume_SMA5_ratio",
    "vol",
]

df = df[FEATURES + ["target"]].dropna().reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4. Train‑test split (time‑series style)
# ────────────────────────────────────────────────────────────────────────────────
print("✂️   Splitting train/test chronologically …")
split_idx = int(len(df) * (1 - TEST_SIZE))

# Remove the last row from training data to avoid lookahead bias
# Since we use shift(-1) for target, the last row has NaN target anyway
X_train, X_test = df.iloc[:split_idx][FEATURES], df.iloc[split_idx:][FEATURES]
y_train, y_test = df.iloc[:split_idx]["target"], df.iloc[split_idx:]["target"]

# Double check: remove any remaining NaN targets
train_mask = ~y_train.isna()
X_train, y_train = X_train[train_mask], y_train[train_mask]

test_mask = ~y_test.isna()
X_test, y_test = X_test[test_mask], y_test[test_mask]

# ────────────────────────────────────────────────────────────────────────────────
# 5. Model fit with Cross-Validation
# ────────────────────────────────────────────────────────────────────────────────
print("🚂  Fitting XGBoost with Cross-Validation …")

# Time Series Cross-Validation
print("📊  Performing Time Series Cross-Validation ...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(XGBClassifier(**XGB_PARAMS), X_train, y_train, 
                           cv=tscv, scoring='accuracy', n_jobs=-1)

print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")

# Fit final model
model = XGBClassifier(**XGB_PARAMS)
model.fit(X_train, y_train)

# ────────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ────────────────────────────────────────────────────────────────────────────────
print("🔍  Evaluating …")
prob = model.predict_proba(X_test)[:, 1]
y_pred = (prob >= 0.5).astype(int)

print("\nCLF report (Up ≥ +0.2 %) vs. (Down/Flat):\n")
print(classification_report(y_test, y_pred, target_names=["Down/Flat", "Up ≥0.2%"], digits=2)) # Adjusted digits for screenshot consistency
print("Raw accuracy:", round(accuracy_score(y_test, y_pred), 2)) # Adjusted digits for screenshot consistency

# Feature importance analysis
print("\n📈  Feature Importance:")
feature_importance = model.feature_importances_
feature_names = FEATURES
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for idx, row in importance_df.iterrows():
    print(f"  {row['feature']:<20}: {row['importance']:.3f}")

# Also print the report on the train set for comparison
print("\nCLF report on TRAIN set (Up ≥ +0.2 %) vs. (Down/Flat):\n")
y_predict_train = model.predict(X_train)
print(classification_report(y_train, y_predict_train, target_names=["Down/Flat", "Up ≥0.2%"], digits=2)) # Adjusted digits for screenshot consistency
print("Raw accuracy on TRAIN set:", round(accuracy_score(y_train, y_predict_train), 2)) # Adjusted digits for screenshot consistency

# ────────────────────────────────────────────────────────────────────────────────
# 8. Model Analysis & Diagnostics
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊  MODEL ANALYSIS SUMMARY")
print("="*60)

train_acc = round(accuracy_score(y_train, y_predict_train), 3)
test_acc = round(accuracy_score(y_test, y_pred), 3)
cv_acc = round(cv_scores.mean(), 3)
cv_std = round(cv_scores.std(), 3)

print(f"Training Accuracy:     {train_acc}")
print(f"Test Accuracy:         {test_acc}")
print(f"CV Accuracy:           {cv_acc} ± {cv_std}")
print(f"Overfitting Gap:       {train_acc - test_acc:.3f}")

if train_acc - test_acc < 0.1:
    print("✅  Low overfitting - Good generalization!")
elif train_acc - test_acc < 0.2:
    print("⚠️   Moderate overfitting - Could be improved")
else:
    print("❌  High overfitting - Model memorizing training data")

print(f"\nClass Distribution (Test Set):")
print(f"  Down/Flat: {(y_test == 0).sum()} samples ({(y_test == 0).mean():.1%})")
print(f"  Up ≥0.2%:  {(y_test == 1).sum()} samples ({(y_test == 1).mean():.1%})")

# ────────────────────────────────────────────────────────────────────────────────
# 7. Save the model if you like
# ────────────────────────────────────────────────────────────────────────────────
# from joblib import dump; dump(model, "xgb_model.joblib")