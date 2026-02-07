from __future__ import annotations

import pandas as pd


def _clean_symbol_series(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "").astype(str).str.strip()
    return text


def ensure_symbol_columns(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    has_ts_code = "ts_code" in df.columns
    has_stock_ticker = "stock_ticker" in df.columns
    if not has_ts_code and not has_stock_ticker:
        raise SystemExit(f"{context} is missing ts_code/stock_ticker.")

    normalized = df.copy()
    ts_series = (
        _clean_symbol_series(normalized["ts_code"])
        if has_ts_code
        else pd.Series([""] * len(normalized), index=normalized.index, dtype="object")
    )
    if has_stock_ticker:
        ticker_series = _clean_symbol_series(normalized["stock_ticker"])
        merged = ticker_series.where(ticker_series != "", ts_series)
    else:
        merged = ts_series

    normalized["ts_code"] = merged
    normalized["stock_ticker"] = merged
    return normalized
