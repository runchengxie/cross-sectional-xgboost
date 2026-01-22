"""Market-aware data providers for daily OHLCV data and basic metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


DEFAULT_DAILY_ENDPOINTS = {
    "cn": "daily",
    "hk": "hk_daily",
    "us": "us_daily",
}

DEFAULT_BASIC_ENDPOINTS = {
    "cn": "stock_basic",
    "hk": "hk_basic",
    "us": "us_basic",
}

DEFAULT_BASIC_FIELDS = {
    "cn": "ts_code,name,list_date",
    "hk": "ts_code,name,list_date",
    "us": "ts_code,name,list_date",
}

DEFAULT_BASIC_PARAMS = {
    "cn": {"list_status": "L"},
    "hk": {},
    "us": {},
}

DEFAULT_COLUMN_MAPS = {
    "cn": {
        "trade_date": "trade_date",
        "ts_code": "ts_code",
        "close": "close",
        "vol": "vol",
        "amount": "amount",
    },
    "hk": {
        "trade_date": "trade_date",
        "ts_code": "ts_code",
        "close": "close",
        "vol": "vol",
        "amount": "amount",
    },
    "us": {
        "trade_date": "trade_date",
        "ts_code": "ts_code",
        "close": "close",
        "vol": "vol",
        "amount": "amount",
    },
}

COLUMN_CANDIDATES = {
    "trade_date": ["trade_date", "date", "trade_dt", "trade_day"],
    "ts_code": ["ts_code", "symbol", "ticker", "code", "sec_code", "tscode"],
    "close": ["close", "close_price", "adj_close", "close_adj", "cls"],
    "vol": ["vol", "volume", "trade_vol", "volume_traded"],
    "amount": ["amount", "turnover", "trade_value", "value"],
}

REQUIRED_DAILY_COLUMNS = ("trade_date", "close", "vol")


def normalize_market(market: Optional[str]) -> str:
    value = str(market or "cn").strip().lower()
    return value if value else "cn"


def _merge_column_map(market: str, data_cfg: Mapping) -> dict[str, str]:
    merged = dict(DEFAULT_COLUMN_MAPS.get(market, {}))
    cfg_map = data_cfg.get("column_map") if isinstance(data_cfg, Mapping) else None
    if isinstance(cfg_map, Mapping):
        for key, value in cfg_map.items():
            if value:
                merged[str(key)] = str(value)
    return merged


def _apply_column_map(df: pd.DataFrame, column_map: Mapping[str, str]) -> pd.DataFrame:
    rename_map = {}
    for standard, source in column_map.items():
        if source in df.columns and standard != source:
            rename_map[source] = standard
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _infer_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    for standard, candidates in COLUMN_CANDIDATES.items():
        if standard in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df = df.rename(columns={candidate: standard})
                break
    return df


def _standardize_daily_frame(
    df: pd.DataFrame,
    market: str,
    data_cfg: Mapping,
    symbol: str,
) -> pd.DataFrame:
    df = _apply_column_map(df, _merge_column_map(market, data_cfg))
    df = _infer_missing_columns(df)
    if "ts_code" not in df.columns:
        df = df.copy()
        df["ts_code"] = symbol
    missing = [col for col in REQUIRED_DAILY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Daily data missing required columns: {missing}")
    return df


def _resolve_endpoint_name(
    market: str,
    data_cfg: Mapping,
    key: str,
    defaults: Mapping[str, str],
) -> Optional[str]:
    if isinstance(data_cfg, Mapping) and data_cfg.get(key):
        return str(data_cfg.get(key)).strip()
    return defaults.get(market)


def _resolve_endpoint_params(
    data_cfg: Mapping,
    params_key: str,
    default_params: Optional[Mapping[str, object]] = None,
) -> dict:
    params = dict(default_params or {})
    cfg_params = data_cfg.get(params_key) if isinstance(data_cfg, Mapping) else None
    if isinstance(cfg_params, Mapping):
        params.update(cfg_params)
    return params


def fetch_daily(
    market: str,
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    pro,
    data_cfg: Optional[Mapping] = None,
) -> pd.DataFrame:
    market = normalize_market(market)
    data_cfg = data_cfg or {}
    cache_file = cache_dir / f"{market}_daily_{symbol}_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    endpoint_name = _resolve_endpoint_name(market, data_cfg, "daily_endpoint", DEFAULT_DAILY_ENDPOINTS)
    if not endpoint_name:
        raise ValueError(f"No daily endpoint configured for market '{market}'.")
    endpoint = getattr(pro, endpoint_name, None)
    if endpoint is None:
        raise ValueError(f"Tushare endpoint '{endpoint_name}' not found for market '{market}'.")

    params = {}
    symbol_param = data_cfg.get("daily_symbol_param", "ts_code")
    start_param = data_cfg.get("daily_start_param", "start_date")
    end_param = data_cfg.get("daily_end_param", "end_date")
    if symbol_param:
        params[str(symbol_param)] = symbol
    if start_param:
        params[str(start_param)] = start_date
    if end_param:
        params[str(end_param)] = end_date

    params.update(_resolve_endpoint_params(data_cfg, "daily_params"))

    fields = data_cfg.get("daily_fields") if isinstance(data_cfg, Mapping) else None
    if fields:
        params["fields"] = fields

    df = endpoint(**params)
    if df is None or df.empty:
        return df

    df = _standardize_daily_frame(df, market, data_cfg, symbol)
    df.to_parquet(cache_file)
    return df


def load_basic(
    market: str,
    cache_dir: Path,
    pro,
    data_cfg: Optional[Mapping] = None,
) -> pd.DataFrame:
    market = normalize_market(market)
    data_cfg = data_cfg or {}
    cache_file = cache_dir / f"{market}_basic.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    endpoint_name = _resolve_endpoint_name(market, data_cfg, "basic_endpoint", DEFAULT_BASIC_ENDPOINTS)
    if not endpoint_name:
        raise ValueError(f"No basic endpoint configured for market '{market}'.")
    endpoint = getattr(pro, endpoint_name, None)
    if endpoint is None:
        raise ValueError(f"Tushare endpoint '{endpoint_name}' not found for market '{market}'.")

    params = _resolve_endpoint_params(data_cfg, "basic_params", DEFAULT_BASIC_PARAMS.get(market))
    fields = data_cfg.get("basic_fields") or DEFAULT_BASIC_FIELDS.get(market)
    if fields:
        params["fields"] = fields

    df_basic = endpoint(**params)
    if df_basic is None or df_basic.empty:
        return df_basic

    df_basic.to_parquet(cache_file)
    return df_basic
