import pandas as pd

from csml import data_providers


def _daily_frame(symbol: str, start: str, end: str, *, close_offset: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    rows = []
    for idx, trade_date in enumerate(dates):
        rows.append(
            {
                "trade_date": trade_date.strftime("%Y%m%d"),
                "ts_code": symbol,
                "close": close_offset + float(idx + 1),
                "vol": 1000.0 + idx,
                "amount": 10000.0 + idx,
            }
        )
    return pd.DataFrame(rows)


def test_fetch_daily_symbol_cache_refresh_window_merges_monotonic(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbol = "AAA"
    cache_file = cache_dir / "us_tushare_daily_AAA.parquet"

    cached = _daily_frame(symbol, "20200101", "20200105", close_offset=0.0)
    cached.to_parquet(cache_file)

    fetch_ranges = []

    def fake_fetch(provider, market, symbol_value, start_date, end_date, client, data_cfg):
        fetch_ranges.append((start_date, end_date))
        return _daily_frame(symbol_value, start_date, end_date, close_offset=100.0)

    monkeypatch.setattr(data_providers, "_fetch_daily_from_provider", fake_fetch)

    data_cfg = {
        "provider": "tushare",
        "cache_mode": "symbol",
        "cache_refresh_days": 2,
        "cache_refresh_on_hit": False,
    }
    result = data_providers.fetch_daily(
        "us",
        symbol,
        "20200102",
        "20200107",
        cache_dir,
        client=None,
        data_cfg=data_cfg,
    )

    assert fetch_ranges == [("20200104", "20200107")]
    assert result["trade_date"].tolist() == [
        "20200102",
        "20200103",
        "20200104",
        "20200105",
        "20200106",
        "20200107",
    ]
    assert result["trade_date"].is_monotonic_increasing
    assert result["trade_date"].nunique() == len(result)

    merged = pd.read_parquet(cache_file).sort_values("trade_date").reset_index(drop=True)
    refreshed_close = float(merged.loc[merged["trade_date"] == "20200104", "close"].iloc[0])
    assert refreshed_close > 100.0


def test_fetch_daily_symbol_cache_refresh_on_hit_triggers_tail_refresh(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbol = "AAA"
    cache_file = cache_dir / "us_tushare_daily_AAA.parquet"

    cached = _daily_frame(symbol, "20200101", "20200105", close_offset=0.0)
    cached.to_parquet(cache_file)

    fetch_ranges = []

    def fake_fetch(provider, market, symbol_value, start_date, end_date, client, data_cfg):
        fetch_ranges.append((start_date, end_date))
        return _daily_frame(symbol_value, start_date, end_date, close_offset=200.0)

    monkeypatch.setattr(data_providers, "_fetch_daily_from_provider", fake_fetch)

    data_cfg = {
        "provider": "tushare",
        "cache_mode": "symbol",
        "cache_refresh_days": 2,
        "cache_refresh_on_hit": True,
    }
    result = data_providers.fetch_daily(
        "us",
        symbol,
        "20200102",
        "20200105",
        cache_dir,
        client=None,
        data_cfg=data_cfg,
    )

    assert fetch_ranges == [("20200104", "20200105")]
    assert result["trade_date"].tolist() == [
        "20200102",
        "20200103",
        "20200104",
        "20200105",
    ]
