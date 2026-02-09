import os

import pandas as pd
import pytest

from csml.data_interface import DataInterface


@pytest.mark.integration
def test_tushare_provider_fetch_daily_real_token(tmp_path):
    if os.getenv("CSML_RUN_PROVIDER_INTEGRATION") != "1":
        pytest.skip("Set CSML_RUN_PROVIDER_INTEGRATION=1 to enable real provider integration tests.")

    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY")
    if not token:
        pytest.skip("Set TUSHARE_TOKEN (or TUSHARE_API_KEY) to run this integration test.")

    symbol = os.getenv("CSML_INTEGRATION_TUSHARE_SYMBOL", "000001.SZ")
    end = (pd.Timestamp.now().normalize() - pd.Timedelta(days=2)).strftime("%Y%m%d")
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=45)).strftime("%Y%m%d")

    di = DataInterface(
        market="cn",
        data_cfg={"provider": "tushare", "retry": {"max_attempts": 1}},
        cache_dir=tmp_path / "cache",
    )

    frame = di.fetch_daily(symbol, start, end)
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert {"trade_date", "ts_code", "close"}.issubset(frame.columns)
