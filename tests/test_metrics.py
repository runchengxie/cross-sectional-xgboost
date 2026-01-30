import numpy as np
import pandas as pd

from csxgb.metrics import (
    daily_ic_series,
    summarize_ic,
    quantile_returns,
    estimate_turnover,
    summarize_active_returns,
)


def test_daily_ic_series_perfect_rank():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01"] * 3 + ["2020-01-02"] * 3
            ),
            "ts_code": ["A", "B", "C"] * 2,
            "pred": [1, 2, 3, 3, 2, 1],
            "target": [1, 2, 3, 3, 2, 1],
        }
    )
    ic_series = daily_ic_series(df, "target", "pred")
    assert ic_series.shape[0] == 2
    assert np.allclose(ic_series.values, 1.0)

    stats = summarize_ic(ic_series)
    assert stats["n"] == 2
    assert stats["mean"] > 0.9


def test_quantile_returns_shape_and_turnover():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01"] * 4 + ["2020-01-02"] * 4
            ),
            "ts_code": ["A", "B", "C", "D"] * 2,
            "pred": [4, 3, 2, 1, 4, 3, 2, 1],
            "target": [0.04, 0.03, 0.02, 0.01] * 2,
        }
    )
    q_ret = quantile_returns(df, "pred", "target", n_quantiles=2)
    assert q_ret.shape[0] == 2
    assert q_ret.shape[1] == 2

    rebalance_dates = sorted(df["trade_date"].unique())
    turnover = estimate_turnover(df, "pred", k=2, rebalance_dates=rebalance_dates)
    assert turnover.shape[0] == 1
    assert np.isclose(turnover.iloc[0], 0.0)


def test_quantile_returns_insufficient_symbols():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 2),
            "ts_code": ["A", "B"],
            "pred": [1, 2],
            "target": [0.01, 0.02],
        }
    )
    q_ret = quantile_returns(df, "pred", "target", n_quantiles=5)
    assert q_ret.empty


def test_summarize_active_returns_basic():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    strategy = pd.Series([0.02, 0.01], index=dates)
    benchmark = pd.Series([0.0, 0.02], index=dates)
    stats, active = summarize_active_returns(strategy, benchmark, periods_per_year=2)
    assert active.shape[0] == 2
    assert np.isclose(stats["mean"], 0.005)
    assert np.isfinite(stats["active_total_return"])
