import numpy as np
import pandas as pd

from csxgb.metrics import (
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


def test_daily_ic_series_pearson_perfect():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3),
            "ts_code": ["A", "B", "C"],
            "pred": [1.0, 2.0, 3.0],
            "target": [2.0, 4.0, 6.0],
        }
    )
    ic_series = daily_ic_series(df, "target", "pred", method="pearson")
    assert ic_series.shape[0] == 1
    assert np.allclose(ic_series.values, 1.0)


def test_regression_error_metrics_basic():
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.0, 1.0])
    stats = regression_error_metrics(y_true, y_pred)
    assert stats["n"] == 2
    assert np.isclose(stats["mae"], 0.5)
    assert np.isclose(stats["rmse"], np.sqrt(0.5))
    assert np.isclose(stats["r2"], -1.0)


def test_hit_rate_and_topk_positive_ratio():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4),
            "ts_code": ["A", "B", "C", "D"],
            "pred": [0.2, -0.1, 0.0, 0.3],
            "target": [0.1, -0.05, 0.0, -0.2],
        }
    )
    stats = hit_rate(df["target"], df["pred"])
    assert stats["n"] == 4
    assert np.isclose(stats["hit_rate"], 0.75)

    topk_stats = topk_positive_ratio(df, "pred", "target", k=2)
    assert topk_stats["n_dates"] == 1
    assert np.isclose(topk_stats["topk_positive_ratio"], 0.5)


def test_bucket_ic_summary_quantile():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "ts_code": ["A", "B", "C", "D"] * 2,
            "pred": [1, 2, 3, 4, 4, 3, 2, 1],
            "target": [1, 2, 3, 4, 4, 3, 2, 1],
            "mcap": [10, 20, 30, 40, 10, 20, 30, 40],
        }
    )
    df["mcap_bucket"] = assign_daily_quantile_bucket(df, "mcap", n_bins=2)
    summary = bucket_ic_summary(df, "target", "pred", "mcap_bucket")
    assert not summary.empty
    assert "mean" in summary.columns


def test_summarize_active_returns_basic():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    strategy = pd.Series([0.02, 0.01], index=dates)
    benchmark = pd.Series([0.0, 0.02], index=dates)
    stats, active = summarize_active_returns(strategy, benchmark, periods_per_year=2)
    assert active.shape[0] == 2
    assert np.isclose(stats["mean"], 0.005)
    assert np.isfinite(stats["active_total_return"])
