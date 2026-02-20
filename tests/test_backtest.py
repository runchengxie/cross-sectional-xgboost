import numpy as np
import pandas as pd
import pytest

from csml.backtest import backtest_topk


def test_backtest_initial_cost_applied():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "ts_code": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=10,
        trading_days_per_year=252,
        exit_mode="rebalance",
    )
    stats, net_series, gross_series, turnover_series, _ = result
    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], 0.10)
    assert np.isclose(net_series.iloc[0], 0.10 - 0.001)
    assert np.isclose(turnover_series.iloc[0], 1.0)
    assert stats["periods_with_delayed_exit"] == 0


def test_backtest_turnover_accounts_for_weight_drift():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "ts_code": ["A", "B"] * 3,
            "pred": [2.0, 1.0] * 3,
            "close": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=2,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
    )
    _, _, _, turnover_series, _ = result
    # First period is initial entry (turnover=1). Second period should reflect drift.
    assert turnover_series.shape[0] == 2
    assert np.isclose(turnover_series.iloc[1], 1 / 6, atol=1e-6)


def test_backtest_label_horizon_overlap_raises():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                ]
            ),
            "ts_code": ["A", "B"] * 4,
            "pred": [2.0, 1.0] * 4,
            "close": [100.0, 100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0],
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    with pytest.raises(ValueError):
        backtest_topk(
            df,
            pred_col="pred",
            price_col="close",
            rebalance_dates=rebalance_dates,
            top_k=1,
            shift_days=0,
            cost_bps=0,
            trading_days_per_year=252,
            exit_mode="label_horizon",
            exit_horizon_days=2,
        )


def test_backtest_long_short_basic():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
            ),
            "ts_code": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        long_only=False,
        short_k=1,
    )
    stats, net_series, gross_series, turnover_series, _ = result
    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], 0.2)
    assert np.isclose(net_series.iloc[0], 0.2)
    assert np.isclose(turnover_series.iloc[0], 2.0)


def test_backtest_exit_delay_uses_next_available_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]
            ),
            "ts_code": ["A", "B", "B", "A", "B"],
            "pred": [2.0, 1.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 100.0, 90.0, 100.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
    )
    stats, net_series, _, _, period_info = result
    assert net_series.index[0] == pd.Timestamp("2020-01-03")
    assert np.isclose(net_series.iloc[0], -0.10)
    assert stats["periods_with_delayed_exit"] == 1
    assert np.isclose(stats["avg_exit_lag_days"], 1.0)
    assert np.isclose(stats["max_exit_lag_days"], 1.0)
    assert period_info[0]["planned_exit_date"] == pd.Timestamp("2020-01-02")
    assert period_info[0]["exit_date"] == pd.Timestamp("2020-01-03")
    assert period_info[0]["exit_delay_steps"] == 1


def test_backtest_buffer_reduces_turnover():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "ts_code": ["A", "B"] * 3,
            "pred": [2.0, 1.0, 1.0, 2.0, 1.0, 2.0],
            "close": [100.0] * 6,
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        buffer_exit=1,
        buffer_entry=0,
    )
    _, _, _, turnover_series, _ = result
    assert turnover_series.shape[0] == 2
    assert np.isclose(turnover_series.iloc[1], 0.0)


def test_backtest_exit_strict_skips_missing_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ts_code": ["A", "A"],
            "pred": [1.0, 1.0],
            "close": [100.0, np.nan],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="strict",
    )
    assert result is None


def test_backtest_exit_ffill_uses_last_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ts_code": ["A", "A"],
            "pred": [1.0, 1.0],
            "close": [100.0, np.nan],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="ffill",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert net_series.index[0] == pd.Timestamp("2020-01-02")
    assert np.isclose(net_series.iloc[0], 0.0)


def test_backtest_tradable_filters_entry_selection():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
            ),
            "ts_code": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
            "is_tradable": [False, True, False, True],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="strict",
        tradable_col="is_tradable",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], -0.10)


def test_backtest_exit_delay_with_none_fallback_skips_unresolved_exit():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "ts_code": ["A", "A", "A"],
            "pred": [1.0, 1.0, 1.0],
            "close": [100.0, np.nan, np.nan],
            "is_tradable": [True, False, False],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
        exit_fallback_policy="none",
        tradable_col="is_tradable",
    )
    assert result is None


def test_backtest_exit_delay_with_ffill_fallback_uses_previous_tradable_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "ts_code": ["A", "A", "A"],
            "pred": [1.0, 1.0, 1.0],
            "close": [100.0, 99.0, 98.0],
            "is_tradable": [True, False, False],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
        exit_fallback_policy="ffill",
        tradable_col="is_tradable",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert net_series.index[0] == pd.Timestamp("2020-01-02")
    assert np.isclose(net_series.iloc[0], 0.0)
