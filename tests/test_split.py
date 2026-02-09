import numpy as np
import pandas as pd

from csml.split import build_sample_weight, time_series_cv_ic


def test_time_series_cv_gap_skips_all():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["A", "B"] * len(dates),
            "f1": [1.0] * (len(dates) * 2),
            "target": [0.1] * (len(dates) * 2),
        }
    )
    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=10,
        purge_days=10,
        model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1},
        signal_direction=1.0,
    )
    assert scores == []


def test_build_sample_weight_date_equal():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["A", "B", "A"],
            "f1": [1.0, 2.0, 3.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    weights = build_sample_weight(df, "date_equal")
    assert weights is not None
    df = df.assign(weight=weights)
    sums = df.groupby("trade_date")["weight"].sum().values
    assert all(abs(value - 1.0) < 1e-12 for value in sums)


def test_time_series_cv_supports_model_cfg():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "ts_code": "A", "f1": 0.0, "target": 0.0 + idx * 0.01})
        rows.append({"trade_date": date, "ts_code": "B", "f1": 1.0, "target": 1.0 + idx * 0.01})
    df = pd.DataFrame(rows)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_supports_custom_date_col_with_unsorted_rows():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "trade_dt": date,
                "ts_code": "A",
                "f1": 0.0,
                "target": 0.0 + idx * 0.01,
            }
        )
        rows.append(
            {
                "trade_dt": date,
                "ts_code": "B",
                "f1": 1.0,
                "target": 1.0 + idx * 0.01,
            }
        )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
        date_col="trade_dt",
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_avoids_trade_date_isin(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "ts_code": "A", "f1": 0.0, "target": 0.0 + idx * 0.01})
        rows.append({"trade_date": date, "ts_code": "B", "f1": 1.0, "target": 1.0 + idx * 0.01})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)

    original_isin = pd.Series.isin
    call_count = 0

    def _count_isin(series, values):
        nonlocal call_count
        if series.name == "trade_date":
            call_count += 1
        return original_isin(series, values)

    monkeypatch.setattr(pd.Series, "isin", _count_isin)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))
    assert call_count == 0
