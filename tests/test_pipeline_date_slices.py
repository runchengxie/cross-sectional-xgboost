import pandas as pd

from csml.pipeline import (
    _build_trade_date_slices,
    _slice_trade_date_range,
    _slice_trade_dates,
)


def _build_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows: list[dict] = []
    for date in dates:
        rows.append({"trade_date": date, "ts_code": "A", "value": 1.0})
        rows.append({"trade_date": date, "ts_code": "B", "value": 2.0})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_slice_trade_date_range_returns_contiguous_blocks():
    frame = _build_frame()
    ordered, dates, start_rows, end_rows, _ = _build_trade_date_slices(frame)
    out = _slice_trade_date_range(ordered, start_rows, end_rows, 1, 3)
    expected_dates = [pd.to_datetime(date) for date in dates[1:4]]
    assert out["trade_date"].drop_duplicates().tolist() == expected_dates
    assert out.shape[0] == 6


def test_slice_trade_dates_contiguous_avoids_isin(monkeypatch):
    frame = _build_frame()
    ordered, dates, start_rows, end_rows, date_to_pos = _build_trade_date_slices(frame)

    original_isin = pd.Series.isin
    call_count = 0

    def _count_isin(series, values):
        nonlocal call_count
        if series.name == "trade_date":
            call_count += 1
        return original_isin(series, values)

    monkeypatch.setattr(pd.Series, "isin", _count_isin)
    out = _slice_trade_dates(ordered, start_rows, end_rows, date_to_pos, dates[2:6])
    expected_dates = [pd.to_datetime(date) for date in dates[2:6]]
    assert out["trade_date"].drop_duplicates().tolist() == expected_dates
    assert call_count == 0


def test_slice_trade_dates_non_contiguous_avoids_isin(monkeypatch):
    frame = _build_frame()
    ordered, dates, start_rows, end_rows, date_to_pos = _build_trade_date_slices(frame)
    pick_dates = [dates[0], dates[2], dates[5]]

    original_isin = pd.Series.isin
    call_count = 0

    def _count_isin(series, values):
        nonlocal call_count
        if series.name == "trade_date":
            call_count += 1
        return original_isin(series, values)

    monkeypatch.setattr(pd.Series, "isin", _count_isin)
    out = _slice_trade_dates(ordered, start_rows, end_rows, date_to_pos, pick_dates)
    assert set(out["trade_date"].drop_duplicates()) == {pd.to_datetime(date) for date in pick_dates}
    assert call_count == 0
