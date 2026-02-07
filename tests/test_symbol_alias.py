import pandas as pd

from csxgb import pipeline
from csxgb.project_tools.symbols import ensure_symbol_columns


def test_ensure_symbol_columns_accepts_stock_ticker_only():
    frame = pd.DataFrame({"stock_ticker": ["AAA", " BBB ", ""], "weight": [0.3, 0.4, 0.3]})
    out = ensure_symbol_columns(frame, context="positions.csv")
    assert out["ts_code"].tolist() == ["AAA", "BBB", ""]
    assert out["stock_ticker"].tolist() == ["AAA", "BBB", ""]


def test_load_universe_by_date_accepts_stock_ticker_column(tmp_path):
    path = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "trade_date": ["20200102", "20200102", "20200103"],
            "stock_ticker": ["AAA", "AAA", "BBB"],
            "selected": [1, 1, 0],
        }
    ).to_csv(path, index=False)

    out = pipeline.load_universe_by_date(path, market="us")
    assert list(out.columns) == ["trade_date", "ts_code"]
    assert len(out) == 1
    assert out.iloc[0]["ts_code"] == "AAA"


def test_annotate_positions_window_adds_stock_ticker_alias():
    frame = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200108"],
            "entry_date": ["20200102", "20200109"],
            "ts_code": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
            "signal": [0.1, 0.2],
            "rank": [1, 2],
            "side": ["long", "long"],
        }
    )
    out = pipeline._annotate_positions_window(frame)
    assert "stock_ticker" in out.columns
    assert out["stock_ticker"].tolist() == out["ts_code"].tolist()


def test_build_rebalance_diff_includes_stock_ticker_alias():
    frame = pd.DataFrame(
        {
            "entry_date": ["20200102", "20200102", "20200109"],
            "ts_code": ["AAA", "BBB", "AAA"],
            "side": ["long", "long", "long"],
            "weight": [0.5, 0.5, 1.0],
            "signal": [0.1, 0.2, 0.3],
            "rank": [1, 2, 1],
        }
    )
    diff = pipeline._build_rebalance_diff(frame)
    assert not diff.empty
    assert "stock_ticker" in diff.columns
    assert diff["stock_ticker"].tolist() == diff["ts_code"].tolist()
