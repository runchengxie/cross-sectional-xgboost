import json
import sys
import types

import numpy as np
import pandas as pd

from csxgb.project_tools import alloc


class _FakeInstrument:
    def __init__(self, order_book_id: str, round_lot: int):
        self.order_book_id = order_book_id
        self.round_lot = round_lot


def _install_fake_rqdatac(monkeypatch, price_map: dict[str, list[float]], lot_map: dict[str, int]):
    trading_dates = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]

    def init(**kwargs):
        return None

    def get_trading_dates(start, end, market=None):
        return trading_dates

    def get_price(
        order_book_ids,
        start_date,
        end_date,
        frequency="1d",
        fields=None,
        expect_df=True,
        market=None,
    ):
        field = fields[0] if fields else "close"
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(trading_dates), order_book_ids],
            names=["datetime", "order_book_id"],
        )
        values: list[float] = []
        for date_idx, _ in enumerate(trading_dates):
            for order_book_id in order_book_ids:
                series = price_map.get(order_book_id, [])
                value = series[date_idx] if date_idx < len(series) else np.nan
                values.append(float(value) if pd.notna(value) else np.nan)
        return pd.DataFrame({field: values}, index=index)

    def instruments(order_book_ids, market=None):
        return [
            _FakeInstrument(order_book_id, lot_map[order_book_id])
            for order_book_id in order_book_ids
        ]

    fake_module = types.SimpleNamespace(
        init=init,
        get_trading_dates=get_trading_dates,
        get_price=get_price,
        instruments=instruments,
    )
    monkeypatch.setitem(sys.modules, "rqdatac", fake_module)


def _write_positions(path, symbols: list[str], symbol_col: str = "ts_code") -> None:
    df = pd.DataFrame(
        {
            "entry_date": ["2020-01-02"] * len(symbols),
            "rebalance_date": ["2020-01-01"] * len(symbols),
            "weight": [1.0 / float(len(symbols))] * len(symbols),
            "signal": [0.1] * len(symbols),
            "rank": list(range(1, len(symbols) + 1)),
            "side": ["long"] * len(symbols),
        }
    )
    df[symbol_col] = symbols
    df.to_csv(path, index=False)


def test_alloc_from_latest_live_holdings(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    positions_path = run_dir / "positions_by_rebalance_live.csv"
    _write_positions(positions_path, ["0001.HK", "0002.HK", "0003.HK"])

    _install_fake_rqdatac(
        monkeypatch,
        price_map={
            "00001.XHKG": [49.0, 50.0],
            "00002.XHKG": [19.0, 20.0],
            "00003.XHKG": [9.0, 10.0],
        },
        lot_map={
            "00001.XHKG": 100,
            "00002.XHKG": 200,
            "00003.XHKG": 500,
        },
    )

    alloc.main(
        [
            "--run-dir",
            str(run_dir),
            "--source",
            "live",
            "--as-of",
            "2020-01-03",
            "--top-n",
            "2",
            "--cash",
            "1000000",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "live"
    assert payload["requested_top_n"] == 2
    assert payload["selected_n"] == 2
    assert payload["price_date"] == "2020-01-03"
    assert payload["cash_left"] == 0.0
    assert payload["total_gap_to_target"] == 0.0

    rows = payload["allocations"]
    assert len(rows) == 2

    row_a = rows[0]
    assert row_a["ts_code"] == "0001.HK"
    assert row_a["stock_ticker"] == "0001.HK"
    assert row_a["order_book_id"] == "00001.XHKG"
    assert row_a["round_lot"] == 100
    assert row_a["lots"] == 100
    assert row_a["shares"] == 10000
    assert row_a["est_value"] == 500000.0

    row_b = rows[1]
    assert row_b["ts_code"] == "0002.HK"
    assert row_b["stock_ticker"] == "0002.HK"
    assert row_b["order_book_id"] == "00002.XHKG"
    assert row_b["round_lot"] == 200
    assert row_b["lots"] == 125
    assert row_b["shares"] == 25000
    assert row_b["est_value"] == 500000.0


def test_alloc_positions_file_mode(tmp_path, monkeypatch, capsys):
    positions_path = tmp_path / "positions.csv"
    _write_positions(positions_path, ["0001.HK", "0002.HK"])

    _install_fake_rqdatac(
        monkeypatch,
        price_map={
            "00001.XHKG": [49.0, 50.0],
            "00002.XHKG": [19.0, 20.0],
        },
        lot_map={
            "00001.XHKG": 100,
            "00002.XHKG": 200,
        },
    )

    alloc.main(
        [
            "--positions-file",
            str(positions_path),
            "--as-of",
            "2020-01-03",
            "--top-n",
            "1",
            "--cash",
            "1000000",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "positions_file"
    assert payload["selected_n"] == 1
    assert payload["cash_left"] == 0.0
    assert payload["total_gap_to_target"] == 0.0
    row = payload["allocations"][0]
    assert row["ts_code"] == "0001.HK"
    assert row["stock_ticker"] == "0001.HK"
    assert row["lots"] == 200
    assert row["shares"] == 20000


def test_alloc_positions_file_accepts_stock_ticker(tmp_path, monkeypatch, capsys):
    positions_path = tmp_path / "positions.csv"
    _write_positions(positions_path, ["0001.HK", "0002.HK"], symbol_col="stock_ticker")

    _install_fake_rqdatac(
        monkeypatch,
        price_map={
            "00001.XHKG": [49.0, 50.0],
            "00002.XHKG": [19.0, 20.0],
        },
        lot_map={
            "00001.XHKG": 100,
            "00002.XHKG": 200,
        },
    )

    alloc.main(
        [
            "--positions-file",
            str(positions_path),
            "--as-of",
            "2020-01-03",
            "--top-n",
            "1",
            "--cash",
            "1000000",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    row = payload["allocations"][0]
    assert row["ts_code"] == "0001.HK"
    assert row["stock_ticker"] == "0001.HK"


def test_alloc_text_output_uses_chinese_labels_and_lots_after_stock_ticker(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    positions_path = run_dir / "positions_by_rebalance_live.csv"
    _write_positions(positions_path, ["0001.HK", "0002.HK"])

    _install_fake_rqdatac(
        monkeypatch,
        price_map={
            "00001.XHKG": [49.0, 50.0],
            "00002.XHKG": [19.0, 20.0],
        },
        lot_map={
            "00001.XHKG": 100,
            "00002.XHKG": 200,
        },
    )

    alloc.main(
        [
            "--run-dir",
            str(run_dir),
            "--source",
            "live",
            "--as-of",
            "2020-01-03",
            "--top-n",
            "2",
            "--cash",
            "1000000",
            "--format",
            "text",
        ]
    )

    output = capsys.readouterr().out
    assert "截至日期: 2020-01-03" in output
    assert "目标缺口合计: 0.00" in output

    lines = output.splitlines()
    header_line = next(line for line in lines if "stock_ticker" in line and "lots" in line)
    assert header_line.split()[:2] == ["stock_ticker", "lots"]
