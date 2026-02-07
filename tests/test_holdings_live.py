import json

import pandas as pd

from csxgb.project_tools import holdings


def _write_positions(
    path,
    entry_date="2020-01-02",
    rebalance_date="2020-01-01",
    symbol_col="ts_code",
):
    df = pd.DataFrame(
        {
            "entry_date": [entry_date],
            "rebalance_date": [rebalance_date],
            "weight": [1.0],
            "signal": [0.1],
            "rank": [1],
        }
    )
    df[symbol_col] = ["0001.HK"]
    df.to_csv(path, index=False)


def test_holdings_live_uses_summary_path(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    positions_path = run_dir / "positions_by_rebalance_live.csv"
    _write_positions(positions_path)
    summary = {
        "data": {"end_date": "20200103"},
        "live": {
            "enabled": True,
            "positions_file": str(positions_path),
            "current_file": None,
        },
        "positions": {
            "by_rebalance_file": str(positions_path),
            "current_file": None,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    holdings.main(
        [
            "--run-dir",
            str(run_dir),
            "--source",
            "live",
            "--as-of",
            "2020-01-03",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "live"
    assert payload["positions_file"].endswith("positions_by_rebalance_live.csv")
    assert payload["run_dir"].endswith("run")


def test_holdings_auto_prefers_live_file(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    live_path = run_dir / "positions_by_rebalance_live.csv"
    backtest_path = run_dir / "positions_by_rebalance.csv"
    _write_positions(live_path, entry_date="2020-01-02")
    _write_positions(backtest_path, entry_date="2020-01-01")

    holdings.main(
        [
            "--run-dir",
            str(run_dir),
            "--source",
            "auto",
            "--as-of",
            "2020-01-03",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "live"
    assert payload["positions_file"].endswith("positions_by_rebalance_live.csv")


def test_holdings_accepts_stock_ticker_column(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    positions_path = run_dir / "positions_by_rebalance_live.csv"
    _write_positions(positions_path, symbol_col="stock_ticker")

    holdings.main(
        [
            "--run-dir",
            str(run_dir),
            "--source",
            "live",
            "--as-of",
            "2020-01-03",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    row = payload["holdings"][0]
    assert row["stock_ticker"] == "0001.HK"
    assert row["ts_code"] == "0001.HK"
