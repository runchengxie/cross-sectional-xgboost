import pandas as pd
import pytest

from csxgb.project_tools import holdings


def _write_positions(run_dir, df, name="positions_by_rebalance.csv"):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    df.to_csv(path, index=False)
    return path


def test_holdings_empty_file_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame(columns=["entry_date", "ts_code", "weight"])
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="positions_by_rebalance.csv is empty."):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "2020-01-03",
            ]
        )


def test_holdings_missing_entry_date_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame({"ts_code": ["AAA"], "weight": [1.0]})
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="positions_by_rebalance.csv is missing entry_date."):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "2020-01-03",
            ]
        )


def test_holdings_unparseable_entry_date_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame({"entry_date": ["bad-date"], "ts_code": ["AAA"], "weight": [1.0]})
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="Failed to parse entry_date column."):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "2020-01-03",
            ]
        )


def test_holdings_asof_before_entries_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame(
        {
            "entry_date": ["2020-01-05"],
            "ts_code": ["AAA"],
            "weight": [1.0],
        }
    )
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="No holdings available before the requested --as-of date."):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "2020-01-04",
            ]
        )


def test_holdings_missing_ts_code_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame({"entry_date": ["2020-01-02"], "weight": [1.0]})
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="positions_by_rebalance.csv is missing ts_code."):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "2020-01-03",
            ]
        )


def test_holdings_invalid_asof_raises(tmp_path):
    run_dir = tmp_path / "run"
    df = pd.DataFrame({"entry_date": ["2020-01-02"], "ts_code": ["AAA"], "weight": [1.0]})
    _write_positions(run_dir, df)
    with pytest.raises(SystemExit, match="Invalid --as-of date: not-a-date"):
        holdings.main(
            [
                "--run-dir",
                str(run_dir),
                "--source",
                "backtest",
                "--as-of",
                "not-a-date",
            ]
        )
