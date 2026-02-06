import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from csxgb import pipeline
from csxgb.data_interface import DataInterface


def _build_daily_frames(symbols: list[str], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    steps = np.arange(len(dates))
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 10.0
        slope = 0.1 * (idx + 1)
        close = base + slope * steps
        vol = np.full(len(dates), 1000 + idx, dtype=float)
        amount = vol * close
        frames[symbol] = pd.DataFrame(
            {
                "trade_date": [d.strftime("%Y%m%d") for d in dates],
                "ts_code": symbol,
                "close": close,
                "vol": vol,
                "amount": amount,
            }
        )
    return frames


@pytest.mark.integration
def test_pipeline_run_offline(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    frames = _build_daily_frames(symbols, dates)

    def fake_init_client(self):
        self.client = None

    def fake_fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return frames[symbol].copy()

    def fake_load_basic(self, symbols=None) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(DataInterface, "_init_client", fake_init_client)
    monkeypatch.setattr(DataInterface, "fetch_daily", fake_fetch_daily)
    monkeypatch.setattr(DataInterface, "load_basic", fake_load_basic)

    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {
            "provider": "tushare",
            "start_date": "20200101",
            "end_date": "20200331",
            "cache_dir": str(tmp_path / "cache"),
            "price_col": "close",
        },
        "universe": {
            "mode": "static",
            "require_by_date": False,
            "symbols": symbols,
            "min_symbols_per_date": 3,
            "drop_suspended": True,
            "suspended_policy": "mark",
        },
        "fundamentals": {"enabled": False},
        "label": {
            "horizon_mode": "next_rebalance",
            "rebalance_frequency": "W",
            "horizon_days": 5,
            "shift_days": 1,
            "target_col": "future_return",
        },
        "features": {
            "list": ["sma_5"],
            "params": {"sma_windows": [5]},
            "cross_sectional": {"method": "none"},
        },
        "model": {
            "type": "xgb_regressor",
            "params": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "max_depth": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 7,
                "objective": "reg:squarederror",
            },
            "sample_weight_mode": "none",
        },
        "eval": {
            "test_size": 0.2,
            "n_splits": 2,
            "n_quantiles": 3,
            "rebalance_frequency": "W",
            "top_k": 2,
            "signal_direction_mode": "fixed",
            "signal_direction": 1,
            "transaction_cost_bps": 0,
            "sample_on_rebalance_dates": False,
            "report_train_ic": False,
            "save_artifacts": True,
            "save_dataset": True,
            "output_dir": str(output_dir),
            "run_name": "e2e",
            "walk_forward": {"enabled": False},
        },
        "backtest": {
            "enabled": True,
            "top_k": 2,
            "rebalance_frequency": "W",
            "transaction_cost_bps": 0,
            "long_only": True,
            "exit_mode": "rebalance",
            "exit_price_policy": "delay",
        },
    }

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    pipeline.run(str(config_path))

    run_dirs = list(Path(output_dir).glob("e2e_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    summary_path = run_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run"]["name"] == "e2e"
    assert summary["dataset"]["file"]

    assert (run_dir / "dataset.parquet").exists()
    assert (run_dir / "ic_test.csv").exists()
    assert (run_dir / "quantile_returns.csv").exists()
    assert (run_dir / "backtest_net.csv").exists()


@pytest.mark.integration
def test_pipeline_ic_uses_rebalance_dates(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=70, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    frames = _build_daily_frames(symbols, dates)

    def fake_init_client(self):
        self.client = None

    def fake_fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return frames[symbol].copy()

    def fake_load_basic(self, symbols=None) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(DataInterface, "_init_client", fake_init_client)
    monkeypatch.setattr(DataInterface, "fetch_daily", fake_fetch_daily)
    monkeypatch.setattr(DataInterface, "load_basic", fake_load_basic)

    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {
            "provider": "tushare",
            "start_date": "20200101",
            "end_date": "20200430",
            "cache_dir": str(tmp_path / "cache"),
            "price_col": "close",
        },
        "universe": {
            "mode": "static",
            "symbols": symbols,
            "min_symbols_per_date": 3,
            "drop_suspended": True,
            "suspended_policy": "mark",
        },
        "fundamentals": {"enabled": False},
        "label": {
            "horizon_mode": "next_rebalance",
            "rebalance_frequency": "W",
            "horizon_days": 5,
            "shift_days": 1,
            "target_col": "future_return",
        },
        "features": {
            "list": ["sma_5"],
            "params": {"sma_windows": [5]},
            "cross_sectional": {"method": "none"},
        },
        "model": {
            "type": "xgb_regressor",
            "params": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "max_depth": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "random_state": 7,
                "objective": "reg:squarederror",
            },
            "sample_weight_mode": "none",
        },
        "eval": {
            "test_size": 0.25,
            "n_splits": 2,
            "n_quantiles": 3,
            "rebalance_frequency": "W",
            "top_k": 2,
            "signal_direction_mode": "fixed",
            "signal_direction": 1,
            "transaction_cost_bps": 0,
            "sample_on_rebalance_dates": False,
            "report_train_ic": False,
            "save_artifacts": True,
            "save_dataset": False,
            "output_dir": str(output_dir),
            "run_name": "e2e_rebalance_eval",
            "walk_forward": {"enabled": False},
        },
        "backtest": {
            "enabled": False,
            "top_k": 2,
            "rebalance_frequency": "W",
            "transaction_cost_bps": 0,
            "long_only": True,
            "exit_mode": "rebalance",
        },
    }

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pipeline.run(str(config_path))

    run_dirs = list(Path(output_dir).glob("e2e_rebalance_eval_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rebalance_dates = summary["eval"]["rebalance_dates"]
    assert rebalance_dates

    ic_test = pd.read_csv(run_dir / "ic_test.csv")
    ic_dates = pd.to_datetime(ic_test["trade_date"], errors="coerce").dt.strftime("%Y%m%d").dropna().tolist()
    assert ic_dates == rebalance_dates
