import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from csxgb import pipeline
from csxgb.data_interface import DataInterface


def _build_frames(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    close_map: dict[str, np.ndarray] | None = None,
    vol_map: dict[str, np.ndarray] | None = None,
    include_amount: bool = True,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    steps = np.arange(len(dates), dtype=float)
    close_map = close_map or {}
    vol_map = vol_map or {}
    for idx, symbol in enumerate(symbols):
        close = close_map.get(symbol)
        if close is None:
            close = 100.0 + steps + idx * 5.0
        vol = vol_map.get(symbol)
        if vol is None:
            vol = np.full(len(dates), 1000.0 + idx, dtype=float)
        payload = {
            "trade_date": [d.strftime("%Y%m%d") for d in dates],
            "ts_code": symbol,
            "close": np.asarray(close, dtype=float),
            "vol": np.asarray(vol, dtype=float),
        }
        if include_amount:
            payload["amount"] = payload["close"] * payload["vol"]
        frames[symbol] = pd.DataFrame(payload)
    return frames


def _run_pipeline(tmp_path, monkeypatch, config, frames, basic_df=None) -> Path:
    def fake_init_client(self):
        self.client = None

    def fake_fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return frames[symbol].copy()

    def fake_load_basic(self, symbols=None) -> pd.DataFrame:
        return basic_df.copy() if basic_df is not None else pd.DataFrame()

    monkeypatch.setattr(DataInterface, "_init_client", fake_init_client)
    monkeypatch.setattr(DataInterface, "fetch_daily", fake_fetch_daily)
    monkeypatch.setattr(DataInterface, "load_basic", fake_load_basic)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pipeline.run(str(config_path))

    output_dir = Path(config["eval"]["output_dir"])
    run_dirs = sorted(output_dir.glob(f"{config['eval']['run_name']}_*"))
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_pipeline_filters_and_fallbacks(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    symbols = ["AAA", "BBB", "CCC"]
    vol_map = {
        "AAA": np.full(len(dates), 100.0),
        "BBB": np.full(len(dates), 120.0),
        "CCC": np.array([50.0, 0.0] + [60.0] * (len(dates) - 2)),
    }
    frames = _build_frames(symbols, dates, vol_map=vol_map, include_amount=False)
    basic_df = pd.DataFrame(
        {"ts_code": symbols, "name": ["Alpha", "ST Beta", "Gamma"]}
    )

    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {
            "provider": "eodhd",
            "start_date": "20200101",
            "end_date": "20200131",
            "cache_dir": str(tmp_path / "cache"),
            "price_col": "close",
        },
        "universe": {
            "mode": "static",
            "symbols": symbols,
            "min_symbols_per_date": 1,
            "drop_suspended": True,
            "suspended_policy": "mark",
            "drop_st": True,
        },
        "fundamentals": {
            "enabled": True,
            "source": "provider",
            "required": False,
        },
        "label": {
            "horizon_mode": "fixed",
            "horizon_days": 1,
            "shift_days": 0,
            "target_col": "future_return",
        },
        "features": {
            "list": ["vol"],
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
            "n_quantiles": 2,
            "rebalance_frequency": "W",
            "top_k": 1,
            "signal_direction_mode": "fixed",
            "signal_direction": 1,
            "transaction_cost_bps": 0,
            "sample_on_rebalance_dates": False,
            "report_train_ic": False,
            "save_artifacts": True,
            "save_dataset": True,
            "output_dir": str(output_dir),
            "run_name": "filters",
            "walk_forward": {"enabled": False},
        },
        "backtest": {"enabled": False},
    }

    run_dir = _run_pipeline(tmp_path, monkeypatch, config, frames, basic_df=basic_df)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["min_symbols_per_date"] == 2
    assert summary["fundamentals"]["enabled"] is False

    dataset = pd.read_parquet(run_dir / "dataset.parquet").reset_index()
    assert "BBB" not in dataset["ts_code"].unique()
    ccc_rows = dataset[dataset["ts_code"] == "CCC"]
    assert not ccc_rows.empty
    zero_vol_rows = ccc_rows[ccc_rows["vol"] == 0.0]
    assert not zero_vol_rows.empty
    assert zero_vol_rows["is_tradable"].eq(False).all()


def test_pipeline_feature_formulas(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    symbols = ["AAA", "BBB"]
    close_map = {
        "AAA": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "BBB": np.full(len(dates), 2.0),
    }
    vol_map = {
        "AAA": np.array([10, 20, 30, 40, 50, 60], dtype=float),
        "BBB": np.full(len(dates), 5.0),
    }
    frames = _build_frames(symbols, dates, close_map=close_map, vol_map=vol_map, include_amount=True)

    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {
            "provider": "tushare",
            "start_date": "20200101",
            "end_date": "20200131",
            "cache_dir": str(tmp_path / "cache"),
            "price_col": "close",
        },
        "universe": {
            "mode": "static",
            "symbols": symbols,
            "min_symbols_per_date": 2,
            "drop_suspended": False,
        },
        "fundamentals": {"enabled": False},
        "label": {
            "horizon_mode": "fixed",
            "horizon_days": 1,
            "shift_days": 0,
            "target_col": "future_return",
        },
        "features": {
            "list": ["sma_3", "sma_3_diff", "volume_sma3_ratio", "vol"],
            "params": {"sma_windows": [3], "volume_sma_windows": [3]},
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
            "n_quantiles": 2,
            "rebalance_frequency": "W",
            "top_k": 1,
            "signal_direction_mode": "fixed",
            "signal_direction": 1,
            "transaction_cost_bps": 0,
            "sample_on_rebalance_dates": False,
            "report_train_ic": False,
            "save_artifacts": True,
            "save_dataset": True,
            "output_dir": str(output_dir),
            "run_name": "features",
            "walk_forward": {"enabled": False},
        },
        "backtest": {"enabled": False},
    }

    run_dir = _run_pipeline(tmp_path, monkeypatch, config, frames)
    dataset = pd.read_parquet(run_dir / "dataset.parquet").reset_index()
    aaa = dataset[dataset["ts_code"] == "AAA"].sort_values("trade_date").reset_index(drop=True)

    close = pd.Series(close_map["AAA"], index=dates)
    vol = pd.Series(vol_map["AAA"], index=dates)
    sma3 = close.rolling(3).mean()
    sma3_diff = sma3.pct_change()
    vol_ratio = vol / vol.rolling(3).mean()
    expected = pd.DataFrame(
        {
            "trade_date": dates,
            "sma_3": sma3,
            "sma_3_diff": sma3_diff,
            "volume_sma3_ratio": vol_ratio,
        }
    ).dropna(subset=["sma_3", "sma_3_diff", "volume_sma3_ratio"])

    np.testing.assert_allclose(aaa["sma_3"].to_numpy(), expected["sma_3"].to_numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        aaa["sma_3_diff"].to_numpy(), expected["sma_3_diff"].to_numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        aaa["volume_sma3_ratio"].to_numpy(),
        expected["volume_sma3_ratio"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
