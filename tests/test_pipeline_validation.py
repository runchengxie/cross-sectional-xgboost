import copy

import pytest
import yaml

from csxgb import pipeline
from csxgb.data_interface import DataInterface


def _base_config(tmp_path):
    return {
        "market": "us",
        "data": {
            "provider": "tushare",
            "start_date": "20200101",
            "end_date": "20200110",
            "cache_dir": str(tmp_path / "cache"),
            "price_col": "close",
        },
        "universe": {
            "mode": "static",
            "symbols": ["AAA", "BBB", "CCC"],
            "min_symbols_per_date": 1,
            "drop_suspended": False,
            "suspended_policy": "mark",
        },
        "fundamentals": {"enabled": False},
        "label": {
            "horizon_mode": "fixed",
            "horizon_days": 1,
            "shift_days": 0,
            "target_col": "future_return",
        },
        "features": {
            "list": ["sma_3"],
            "params": {"sma_windows": [3]},
            "cross_sectional": {"method": "none"},
        },
        "model": {
            "type": "xgb_regressor",
            "params": {
                "n_estimators": 1,
                "learning_rate": 0.1,
                "max_depth": 2,
                "objective": "reg:squarederror",
            },
            "sample_weight_mode": "none",
        },
        "eval": {
            "test_size": 0.2,
            "n_splits": 1,
            "n_quantiles": 2,
            "rebalance_frequency": "W",
            "top_k": 1,
            "signal_direction_mode": "fixed",
            "signal_direction": 1,
            "transaction_cost_bps": 0,
            "sample_on_rebalance_dates": False,
            "report_train_ic": False,
            "save_artifacts": True,
            "save_dataset": False,
            "output_dir": str(tmp_path / "runs"),
            "run_name": "validation",
            "walk_forward": {"enabled": False},
        },
        "backtest": {
            "enabled": True,
            "top_k": 1,
            "rebalance_frequency": "W",
            "transaction_cost_bps": 0,
            "long_only": True,
            "exit_mode": "rebalance",
            "exit_price_policy": "strict",
            "exit_fallback_policy": "ffill",
        },
        "live": {"enabled": False},
    }


def _write_config(tmp_path, config):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def no_client(monkeypatch):
    monkeypatch.setattr(DataInterface, "_init_client", lambda self: None)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (("backtest", "exit_mode"), "oops", "backtest.exit_mode must be one of: rebalance, label_horizon."),
        (
            ("backtest", "exit_price_policy"),
            "oops",
            "backtest.exit_price_policy must be one of: strict, ffill, delay.",
        ),
        (
            ("backtest", "exit_fallback_policy"),
            "oops",
            "backtest.exit_fallback_policy must be one of: ffill, none.",
        ),
    ],
)
def test_pipeline_backtest_validation(tmp_path, no_client, key, value, message):
    config = copy.deepcopy(_base_config(tmp_path))
    cfg = config
    for part in key[:-1]:
        cfg = cfg[part]
    cfg[key[-1]] = value
    config_path = _write_config(tmp_path, config)
    with pytest.raises(SystemExit, match=message):
        pipeline.run(str(config_path))


def test_pipeline_live_train_mode_validation(tmp_path, no_client):
    config = copy.deepcopy(_base_config(tmp_path))
    config["live"] = {"enabled": False, "train_mode": "bad"}
    config_path = _write_config(tmp_path, config)
    with pytest.raises(SystemExit, match="live.train_mode must be one of: full, train."):
        pipeline.run(str(config_path))


def test_pipeline_live_requires_artifacts(tmp_path, no_client):
    config = copy.deepcopy(_base_config(tmp_path))
    config["eval"]["save_artifacts"] = False
    config["live"] = {"enabled": True, "train_mode": "full"}
    config_path = _write_config(tmp_path, config)
    with pytest.raises(
        SystemExit,
        match="live.enabled=true requires eval.save_artifacts=true to persist holdings.",
    ):
        pipeline.run(str(config_path))
