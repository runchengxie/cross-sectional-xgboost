import json
from pathlib import Path

import pandas as pd
import yaml

from csml import pipeline as pipeline_mod
from csml.project_tools import run_grid


def _build_scored_data() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"])
    symbols = ["AAA", "BBB", "CCC"]
    rows = []
    for d_idx, trade_date in enumerate(dates):
        for s_idx, symbol in enumerate(symbols):
            pred = float(3 - s_idx + d_idx * 0.01)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": symbol,
                    "close": 100.0 + d_idx + s_idx,
                    "future_return": 0.01 * (3 - s_idx) + 0.001 * d_idx,
                    "pred": pred,
                    "signal_eval": pred,
                    "signal_backtest": pred,
                    "is_tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_grid_reuses_single_pipeline_run(tmp_path, monkeypatch):
    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {"provider": "tushare", "price_col": "close"},
        "label": {"target_col": "future_return", "horizon_days": 5, "shift_days": 0},
        "eval": {
            "n_quantiles": 3,
            "rebalance_frequency": "W",
            "output_dir": str(output_dir),
            "run_name": "grid_test",
        },
        "backtest": {
            "enabled": True,
            "rebalance_frequency": "W",
            "long_only": True,
            "exit_mode": "rebalance",
            "exit_price_policy": "strict",
            "exit_fallback_policy": "ffill",
            "tradable_col": "is_tradable",
        },
    }
    config_path = tmp_path / "grid_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    calls = []
    scored_data = _build_scored_data()
    rebalance_dates = [d.strftime("%Y%m%d") for d in sorted(scored_data["trade_date"].unique())]

    def fake_pipeline_run(cfg_path: str) -> None:
        calls.append(cfg_path)
        loaded = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        run_name = loaded["eval"]["run_name"]
        run_dir = Path(loaded["eval"]["output_dir"]) / f"{run_name}_20200101_000000_deadbeef"
        run_dir.mkdir(parents=True, exist_ok=True)

        scored_path = run_dir / "eval_scored.parquet"
        scored_data.to_parquet(scored_path)

        summary = {
            "run": {"output_dir": str(run_dir)},
            "data": {"min_symbols_per_date": 2},
            "label": {"horizon_days": 5},
            "eval": {
                "rebalance_frequency": "W",
                "rebalance_dates": rebalance_dates,
                "scored_file": str(scored_path),
                "scored_signal_col": "signal_eval",
                "scored_signal_backtest_col": "signal_backtest",
            },
            "backtest": {
                "enabled": True,
                "rebalance_frequency": "W",
                "rebalance_dates": rebalance_dates,
                "shift_days": 0,
                "trading_days_per_year": 252,
                "exit_price_policy": "strict",
                "exit_fallback_policy": "ffill",
                "tradable_col": "is_tradable",
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    monkeypatch.setattr(pipeline_mod, "run", fake_pipeline_run)

    output_csv = tmp_path / "grid_summary.csv"
    run_grid.main(
        [
            "--config",
            str(config_path),
            "--top-k",
            "1,2",
            "--cost-bps",
            "10,20",
            "--run-name-prefix",
            "demo",
            "--output",
            str(output_csv),
        ]
    )

    assert len(calls) == 1
    assert output_csv.exists()

    result = pd.read_csv(output_csv)
    assert len(result) == 4
    assert set(result["run_name"]) == {
        "demo_k1_bps10",
        "demo_k1_bps20",
        "demo_k2_bps10",
        "demo_k2_bps20",
    }
    assert result["status"].isin(["ok", "no_backtest"]).all()


def test_grid_supports_buffer_sweep(tmp_path, monkeypatch):
    output_dir = tmp_path / "runs"
    config = {
        "market": "us",
        "data": {"provider": "tushare", "price_col": "close"},
        "label": {"target_col": "future_return", "horizon_days": 5, "shift_days": 0},
        "eval": {
            "n_quantiles": 3,
            "rebalance_frequency": "W",
            "output_dir": str(output_dir),
            "run_name": "grid_test",
            "buffer_exit": 2,
            "buffer_entry": 1,
        },
        "backtest": {
            "enabled": True,
            "rebalance_frequency": "W",
            "long_only": True,
            "exit_mode": "rebalance",
            "exit_price_policy": "strict",
            "exit_fallback_policy": "ffill",
            "tradable_col": "is_tradable",
            "buffer_exit": 2,
            "buffer_entry": 1,
        },
    }
    config_path = tmp_path / "grid_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    calls = []
    scored_data = _build_scored_data()
    rebalance_dates = [d.strftime("%Y%m%d") for d in sorted(scored_data["trade_date"].unique())]

    def fake_pipeline_run(cfg_path: str) -> None:
        calls.append(cfg_path)
        loaded = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        run_name = loaded["eval"]["run_name"]
        run_dir = Path(loaded["eval"]["output_dir"]) / f"{run_name}_20200101_000000_deadbeef"
        run_dir.mkdir(parents=True, exist_ok=True)

        scored_path = run_dir / "eval_scored.parquet"
        scored_data.to_parquet(scored_path)

        summary = {
            "run": {"output_dir": str(run_dir)},
            "data": {"min_symbols_per_date": 2},
            "label": {"horizon_days": 5},
            "eval": {
                "rebalance_frequency": "W",
                "rebalance_dates": rebalance_dates,
                "scored_file": str(scored_path),
                "scored_signal_col": "signal_eval",
                "scored_signal_backtest_col": "signal_backtest",
            },
            "backtest": {
                "enabled": True,
                "rebalance_frequency": "W",
                "rebalance_dates": rebalance_dates,
                "shift_days": 0,
                "trading_days_per_year": 252,
                "exit_price_policy": "strict",
                "exit_fallback_policy": "ffill",
                "tradable_col": "is_tradable",
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    monkeypatch.setattr(pipeline_mod, "run", fake_pipeline_run)

    output_csv = tmp_path / "grid_summary_buffer.csv"
    run_grid.main(
        [
            "--config",
            str(config_path),
            "--top-k",
            "1",
            "--cost-bps",
            "10",
            "--buffer-exit",
            "2,4",
            "--buffer-entry",
            "1",
            "--run-name-prefix",
            "demo",
            "--output",
            str(output_csv),
        ]
    )

    assert len(calls) == 1
    result = pd.read_csv(output_csv)
    assert len(result) == 2
    assert set(result["run_name"]) == {"demo_k1_bps10_bx2_be1", "demo_k1_bps10_bx4_be1"}
    assert set(result["buffer_exit"]) == {2, 4}
    assert set(result["buffer_entry"]) == {1}
