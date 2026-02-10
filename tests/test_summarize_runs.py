import json

import pandas as pd
import pytest
import yaml

from csml.project_tools import summarize_runs


def _write_run(run_dir, summary, config) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.used.yml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise AssertionError(f"Cannot parse bool value: {value}")


def test_summarize_runs_collects_metrics_and_flags(tmp_path):
    runs_dir = tmp_path / "runs"

    run_a = runs_dir / "alpha_20260101_120000_deadbeef"
    summary_a = {
        "run": {
            "name": "alpha",
            "timestamp": "20260101_120000",
            "config_hash": "deadbeef",
            "output_dir": str(run_a),
        },
        "data": {
            "market": "hk",
            "provider": "rqdata",
            "start_date": "20200101",
            "end_date": "20251231",
        },
        "universe": {"mode": "pit"},
        "label": {"horizon_days": 20, "shift_days": 1},
        "eval": {
            "ic": {"mean": 0.03, "ir": 0.4},
            "long_short": 0.02,
            "turnover_mean": 0.65,
            "buffer_exit": 2,
            "buffer_entry": 1,
            "rebalance_frequency": "M",
        },
        "backtest": {
            "enabled": True,
            "top_k": 5,
            "transaction_cost_bps": 15.0,
            "rebalance_frequency": "M",
            "buffer_exit": 2,
            "buffer_entry": 1,
            "stats": {
                "periods": 30,
                "total_return": 0.6,
                "ann_return": 0.5,
                "ann_vol": 0.25,
                "sharpe": 2.0,
                "max_drawdown": -0.1,
                "avg_turnover": 0.5,
                "avg_cost_drag": 0.01,
            },
        },
    }
    config_a = {
        "market": "hk",
        "data": {"provider": "rqdata", "start_date": "20200101", "end_date": "20251231"},
        "universe": {"mode": "pit"},
        "label": {"horizon_days": 20, "shift_days": 1},
        "eval": {"top_k": 5, "transaction_cost_bps": 15.0, "buffer_exit": 2, "buffer_entry": 1},
        "backtest": {"enabled": True, "top_k": 5, "buffer_exit": 2, "buffer_entry": 1},
    }
    _write_run(run_a, summary_a, config_a)

    run_b = runs_dir / "beta_20260102_130000_abcd1234"
    summary_b = {
        "data": {"market": "us"},
        "eval": {"ic": {"mean": 0.01, "ir": 0.2}, "long_short": -0.03},
        "backtest": {"enabled": False},
    }
    config_b = {
        "market": "us",
        "data": {"provider": "eodhd", "start_date": "20200101", "end_date": "20201231"},
        "universe": {"mode": "static"},
        "label": {"horizon_days": 10, "shift_days": 2},
        "eval": {
            "top_k": 8,
            "transaction_cost_bps": 12.0,
            "buffer_exit": 1,
            "buffer_entry": 0,
            "rebalance_frequency": "W",
        },
        "backtest": {
            "enabled": False,
            "top_k": 8,
            "rebalance_frequency": "W",
            "buffer_exit": 1,
            "buffer_entry": 0,
        },
    }
    _write_run(run_b, summary_b, config_b)

    output_csv = tmp_path / "runs_summary.csv"
    summarize_runs.main(["--runs-dir", str(runs_dir), "--output", str(output_csv)])

    result = pd.read_csv(output_csv)
    assert len(result) == 2

    row_a = result[result["run_name"] == "alpha"].iloc[0]
    assert row_a["status"] == "ok"
    assert row_a["market"] == "hk"
    assert int(row_a["backtest_top_k"]) == 5
    assert _as_bool(row_a["flag_short_sample"]) is False
    assert _as_bool(row_a["flag_negative_long_short"]) is False
    assert _as_bool(row_a["flag_high_turnover"]) is False
    assert _as_bool(row_a["flag_relative_end_date"]) is False
    assert float(row_a["score"]) == pytest.approx(1.85)

    row_b = result[result["run_name"] == "beta"].iloc[0]
    assert row_b["status"] == "no_backtest"
    assert row_b["run_timestamp"] == "20260102_130000"
    assert row_b["config_hash"] == "abcd1234"
    assert int(row_b["eval_top_k"]) == 8
    assert row_b["transaction_cost_bps"] == pytest.approx(12.0)
    assert _as_bool(row_b["flag_negative_long_short"]) is True


def test_summarize_runs_default_output_and_recursive_scan(tmp_path):
    runs_dir = tmp_path / "nested_runs"
    run_dir = runs_dir / "batch_a" / "gamma_20260103_140000_1234abcd"

    summary = {
        "run": {"name": "gamma", "timestamp": "20260103_140000", "config_hash": "1234abcd"},
        "data": {"market": "cn", "provider": "tushare"},
        "eval": {"ic": {"mean": 0.02, "ir": 0.3}, "long_short": 0.01},
        "backtest": {
            "enabled": True,
            "stats": {
                "periods": 12,
                "total_return": 0.1,
                "ann_return": 0.11,
                "ann_vol": 0.2,
                "sharpe": 0.5,
                "max_drawdown": -0.2,
                "avg_turnover": 0.8,
                "avg_cost_drag": 0.01,
            },
        },
    }
    config = {"market": "cn", "eval": {"top_k": 10}, "backtest": {"top_k": 10}}
    _write_run(run_dir, summary, config)

    summarize_runs.main(["--runs-dir", str(runs_dir), "--runs-dir", str(tmp_path / "missing")])

    output_csv = runs_dir / "runs_summary.csv"
    assert output_csv.exists()

    result = pd.read_csv(output_csv)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["run_name"] == "gamma"
    assert _as_bool(row["flag_short_sample"]) is True
    assert _as_bool(row["flag_high_turnover"]) is True


def test_summarize_runs_filters_prefix_since_latest_n(tmp_path):
    runs_dir = tmp_path / "runs"

    def _mk(run_name: str, timestamp: str, config_hash: str) -> None:
        run_dir = runs_dir / f"{run_name}_{timestamp}_{config_hash}"
        summary = {
            "run": {
                "name": run_name,
                "timestamp": timestamp,
                "config_hash": config_hash,
            },
            "data": {"market": "hk", "provider": "rqdata"},
            "eval": {"ic": {"mean": 0.01, "ir": 0.1}, "long_short": 0.01},
            "backtest": {
                "stats": {
                    "periods": 12,
                    "total_return": 0.1,
                    "ann_return": 0.1,
                    "ann_vol": 0.2,
                    "sharpe": 0.5,
                    "max_drawdown": -0.2,
                    "avg_turnover": 0.6,
                    "avg_cost_drag": 0.01,
                }
            },
        }
        config = {"market": "hk", "eval": {"top_k": 5}, "backtest": {"top_k": 5}}
        _write_run(run_dir, summary, config)

    _mk("hk_grid_base", "20260101_010101", "11111111")
    _mk("hk_grid_base", "20260103_010101", "22222222")
    _mk("manual_run", "20260104_010101", "33333333")

    output_csv = tmp_path / "filtered.csv"
    summarize_runs.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_csv),
            "--run-name-prefix",
            "hk_grid",
            "--since",
            "2026-01-02",
            "--latest-n",
            "1",
        ]
    )

    result = pd.read_csv(output_csv)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["run_name"] == "hk_grid_base"
    assert row["run_timestamp"] == "20260103_010101"
    assert int(row["config_hash"]) == 22222222


def test_summarize_runs_latest_n_validation(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "alpha_20260101_120000_deadbeef"
    _write_run(run_dir, {"run": {"name": "alpha", "timestamp": "20260101_120000"}}, {"market": "hk"})

    with pytest.raises(SystemExit, match="--latest-n must be a positive integer."):
        summarize_runs.main(
            [
                "--runs-dir",
                str(runs_dir),
                "--latest-n",
                "0",
            ]
        )


def test_summarize_runs_marks_relative_end_date_from_config(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "alpha_20260101_120000_deadbeef"
    summary = {
        "run": {"name": "alpha", "timestamp": "20260101_120000", "config_hash": "deadbeef"},
        "data": {"market": "hk", "provider": "rqdata", "end_date": "20260131"},
        "eval": {"long_short": 0.01},
        "backtest": {"stats": {"periods": 24, "avg_turnover": 0.6, "sharpe": 1.0}},
    }
    config = {"market": "hk", "data": {"end_date": "today"}}
    _write_run(run_dir, summary, config)

    output_csv = tmp_path / "summary.csv"
    summarize_runs.main(["--runs-dir", str(runs_dir), "--output", str(output_csv)])

    result = pd.read_csv(output_csv)
    row = result.iloc[0]
    assert str(row["data_end_date"]) == "20260131"
    assert row["data_end_date_config"] == "today"
    assert _as_bool(row["flag_relative_end_date"]) is True


def test_summarize_runs_exclude_flags_and_sort_by_score(tmp_path):
    runs_dir = tmp_path / "runs"

    def _mk(
        run_name: str,
        timestamp: str,
        *,
        periods: int,
        turnover: float,
        sharpe: float,
        end_date_cfg: str,
    ) -> None:
        run_dir = runs_dir / f"{run_name}_{timestamp}_deadbeef"
        summary = {
            "run": {"name": run_name, "timestamp": timestamp, "config_hash": "deadbeef"},
            "data": {"market": "hk", "provider": "rqdata", "end_date": "20260131"},
            "eval": {"long_short": 0.02},
            "backtest": {
                "stats": {
                    "periods": periods,
                    "sharpe": sharpe,
                    "max_drawdown": -0.1,
                    "avg_turnover": turnover,
                    "avg_cost_drag": 0.01,
                }
            },
        }
        config = {"market": "hk", "data": {"end_date": end_date_cfg}}
        _write_run(run_dir, summary, config)

    _mk("alpha", "20260101_120000", periods=30, turnover=0.6, sharpe=1.0, end_date_cfg="today")
    _mk("beta", "20260102_120000", periods=30, turnover=0.8, sharpe=3.0, end_date_cfg="20251231")
    _mk("gamma", "20260103_120000", periods=28, turnover=0.5, sharpe=2.0, end_date_cfg="20251231")

    output_csv = tmp_path / "filtered.csv"
    summarize_runs.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(output_csv),
            "--exclude-flag-short-sample",
            "--exclude-flag-high-turnover",
            "--exclude-flag-relative-end-date",
            "--sort-by",
            "score",
        ]
    )

    result = pd.read_csv(output_csv)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["run_name"] == "gamma"
