import csv
from pathlib import Path

import pytest
import yaml

from csml import pipeline as pipeline_mod
from csml.project_tools import linear_sweep
from csml.project_tools import summarize_runs as summarize_tool


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_linear_sweep_generates_configs_runs_and_summarizes(tmp_path, monkeypatch):
    base_cfg = {
        "market": "hk",
        "model": {
            "type": "xgb_regressor",
            "params": {"random_state": 7},
            "sample_weight_mode": "date_equal",
        },
        "eval": {
            "output_dir": "out/runs",
            "run_name": "base",
        },
    }
    config_path = tmp_path / "base.yml"
    config_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    run_calls: list[str] = []

    def fake_pipeline_run(cfg_path: str) -> None:
        run_calls.append(cfg_path)

    summarize_calls: list[list[str]] = []

    def fake_summarize_main(argv: list[str]) -> None:
        summarize_calls.append(list(argv))
        output_idx = argv.index("--output") + 1
        output_path = Path(argv[output_idx])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("run_name\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_mod, "run", fake_pipeline_run)
    monkeypatch.setattr(summarize_tool, "main", fake_summarize_main)

    runs_dir = tmp_path / "runs"
    sweeps_dir = tmp_path / "sweeps"
    linear_sweep.main(
        [
            "--config",
            str(config_path),
            "--run-name-prefix",
            "demo_",
            "--sweeps-dir",
            str(sweeps_dir),
            "--tag",
            "unit_test",
            "--runs-dir",
            str(runs_dir),
            "--ridge-alpha",
            "0.1,1",
            "--elasticnet-alpha",
            "0.2",
            "--elasticnet-l1-ratio",
            "0.3,0.7",
        ]
    )

    assert len(run_calls) == 4

    sweep_dir = sweeps_dir / "unit_test"
    jobs_csv = sweep_dir / "jobs.csv"
    results_csv = sweep_dir / "run_results.csv"
    summary_csv = sweep_dir / "runs_summary.csv"
    assert jobs_csv.exists()
    assert results_csv.exists()
    assert summary_csv.exists()

    jobs = _read_csv(jobs_csv)
    assert len(jobs) == 4
    assert [row["run_name"] for row in jobs] == [
        "demo_ridge_a0.1",
        "demo_ridge_a1",
        "demo_en_a0.2_l0.3",
        "demo_en_a0.2_l0.7",
    ]

    ridge_cfg = yaml.safe_load((sweep_dir / "configs" / "ridge_a0.1.yml").read_text(encoding="utf-8"))
    assert ridge_cfg["model"]["type"] == "ridge"
    assert ridge_cfg["model"]["params"]["alpha"] == pytest.approx(0.1)
    assert ridge_cfg["model"]["params"]["random_state"] == 7
    assert ridge_cfg["eval"]["run_name"] == "demo_ridge_a0.1"
    assert ridge_cfg["eval"]["output_dir"] == str(runs_dir)

    en_cfg = yaml.safe_load(
        (sweep_dir / "configs" / "elasticnet_a0.2_l0.7.yml").read_text(encoding="utf-8")
    )
    assert en_cfg["model"]["type"] == "elasticnet"
    assert en_cfg["model"]["params"]["alpha"] == pytest.approx(0.2)
    assert en_cfg["model"]["params"]["l1_ratio"] == pytest.approx(0.7)
    assert en_cfg["eval"]["run_name"] == "demo_en_a0.2_l0.7"

    results = _read_csv(results_csv)
    assert len(results) == 4
    assert {row["status"] for row in results} == {"ok"}

    assert len(summarize_calls) == 1
    summarize_argv = summarize_calls[0]
    assert summarize_argv[summarize_argv.index("--runs-dir") + 1] == str(runs_dir.resolve())
    assert summarize_argv[summarize_argv.index("--run-name-prefix") + 1] == "demo_"
    assert summarize_argv[summarize_argv.index("--output") + 1] == str(summary_csv.resolve())
    assert "--since" in summarize_argv


def test_linear_sweep_stops_on_first_failure(tmp_path, monkeypatch):
    base_cfg = {
        "model": {"type": "xgb_regressor", "params": {"random_state": 42}},
        "eval": {"output_dir": str(tmp_path / "runs"), "run_name": "base"},
    }
    config_path = tmp_path / "base.yml"
    config_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    call_count = {"value": 0}

    def fake_pipeline_run(_cfg_path: str) -> None:
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise SystemExit("boom")

    monkeypatch.setattr(pipeline_mod, "run", fake_pipeline_run)

    with pytest.raises(SystemExit, match="Sweep stopped on first failure"):
        linear_sweep.main(
            [
                "--config",
                str(config_path),
                "--sweeps-dir",
                str(tmp_path / "sweeps"),
                "--tag",
                "stop_case",
                "--skip-elasticnet",
                "--ridge-alpha",
                "0.1,1",
                "--skip-summarize",
            ]
        )

    results_csv = tmp_path / "sweeps" / "stop_case" / "run_results.csv"
    rows = _read_csv(results_csv)
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "failed"
