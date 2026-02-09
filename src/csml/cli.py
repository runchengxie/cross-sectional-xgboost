from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from .config_utils import read_package_text, resolve_pipeline_config, resolve_pipeline_filename


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def _render_pct_bar(pct: float, width: int = 20) -> str:
    if pct <= 0:
        filled = 0
    elif pct >= 100:
        filled = width
    else:
        filled = int(round(width * pct / 100))
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:.2f}%"


def _coerce_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _augment_quota_entry(entry: dict) -> dict:
    bytes_used = _coerce_float(entry.get("bytes_used"))
    bytes_limit = _coerce_float(entry.get("bytes_limit"))
    if bytes_used is None or bytes_limit is None:
        return entry
    if bytes_limit <= 0:
        return entry
    used_pct = min(bytes_used / bytes_limit * 100.0, 100.0)
    remaining_pct = max(0.0, 100.0 - used_pct)
    bytes_remaining = max(bytes_limit - bytes_used, 0.0)
    entry["bytes_remaining"] = bytes_remaining
    entry["used_pct"] = round(used_pct, 2)
    entry["remaining_pct"] = round(remaining_pct, 2)
    return entry


def _augment_quota_payload(payload):
    if isinstance(payload, dict):
        return _augment_quota_entry(payload)
    if isinstance(payload, list):
        updated = []
        for entry in payload:
            if isinstance(entry, dict):
                updated.append(_augment_quota_entry(entry))
            else:
                updated.append(entry)
        return updated
    return payload


def _format_quota_entry(entry: dict, label: str | None = None) -> str:
    lines: list[str] = []
    if label:
        lines.append(label)
    if "license_type" in entry:
        lines.append(f"license_type: {entry.get('license_type')}")
    if "remaining_days" in entry:
        lines.append(f"remaining_days: {entry.get('remaining_days')}")

    bytes_used = entry.get("bytes_used")
    bytes_limit = entry.get("bytes_limit")
    bytes_remaining = entry.get("bytes_remaining")
    used_pct = entry.get("used_pct")
    remaining_pct = entry.get("remaining_pct")

    bytes_used_val = _coerce_float(bytes_used)
    bytes_limit_val = _coerce_float(bytes_limit)
    bytes_remaining_val = _coerce_float(bytes_remaining)
    used_pct_val = _coerce_float(used_pct)
    remaining_pct_val = _coerce_float(remaining_pct)

    if bytes_used_val is not None:
        lines.append(
            f"bytes_used: {_format_bytes(bytes_used_val)} ({int(bytes_used_val)} B)"
        )
    elif bytes_used is not None:
        lines.append(f"bytes_used: {bytes_used}")

    if bytes_limit_val is not None:
        lines.append(
            f"bytes_limit: {_format_bytes(bytes_limit_val)} ({int(bytes_limit_val)} B)"
        )
    elif bytes_limit is not None:
        lines.append(f"bytes_limit: {bytes_limit}")

    if bytes_remaining_val is not None:
        lines.append(
            f"bytes_remaining: {_format_bytes(bytes_remaining_val)} ({int(bytes_remaining_val)} B)"
        )
    elif bytes_remaining is not None:
        lines.append(f"bytes_remaining: {bytes_remaining}")

    if used_pct_val is not None:
        lines.append(f"used_pct: {used_pct_val:.2f}%")
    elif used_pct is not None:
        lines.append(f"used_pct: {used_pct}")

    if remaining_pct_val is not None:
        lines.append(f"remaining_pct: {remaining_pct_val:.2f}%")
    elif remaining_pct is not None:
        lines.append(f"remaining_pct: {remaining_pct}")

    if used_pct_val is not None:
        lines.append(f"usage: {_render_pct_bar(used_pct_val)} used")
    return "\n".join(lines)


def _format_quota_pretty(payload) -> str:
    if isinstance(payload, dict):
        return _format_quota_entry(payload, label="Quota usage")
    if isinstance(payload, list):
        blocks: list[str] = []
        for idx, entry in enumerate(payload, start=1):
            if isinstance(entry, dict):
                blocks.append(_format_quota_entry(entry, label=f"Quota usage #{idx}"))
            else:
                blocks.append(f"Quota usage #{idx}\n{entry}")
        return "\n\n".join(blocks)
    return str(payload)


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    resolved = resolve_pipeline_config(path)
    return resolved.data


def _init_rqdatac(args) -> object:
    try:
        import rqdatac
    except ImportError as exc:
        raise SystemExit(
            "rqdatac is not installed. Install with: pip install '.[rqdata]'"
        ) from exc

    load_dotenv()
    init_kwargs: dict = {}
    cfg = _load_config(args.config) if getattr(args, "config", None) else {}
    rq_cfg = cfg.get("data", {}).get("rqdata", {}) if isinstance(cfg, dict) else {}
    if isinstance(rq_cfg, dict) and isinstance(rq_cfg.get("init"), dict):
        init_kwargs.update(rq_cfg.get("init"))

    if getattr(args, "username", None):
        init_kwargs["username"] = args.username
    if getattr(args, "password", None):
        init_kwargs["password"] = args.password

    env_username = os.getenv("RQDATA_USERNAME") or os.getenv("RQDATA_USER")
    env_password = os.getenv("RQDATA_PASSWORD")
    if env_username and "username" not in init_kwargs:
        init_kwargs["username"] = env_username
    if env_password and "password" not in init_kwargs:
        init_kwargs["password"] = env_password

    try:
        rqdatac.init(**init_kwargs)
    except Exception as exc:
        raise SystemExit(f"rqdatac.init failed: {exc}") from exc
    return rqdatac


def _handle_run(args) -> int:
    from . import pipeline

    pipeline.run(args.config)
    return 0


def _handle_rqdata_info(args) -> int:
    rqdatac = _init_rqdatac(args)
    info = rqdatac.info()
    print(info)
    return 0


def _handle_rqdata_quota(args) -> int:
    rqdatac = _init_rqdatac(args)
    quota = rqdatac.user.get_quota()
    payload = quota
    if hasattr(quota, "to_dict"):
        try:
            payload = quota.to_dict(orient="records")
        except TypeError:
            payload = quota.to_dict()
    payload = _augment_quota_payload(payload)
    if getattr(args, "pretty", False):
        print(_format_quota_pretty(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _handle_universe_hk_connect(args) -> int:
    from .project_tools import build_hk_connect_universe

    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    argv += list(args.args or [])
    build_hk_connect_universe.main(argv)
    return 0


def _handle_universe_index_components(args) -> int:
    from .project_tools import fetch_index_components

    fetch_index_components.main(args.args)
    return 0


def _handle_tushare_verify(args) -> int:
    from .project_tools import verify_tushare_tokens

    verify_tushare_tokens.main(args.args)
    return 0


def _handle_grid(args) -> int:
    from .project_tools import run_grid

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "top_k", None):
        for entry in args.top_k:
            argv += ["--top-k", entry]
    if getattr(args, "cost_bps", None):
        for entry in args.cost_bps:
            argv += ["--cost-bps", entry]
    if getattr(args, "output", None):
        argv += ["--output", args.output]
    if getattr(args, "run_name_prefix", None):
        argv += ["--run-name-prefix", args.run_name_prefix]
    if getattr(args, "log_level", None):
        argv += ["--log-level", args.log_level]
    if getattr(args, "args", None):
        argv += list(args.args)
    run_grid.main(argv)
    return 0


def _handle_sweep_linear(args) -> int:
    from .project_tools import linear_sweep

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "run_name_prefix", None):
        argv += ["--run-name-prefix", args.run_name_prefix]
    if getattr(args, "sweeps_dir", None):
        argv += ["--sweeps-dir", args.sweeps_dir]
    if getattr(args, "tag", None):
        argv += ["--tag", args.tag]
    if getattr(args, "runs_dir", None):
        argv += ["--runs-dir", args.runs_dir]
    if getattr(args, "ridge_alpha", None):
        for entry in args.ridge_alpha:
            argv += ["--ridge-alpha", entry]
    if getattr(args, "elasticnet_alpha", None):
        for entry in args.elasticnet_alpha:
            argv += ["--elasticnet-alpha", entry]
    if getattr(args, "elasticnet_l1_ratio", None):
        for entry in args.elasticnet_l1_ratio:
            argv += ["--elasticnet-l1-ratio", entry]
    if getattr(args, "skip_ridge", None):
        argv += ["--skip-ridge"]
    if getattr(args, "skip_elasticnet", None):
        argv += ["--skip-elasticnet"]
    if getattr(args, "dry_run", None):
        argv += ["--dry-run"]
    if getattr(args, "continue_on_error", None):
        argv += ["--continue-on-error"]
    if getattr(args, "skip_summarize", None):
        argv += ["--skip-summarize"]
    if getattr(args, "summary_output", None):
        argv += ["--summary-output", args.summary_output]
    if getattr(args, "log_level", None):
        argv += ["--log-level", args.log_level]
    if getattr(args, "args", None):
        argv += list(args.args)
    linear_sweep.main(argv)
    return 0


def _handle_summarize(args) -> int:
    from .project_tools import summarize_runs

    summarize_runs.run(args)
    return 0


def _handle_holdings(args) -> int:
    from .project_tools import holdings

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "run_dir", None):
        argv += ["--run-dir", args.run_dir]
    if getattr(args, "top_k", None) is not None:
        argv += ["--top-k", str(args.top_k)]
    if getattr(args, "as_of", None):
        argv += ["--as-of", args.as_of]
    if getattr(args, "source", None):
        argv += ["--source", args.source]
    if getattr(args, "format", None):
        argv += ["--format", args.format]
    if getattr(args, "out", None):
        argv += ["--out", args.out]
    holdings.main(argv)
    return 0


def _handle_snapshot(args) -> int:
    from .project_tools import snapshot

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "run_dir", None):
        argv += ["--run-dir", args.run_dir]
    if getattr(args, "as_of", None):
        argv += ["--as-of", args.as_of]
    if getattr(args, "skip_run", None):
        argv += ["--skip-run"]
    if getattr(args, "top_k", None) is not None:
        argv += ["--top-k", str(args.top_k)]
    if getattr(args, "format", None):
        argv += ["--format", args.format]
    if getattr(args, "out", None):
        argv += ["--out", args.out]
    snapshot.main(argv)
    return 0


def _handle_alloc(args) -> int:
    from .project_tools import alloc

    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", args.config]
    if getattr(args, "run_dir", None):
        argv += ["--run-dir", args.run_dir]
    if getattr(args, "positions_file", None):
        argv += ["--positions-file", args.positions_file]
    if getattr(args, "top_k", None) is not None:
        argv += ["--top-k", str(args.top_k)]
    if getattr(args, "as_of", None):
        argv += ["--as-of", args.as_of]
    if getattr(args, "source", None):
        argv += ["--source", args.source]
    if getattr(args, "side", None):
        argv += ["--side", args.side]
    if getattr(args, "top_n", None) is not None:
        argv += ["--top-n", str(args.top_n)]
    if getattr(args, "cash", None) is not None:
        argv += ["--cash", str(args.cash)]
    if getattr(args, "buffer_bps", None) is not None:
        argv += ["--buffer-bps", str(args.buffer_bps)]
    if getattr(args, "price_field", None):
        argv += ["--price-field", args.price_field]
    if getattr(args, "price_lookback_days", None) is not None:
        argv += ["--price-lookback-days", str(args.price_lookback_days)]
    if getattr(args, "username", None):
        argv += ["--username", args.username]
    if getattr(args, "password", None):
        argv += ["--password", args.password]
    if getattr(args, "format", None):
        argv += ["--format", args.format]
    if getattr(args, "out", None):
        argv += ["--out", args.out]
    alloc.main(argv)
    return 0


def _handle_init_config(args) -> int:
    filename = resolve_pipeline_filename(args.market)
    content = read_package_text("csml.config", filename)

    if args.out:
        out_path = Path(args.out)
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / filename
        elif not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / filename
    else:
        out_dir = Path.cwd() / "config"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csml", description="Cross-sectional Machine Learning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the main training/eval/backtest pipeline")
    run.add_argument(
        "--config",
        help="Path to YAML config or built-in name (default/cn/hk/us).",
    )
    run.set_defaults(func=_handle_run)

    rqdata = subparsers.add_parser("rqdata", help="RQData utilities")
    rq_sub = rqdata.add_subparsers(dest="rq_command", required=True)

    rq_info = rq_sub.add_parser("info", help="Show rqdatac login/info")
    rq_info.add_argument("--config", help="Optional config path to load rqdata.init")
    rq_info.add_argument("--username", help="Override RQData username")
    rq_info.add_argument("--password", help="Override RQData password")
    rq_info.set_defaults(func=_handle_rqdata_info)

    rq_quota = rq_sub.add_parser("quota", help="Show rqdatac quota usage")
    rq_quota.add_argument("--config", help="Optional config path to load rqdata.init")
    rq_quota.add_argument("--username", help="Override RQData username")
    rq_quota.add_argument("--password", help="Override RQData password")
    rq_quota.add_argument(
        "--pretty",
        action="store_true",
        help="Show human-friendly output with percent and progress bar",
    )
    rq_quota.set_defaults(func=_handle_rqdata_quota)

    universe = subparsers.add_parser("universe", help="Universe construction helpers")
    uni_sub = universe.add_subparsers(dest="uni_command", required=True)

    hk = uni_sub.add_parser("hk-connect", help="Build HK Connect universe")
    hk.add_argument("--config", help="YAML config path (optional).")
    hk.add_argument("args", nargs=argparse.REMAINDER)
    hk.set_defaults(func=_handle_universe_hk_connect)

    index_components = uni_sub.add_parser(
        "index-components", help="Fetch index constituents (TuShare)"
    )
    index_components.add_argument("args", nargs=argparse.REMAINDER)
    index_components.set_defaults(func=_handle_universe_index_components)

    tushare = subparsers.add_parser("tushare", help="TuShare utilities")
    tu_sub = tushare.add_subparsers(dest="tushare_command", required=True)

    verify = tu_sub.add_parser("verify-token", help="Verify TuShare token(s)")
    verify.add_argument("args", nargs=argparse.REMAINDER)
    verify.set_defaults(func=_handle_tushare_verify)

    grid = subparsers.add_parser("grid", help="Run Top-K × cost grid and summarize results")
    from .project_tools import run_grid

    run_grid.add_grid_args(grid)
    grid.set_defaults(func=_handle_grid)

    sweep_linear = subparsers.add_parser(
        "sweep-linear",
        help="Run ridge/elasticnet hyper-parameter sweep and auto summarize",
    )
    from .project_tools import linear_sweep

    linear_sweep.add_linear_sweep_args(sweep_linear)
    sweep_linear.set_defaults(func=_handle_sweep_linear)

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate saved runs into a summary CSV"
    )
    from .project_tools import summarize_runs

    summarize_runs.add_summarize_args(summarize)
    summarize.set_defaults(func=_handle_summarize)

    holdings = subparsers.add_parser("holdings", help="Show latest holdings from saved runs")
    holdings.add_argument(
        "--config",
        help="Pipeline config path or built-in name (default: default).",
    )
    holdings.add_argument(
        "--run-dir",
        help="Explicit run directory to read (overrides --config).",
    )
    holdings.add_argument(
        "--top-k",
        type=int,
        help="Optional Top-K filter when selecting the latest run.",
    )
    holdings.add_argument(
        "--as-of",
        default="t-1",
        help="As-of date (YYYYMMDD, YYYY-MM-DD, today, t-1). Default: t-1.",
    )
    holdings.add_argument(
        "--source",
        default="auto",
        choices=["auto", "backtest", "live"],
        help="Positions source (auto/backtest/live). Default: auto.",
    )
    holdings.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "json"],
        help="Output format (text/csv/json). Default: text.",
    )
    holdings.add_argument(
        "--out",
        help="Optional output path (default: stdout).",
    )
    holdings.set_defaults(func=_handle_holdings)

    alloc = subparsers.add_parser(
        "alloc",
        help="Compute equal-weight lot sizing from latest holdings using rqdata prices.",
    )
    alloc.add_argument(
        "--config",
        help="Pipeline config path or built-in name (default: default).",
    )
    alloc.add_argument(
        "--run-dir",
        help="Explicit run directory to read (overrides --config).",
    )
    alloc.add_argument(
        "--positions-file",
        help="Explicit positions CSV path (overrides --config/--run-dir).",
    )
    alloc.add_argument(
        "--top-k",
        type=int,
        help="Optional Top-K filter when selecting the latest run.",
    )
    alloc.add_argument(
        "--as-of",
        default="t-1",
        help="As-of date (YYYYMMDD, YYYY-MM-DD, today, t-1). Default: t-1.",
    )
    alloc.add_argument(
        "--source",
        default="auto",
        choices=["auto", "backtest", "live"],
        help="Positions source (auto/backtest/live). Default: auto.",
    )
    alloc.add_argument(
        "--side",
        default="long",
        choices=["long", "short", "all"],
        help="Select side for allocation (long/short/all). Default: long.",
    )
    alloc.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of names to allocate equally from sorted holdings. Default: 20.",
    )
    alloc.add_argument(
        "--cash",
        type=float,
        default=1_000_000,
        help="Total portfolio cash for sizing. Default: 1000000.",
    )
    alloc.add_argument(
        "--buffer-bps",
        type=float,
        default=0.0,
        help="Cash buffer in bps reserved from investment. Default: 0.",
    )
    alloc.add_argument(
        "--price-field",
        default="close",
        help="Price field fetched from rqdata.get_price. Default: close.",
    )
    alloc.add_argument(
        "--price-lookback-days",
        type=int,
        default=20,
        help="Price lookback window in calendar days before price date. Default: 20.",
    )
    alloc.add_argument("--username", help="Override RQData username.")
    alloc.add_argument("--password", help="Override RQData password.")
    alloc.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "json"],
        help="Output format (text/csv/json). Default: text.",
    )
    alloc.add_argument(
        "--out",
        help="Optional output path (default: stdout).",
    )
    alloc.set_defaults(func=_handle_alloc)

    snapshot = subparsers.add_parser(
        "snapshot", help="Run a live snapshot and emit latest holdings"
    )
    snapshot.add_argument(
        "--config",
        help="Pipeline config path or built-in name.",
    )
    snapshot.add_argument(
        "--run-dir",
        help="Use an existing run directory (skips pipeline run).",
    )
    snapshot.add_argument(
        "--as-of",
        default="t-1",
        help="As-of date (YYYYMMDD, YYYY-MM-DD, today, t-1). Default: t-1.",
    )
    snapshot.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running the pipeline and only emit holdings from the latest run.",
    )
    snapshot.add_argument(
        "--top-k",
        type=int,
        help="Optional Top-K filter when selecting the latest run.",
    )
    snapshot.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "json"],
        help="Output format (text/csv/json). Default: text.",
    )
    snapshot.add_argument(
        "--out",
        help="Optional output path (default: stdout).",
    )
    snapshot.set_defaults(func=_handle_snapshot)

    init_cfg = subparsers.add_parser(
        "init-config", help="Export a packaged config template to the filesystem"
    )
    init_cfg.add_argument(
        "--market",
        default="default",
        help="Template to export (default/cn/hk/us).",
    )
    init_cfg.add_argument(
        "--out",
        help="Output path or directory (default: ./config/<template>.yml).",
    )
    init_cfg.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    init_cfg.set_defaults(func=_handle_init_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return int(func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
