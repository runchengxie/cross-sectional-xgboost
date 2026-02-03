from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from .config_utils import read_package_text, resolve_pipeline_config, resolve_pipeline_filename


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
    if getattr(args, "format", None):
        argv += ["--format", args.format]
    if getattr(args, "out", None):
        argv += ["--out", args.out]
    holdings.main(argv)
    return 0


def _handle_init_config(args) -> int:
    filename = resolve_pipeline_filename(args.market)
    content = read_package_text("csxgb.config", filename)

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
    parser = argparse.ArgumentParser(prog="csxgb", description="Cross-sectional XGBoost CLI")
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
