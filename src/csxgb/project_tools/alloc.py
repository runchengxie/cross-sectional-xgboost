from __future__ import annotations

import argparse
import io
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config_utils import resolve_pipeline_config
from ..data_providers import normalize_market
from . import holdings
from .symbols import ensure_symbol_columns


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    resolved = resolve_pipeline_config(path)
    return resolved.data


def _init_rqdatac(
    config_path: str | None,
    username: str | None,
    password: str | None,
):
    try:
        import rqdatac
    except ImportError as exc:
        raise SystemExit(
            "rqdatac is not installed. Install with: pip install '.[rqdata]'"
        ) from exc

    load_dotenv()
    init_kwargs: dict = {}
    cfg = _load_config(config_path)
    rq_cfg = cfg.get("data", {}).get("rqdata", {}) if isinstance(cfg, dict) else {}
    if isinstance(rq_cfg, dict) and isinstance(rq_cfg.get("init"), dict):
        init_kwargs.update(rq_cfg.get("init"))

    if username:
        init_kwargs["username"] = username
    if password:
        init_kwargs["password"] = password

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


def _resolve_market(cfg: dict, symbols: list[str]) -> str | None:
    data_cfg = cfg.get("data") if isinstance(cfg, dict) else None
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    rq_cfg = data_cfg.get("rqdata") if isinstance(data_cfg, dict) else None

    rq_market = rq_cfg.get("market") if isinstance(rq_cfg, dict) else None
    if rq_market:
        return normalize_market(rq_market)
    cfg_market = cfg.get("market") if isinstance(cfg, dict) else None
    if cfg_market:
        return normalize_market(cfg_market)
    data_market = data_cfg.get("market") if isinstance(data_cfg, dict) else None
    if data_market:
        return normalize_market(data_market)

    inferred: set[str] = set()
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text.endswith(".HK") or text.endswith(".XHKG"):
            inferred.add("hk")
        elif text.endswith(".SH") or text.endswith(".SZ"):
            inferred.add("cn")
        elif text.endswith(".XSHG") or text.endswith(".XSHE"):
            inferred.add("cn")
    if len(inferred) == 1:
        return next(iter(inferred))
    return None


def _to_rq_order_book_id(symbol: str, market: str | None) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return text
    if text.endswith(".XHKG") or text.endswith(".XSHG") or text.endswith(".XSHE"):
        return text
    if text.endswith(".SH"):
        return f"{text[:-3]}.XSHG"
    if text.endswith(".SZ"):
        return f"{text[:-3]}.XSHE"

    if text.endswith(".HK") or market == "hk":
        if text.endswith(".HK"):
            text = text[:-3]
        if text.endswith(".XHKG"):
            text = text[:-5]
        if text.isdigit():
            text = text.zfill(5)
        return f"{text}.XHKG"

    return text


def _resolve_price_date(rqdatac, as_of: pd.Timestamp, market: str | None) -> pd.Timestamp:
    get_trading_dates = getattr(rqdatac, "get_trading_dates", None)
    if get_trading_dates is None:
        return as_of
    start = (as_of - pd.Timedelta(days=366)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    kwargs = {"market": market} if market else {}
    try:
        trading_dates = get_trading_dates(start, end, **kwargs)
    except TypeError:
        trading_dates = get_trading_dates(start, end)
    except Exception:
        return as_of
    if not trading_dates:
        return as_of
    parsed = pd.to_datetime(list(trading_dates), errors="coerce")
    parsed = [pd.Timestamp(ts).normalize() for ts in parsed if pd.notna(ts)]
    parsed = [ts for ts in parsed if ts <= as_of]
    if not parsed:
        return as_of
    return max(parsed)


def _extract_price_wide_frame(
    payload,
    field: str,
    order_book_ids: list[str],
) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, pd.Series):
        if isinstance(payload.index, pd.MultiIndex):
            if "order_book_id" in payload.index.names:
                wide = payload.unstack("order_book_id")
            else:
                wide = payload.unstack(level=0)
        else:
            wide = payload.to_frame(name=order_book_ids[0] if order_book_ids else "value")
    elif isinstance(payload, pd.DataFrame):
        if payload.empty:
            return payload
        if isinstance(payload.index, pd.MultiIndex):
            if field in payload.columns:
                values = payload[field]
            elif payload.shape[1] == 1:
                values = payload.iloc[:, 0]
            else:
                raise SystemExit(f"get_price result is missing '{field}' column.")
            if "order_book_id" in payload.index.names:
                wide = values.unstack("order_book_id")
            else:
                wide = values.unstack(level=0)
        else:
            if set(order_book_ids).issubset(payload.columns):
                wide = payload[order_book_ids].copy()
            elif field in payload.columns and len(order_book_ids) == 1:
                wide = payload[[field]].rename(columns={field: order_book_ids[0]})
            elif {"order_book_id", field}.issubset(payload.columns):
                date_col = None
                for candidate in ("date", "trade_date", "datetime", "time"):
                    if candidate in payload.columns:
                        date_col = candidate
                        break
                if not date_col:
                    raise SystemExit(
                        "Unable to parse get_price output: missing date column for long-form frame."
                    )
                wide = payload.pivot(index=date_col, columns="order_book_id", values=field)
            elif len(order_book_ids) == 1 and payload.shape[1] == 1:
                wide = payload.copy()
                wide.columns = [order_book_ids[0]]
            else:
                raise SystemExit("Unexpected get_price output format.")
    else:
        raise SystemExit("Unexpected get_price output type.")

    wide = wide.copy()
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    wide = wide[wide.index.notna()]
    return wide.sort_index()


def _fetch_latest_price_map(
    rqdatac,
    order_book_ids: list[str],
    *,
    field: str,
    start_date: str,
    end_date: str,
    market: str | None,
) -> dict[str, float]:
    kwargs = {
        "frequency": "1d",
        "fields": [field],
        "expect_df": True,
    }
    if market:
        kwargs["market"] = market
    try:
        payload = rqdatac.get_price(order_book_ids, start_date, end_date, **kwargs)
    except TypeError:
        kwargs.pop("expect_df", None)
        payload = rqdatac.get_price(order_book_ids, start_date, end_date, **kwargs)
    if payload is None:
        raise SystemExit("rqdatac.get_price returned no data.")

    wide = _extract_price_wide_frame(payload, field, order_book_ids)
    if wide.empty:
        raise SystemExit("rqdatac.get_price returned an empty frame.")

    price_map: dict[str, float] = {}
    missing: list[str] = []
    for order_book_id in order_book_ids:
        if order_book_id not in wide.columns:
            missing.append(order_book_id)
            continue
        values = pd.to_numeric(wide[order_book_id], errors="coerce").dropna()
        if values.empty:
            missing.append(order_book_id)
            continue
        price_map[order_book_id] = float(values.iloc[-1])
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing '{field}' price for {len(missing)} symbol(s): {joined}."
        )
    return price_map


def _fetch_round_lot_map(
    rqdatac,
    order_book_ids: list[str],
    market: str | None,
) -> dict[str, int]:
    kwargs = {"market": market} if market else {}
    try:
        instruments = rqdatac.instruments(order_book_ids, **kwargs)
    except TypeError:
        instruments = rqdatac.instruments(order_book_ids)

    if not isinstance(instruments, list):
        instruments = [instruments]
    lot_map: dict[str, int] = {}
    for ins in instruments:
        if ins is None:
            continue
        order_book_id = getattr(ins, "order_book_id", None)
        if order_book_id is None:
            continue
        round_lot = getattr(ins, "round_lot", None)
        try:
            lot_map[str(order_book_id)] = max(1, int(round_lot))
        except (TypeError, ValueError):
            continue
    for order_book_id in order_book_ids:
        lot_map.setdefault(order_book_id, 1)
    return lot_map


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    lines = [header_line, sep_line]
    for row in rows:
        lines.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _select_from_positions_file(
    positions_path: Path,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not positions_path.exists():
        raise SystemExit(f"Positions file not found: {positions_path}")
    df = pd.read_csv(positions_path)
    if df.empty:
        raise SystemExit(f"{positions_path.name} is empty.")
    if "entry_date" not in df.columns:
        raise SystemExit(f"{positions_path.name} is missing entry_date.")
    entry_dates = holdings._parse_date_column(df["entry_date"])
    if entry_dates.isna().all():
        raise SystemExit("Failed to parse entry_date column.")
    eligible = entry_dates <= as_of
    if not eligible.any():
        raise SystemExit("No holdings available before the requested --as-of date.")
    latest_entry = entry_dates[eligible].max()
    selection = df[entry_dates == latest_entry].copy()
    selection = ensure_symbol_columns(selection, context=positions_path.name)
    if selection.empty:
        raise SystemExit("No holdings found for the latest entry date.")
    return selection, latest_entry


def _load_holdings_payload(args) -> dict:
    argv: list[str] = ["--as-of", args.as_of, "--source", args.source, "--format", "json"]
    if args.config:
        argv += ["--config", args.config]
    if args.run_dir:
        argv += ["--run-dir", args.run_dir]
    if args.top_k is not None:
        argv += ["--top-k", str(args.top_k)]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        holdings.main(argv)
    raw = buffer.getvalue().strip()
    if not raw:
        raise SystemExit("Failed to read holdings output.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse holdings output: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Unexpected holdings output payload.")
    return payload


def _prepare_selection(
    selection: pd.DataFrame,
    *,
    side: str,
    top_n: int,
) -> pd.DataFrame:
    prepared = ensure_symbol_columns(selection, context="Holdings payload")
    if "side" not in prepared.columns:
        prepared["side"] = "long"
    if "rank" not in prepared.columns:
        prepared["rank"] = np.nan
    prepared["side"] = prepared["side"].astype(str).str.lower()
    if side != "all":
        prepared = prepared[prepared["side"] == side].copy()
    if prepared.empty:
        raise SystemExit(f"No holdings available for --side={side}.")
    prepared.sort_values(["side", "rank", "ts_code"], inplace=True, na_position="last")
    prepared = prepared.head(top_n).copy()
    if prepared.empty:
        raise SystemExit("No holdings available after --top-n filtering.")
    prepared.reset_index(drop=True, inplace=True)
    return prepared


def _allocate_equal_weight(
    selection: pd.DataFrame,
    *,
    cash: float,
    buffer_bps: float,
    ts_to_order_book_id: dict[str, str],
    price_map: dict[str, float],
    lot_map: dict[str, int],
) -> tuple[pd.DataFrame, float, float, float]:
    if selection.empty:
        raise SystemExit("No holdings selected for allocation.")
    investable_cash = float(cash) * max(0.0, 1.0 - float(buffer_bps) / 10000.0)
    target_value = investable_cash / float(len(selection))

    rows: list[dict] = []
    for _, row in selection.iterrows():
        ts_code = str(row["ts_code"])
        order_book_id = ts_to_order_book_id[ts_code]
        price = float(price_map[order_book_id])
        round_lot = max(1, int(lot_map.get(order_book_id, 1)))
        lot_cost = price * round_lot
        lots = 0 if lot_cost <= 0 else int(math.floor(target_value / lot_cost))
        shares = lots * round_lot
        est_value = shares * price
        rank_value = row.get("rank")
        rows.append(
            {
                "ts_code": ts_code,
                "stock_ticker": ts_code,
                "order_book_id": order_book_id,
                "side": str(row.get("side", "long")),
                "rank": int(rank_value) if pd.notna(rank_value) else None,
                "price": price,
                "round_lot": round_lot,
                "target_value": target_value,
                "lot_cost": lot_cost,
                "lots": lots,
                "shares": shares,
                "est_value": est_value,
            }
        )

    alloc_df = pd.DataFrame(rows)
    alloc_df["gap_to_target"] = alloc_df["target_value"] - alloc_df["est_value"]
    est_total = float(alloc_df["est_value"].sum())
    cash_left = investable_cash - est_total
    return alloc_df, investable_cash, est_total, cash_left


def _render_text(payload: dict, alloc_df: pd.DataFrame) -> str:
    lines = [
        f"截至日期: {payload['as_of']}",
        f"建仓日期: {payload['entry_date']}",
        f"价格日期: {payload['price_date']}",
        f"来源: {payload['source']}",
        f"方向: {payload['side']}",
        f"Top-N 请求/实际: {payload['requested_top_n']} / {payload['selected_n']}",
        f"总资金: {_money(float(payload['cash']))}",
        f"可投资资金: {_money(float(payload['investable_cash']))}",
        f"预计持仓金额: {_money(float(payload['estimated_value']))}",
        f"预计剩余现金: {_money(float(payload['cash_left']))}",
        f"目标缺口合计: {_money(float(payload['total_gap_to_target']))}",
    ]
    if payload.get("run_dir"):
        lines.append(f"运行目录: {payload['run_dir']}")
    if payload.get("positions_file"):
        lines.append(f"持仓文件: {payload['positions_file']}")
    lines.append("")

    table_headers = [
        "stock_ticker",
        "lots",
        "价格",
        "每手股数",
        "目标金额",
        "股数",
        "预计金额",
        "目标缺口",
    ]
    table_rows: list[list[str]] = []
    for _, row in alloc_df.iterrows():
        table_rows.append(
            [
                str(row["stock_ticker"]),
                str(int(row["lots"])),
                f"{float(row['price']):.4f}",
                str(int(row["round_lot"])),
                _money(float(row["target_value"])),
                str(int(row["shares"])),
                _money(float(row["est_value"])),
                _money(float(row["gap_to_target"])),
            ]
        )
    lines.append(_format_table(table_rows, table_headers))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Size equal-weight holdings into shares/lots using rqdata prices and round_lot."
    )
    parser.add_argument(
        "--config",
        help="Pipeline config path or built-in name (default: default).",
    )
    parser.add_argument(
        "--run-dir",
        help="Explicit run directory to read (overrides --config).",
    )
    parser.add_argument(
        "--positions-file",
        help="Explicit positions CSV path (overrides --config/--run-dir).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional Top-K filter when selecting the latest run.",
    )
    parser.add_argument(
        "--as-of",
        default="t-1",
        help="As-of date (YYYYMMDD, YYYY-MM-DD, today, t-1). Default: t-1.",
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "backtest", "live"],
        help="Positions source (auto/backtest/live). Default: auto.",
    )
    parser.add_argument(
        "--side",
        default="long",
        choices=["long", "short", "all"],
        help="Select side for allocation (long/short/all). Default: long.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of names to allocate equally from the sorted holdings list. Default: 20.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=1_000_000,
        help="Total portfolio cash for sizing. Default: 1000000.",
    )
    parser.add_argument(
        "--buffer-bps",
        type=float,
        default=0.0,
        help="Cash buffer in bps reserved from investment (e.g., fees). Default: 0.",
    )
    parser.add_argument(
        "--price-field",
        default="close",
        help="Price field fetched from rqdata.get_price. Default: close.",
    )
    parser.add_argument(
        "--price-lookback-days",
        type=int,
        default=20,
        help="Price lookback window in calendar days before price date. Default: 20.",
    )
    parser.add_argument(
        "--username",
        help="Override RQData username.",
    )
    parser.add_argument(
        "--password",
        help="Override RQData password.",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "csv", "json"],
        help="Output format (text/csv/json). Default: text.",
    )
    parser.add_argument(
        "--out",
        help="Optional output path (default: stdout).",
    )
    args = parser.parse_args(argv)

    if args.top_n <= 0:
        raise SystemExit("--top-n must be a positive integer.")
    if args.cash <= 0:
        raise SystemExit("--cash must be positive.")
    if args.buffer_bps < 0:
        raise SystemExit("--buffer-bps must be non-negative.")
    if args.price_lookback_days <= 0:
        raise SystemExit("--price-lookback-days must be a positive integer.")

    as_of = holdings._resolve_as_of(args.as_of)
    run_dir: Path | None = None
    positions_path: Path | None = None

    if args.positions_file:
        positions_path = Path(args.positions_file).expanduser()
        if not positions_path.is_absolute():
            positions_path = (Path.cwd() / positions_path).resolve()
        selection, entry_date = _select_from_positions_file(positions_path, as_of)
        source = "positions_file"
    else:
        payload = _load_holdings_payload(args)
        rows = payload.get("holdings")
        if not isinstance(rows, list):
            raise SystemExit("Invalid holdings payload: missing holdings list.")
        selection = pd.DataFrame(rows)
        if selection.empty:
            raise SystemExit("Holdings payload is empty.")
        entry_date = pd.to_datetime(payload.get("entry_date"), errors="coerce")
        if pd.isna(entry_date):
            if "entry_date" in selection.columns:
                parsed_entries = holdings._parse_date_column(selection["entry_date"])
                if parsed_entries.notna().any():
                    entry_date = parsed_entries.max()
        if pd.isna(entry_date):
            raise SystemExit("Failed to parse entry_date from holdings payload.")
        entry_date = pd.Timestamp(entry_date).normalize()
        run_value = payload.get("run_dir")
        if run_value:
            run_dir = Path(str(run_value))
        positions_value = payload.get("positions_file")
        if positions_value:
            positions_path = Path(str(positions_value))
        source = str(payload.get("source") or args.source)

    prepared = _prepare_selection(selection, side=args.side, top_n=args.top_n)
    symbols = [str(value) for value in prepared["ts_code"].tolist()]
    cfg = _load_config(args.config)
    market = _resolve_market(cfg, symbols)

    rqdatac = _init_rqdatac(args.config, args.username, args.password)
    price_date = _resolve_price_date(rqdatac, as_of, market)
    start_date = (price_date - pd.Timedelta(days=int(args.price_lookback_days))).strftime(
        "%Y%m%d"
    )
    end_date = price_date.strftime("%Y%m%d")

    ts_to_order_book_id: dict[str, str] = {}
    order_book_ids: list[str] = []
    for symbol in symbols:
        order_book_id = _to_rq_order_book_id(symbol, market)
        ts_to_order_book_id[symbol] = order_book_id
        if order_book_id not in order_book_ids:
            order_book_ids.append(order_book_id)

    price_map = _fetch_latest_price_map(
        rqdatac,
        order_book_ids,
        field=args.price_field,
        start_date=start_date,
        end_date=end_date,
        market=market,
    )
    lot_map = _fetch_round_lot_map(rqdatac, order_book_ids, market)

    alloc_df, investable_cash, est_total, cash_left = _allocate_equal_weight(
        prepared,
        cash=float(args.cash),
        buffer_bps=float(args.buffer_bps),
        ts_to_order_book_id=ts_to_order_book_id,
        price_map=price_map,
        lot_map=lot_map,
    )
    total_gap_to_target = float(alloc_df["gap_to_target"].sum())

    payload = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "price_date": price_date.strftime("%Y-%m-%d"),
        "source": source,
        "side": args.side,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "positions_file": str(positions_path) if positions_path is not None else None,
        "market": market,
        "requested_top_n": int(args.top_n),
        "selected_n": int(len(alloc_df)),
        "equal_weight": 1.0 / float(len(alloc_df)),
        "cash": float(args.cash),
        "buffer_bps": float(args.buffer_bps),
        "investable_cash": float(investable_cash),
        "estimated_value": float(est_total),
        "cash_left": float(cash_left),
        "total_gap_to_target": total_gap_to_target,
        "price_field": args.price_field,
        "allocations": alloc_df.to_dict(orient="records"),
    }

    if args.format == "text":
        content = _render_text(payload, alloc_df)
    elif args.format == "csv":
        content = alloc_df.to_csv(index=False)
    else:
        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    if args.out:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(content)


if __name__ == "__main__":
    main()
