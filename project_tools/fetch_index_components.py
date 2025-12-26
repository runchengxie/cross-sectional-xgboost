# -*- coding: utf-8 -*-
"""Fetch index constituents from TuShare and write symbols to a text file."""
from __future__ import annotations

import argparse
import calendar
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import tushare as ts


def require_token() -> str:
    load_dotenv()
    token = os.getenv("TUSHARE_API_KEY") or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("Please set TUSHARE_API_KEY or TUSHARE_TOKEN before running.")
    return token


def validate_yyyymm(value: str) -> str:
    if not value or len(value) != 6 or not value.isdigit():
        raise SystemExit("month must be in YYYYMM format.")
    return value


def validate_yyyymmdd(value: str, label: str) -> str:
    if not value or len(value) != 8 or not value.isdigit():
        raise SystemExit(f"{label} must be in YYYYMMDD format.")
    return value


def month_bounds(month: str) -> tuple[str, str]:
    month = validate_yyyymm(month)
    year = int(month[:4])
    mon = int(month[4:])
    if not 1 <= mon <= 12:
        raise SystemExit("month must be between 01 and 12.")
    last_day = calendar.monthrange(year, mon)[1]
    start = f"{year:04d}{mon:02d}01"
    end = f"{year:04d}{mon:02d}{last_day:02d}"
    return start, end


def resolve_index_code(
    pro: ts.pro_api, index_code: str | None, index_name: str | None, market: str
) -> str:
    if index_code:
        return index_code.strip()
    if not index_name:
        raise SystemExit("Provide --index-code or --index-name.")
    df = pro.index_basic(market=market, fields="ts_code,name")
    if df is None or df.empty:
        raise SystemExit("index_basic returned no data; check market or token permissions.")
    matches = df[df["name"].str.contains(index_name, na=False)]
    if matches.empty:
        raise SystemExit(f"No index matches name: {index_name}")
    if len(matches) > 1:
        options = ", ".join(matches["name"].head(5).tolist())
        raise SystemExit(
            f"Multiple matches for '{index_name}'. Use --index-code. Example matches: {options}"
        )
    return str(matches.iloc[0]["ts_code"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TuShare index constituents.")
    parser.add_argument("--index-code", help="Index code, e.g. 000300.SH")
    parser.add_argument("--index-name", help="Index name from TuShare index_basic")
    parser.add_argument("--market", default="CSI", help="Index market (default: CSI)")
    parser.add_argument("--month", help="Month in YYYYMM; overrides start/end dates")
    parser.add_argument("--start-date", help="Start date in YYYYMMDD")
    parser.add_argument("--end-date", help="End date in YYYYMMDD")
    parser.add_argument("--out", help="Output file for symbols (one per line)")
    args = parser.parse_args()

    token = require_token()
    pro = ts.pro_api(token=token)

    index_code = resolve_index_code(pro, args.index_code, args.index_name, args.market)

    if args.month:
        start_date, end_date = month_bounds(args.month)
    elif args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            raise SystemExit("Provide both --start-date and --end-date.")
        start_date = validate_yyyymmdd(args.start_date, "start-date")
        end_date = validate_yyyymmdd(args.end_date, "end-date")
    else:
        start_date, end_date = month_bounds(datetime.now().strftime("%Y%m"))

    df = pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise SystemExit(
            "index_weight returned no data. Check the index code, date range, "
            "or your TuShare permission level."
        )

    symbols = sorted(df["con_code"].dropna().unique().tolist())
    out_path = Path(args.out) if args.out else Path(f"{index_code}_symbols.txt")
    out_path.write_text("\n".join(symbols), encoding="utf-8")

    print(f"Index: {index_code}")
    print(f"Date range: {start_date} - {end_date}")
    print(f"Wrote {len(symbols)} symbols to {out_path}")


if __name__ == "__main__":
    main()
