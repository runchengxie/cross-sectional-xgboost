from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def select_holdings(
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    k: int,
    pred_col: str,
    *,
    ascending: bool,
    price_table: pd.DataFrame,
    tradable_table: Optional[pd.DataFrame],
    prev_holdings: Optional[set[str]],
    buffer_exit: int,
    buffer_entry: int,
) -> tuple[list[str], pd.Series]:
    if day.empty or k <= 0:
        return [], pd.Series(dtype=float)
    if entry_date not in price_table.index:
        return [], pd.Series(dtype=float)

    ranked = day.sort_values(pred_col, ascending=ascending)
    ranked_codes = ranked["ts_code"].tolist()

    if prev_holdings is None or (buffer_exit <= 0 and buffer_entry <= 0):
        candidate_order = ranked_codes
    else:
        keep_limit = min(len(ranked_codes), k + max(0, buffer_exit))
        entry_limit = min(len(ranked_codes), max(0, k - max(0, buffer_entry)))
        keep_set = set(ranked_codes[:keep_limit]) & prev_holdings
        candidate_order = [code for code in ranked_codes if code in keep_set]
        preferred = set(ranked_codes[:entry_limit]) if entry_limit > 0 else set()
        for code in ranked_codes:
            if len(candidate_order) >= k:
                break
            if code in candidate_order:
                continue
            if preferred and code not in preferred:
                continue
            candidate_order.append(code)
        if len(candidate_order) < k:
            for code in ranked_codes:
                if len(candidate_order) >= k:
                    break
                if code in candidate_order:
                    continue
                candidate_order.append(code)

    entry_prices = price_table.loc[entry_date]
    tradable_flags = None
    if tradable_table is not None:
        if entry_date not in tradable_table.index:
            return [], pd.Series(dtype=float)
        tradable_flags = tradable_table.loc[entry_date]

    holdings: list[str] = []
    for symbol in candidate_order:
        if len(holdings) >= k:
            break
        price = entry_prices.get(symbol, np.nan)
        if not np.isfinite(price):
            continue
        if tradable_flags is not None and not bool(tradable_flags.get(symbol, False)):
            continue
        holdings.append(symbol)
    if not holdings:
        return [], pd.Series(dtype=float)
    return holdings, entry_prices.reindex(holdings)


def build_positions_by_rebalance(
    data: pd.DataFrame,
    pred_col: str,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    top_k: int,
    shift_days: int,
    *,
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    long_only: bool = True,
    short_k: Optional[int] = None,
    tradable_col: Optional[str] = None,
) -> pd.DataFrame:
    if data.empty or not rebalance_dates or top_k <= 0:
        return pd.DataFrame(
            columns=["rebalance_date", "entry_date", "ts_code", "weight", "signal", "rank", "side"]
        )

    trade_dates = sorted(data["trade_date"].unique())
    if len(trade_dates) < 2:
        return pd.DataFrame(
            columns=["rebalance_date", "entry_date", "ts_code", "weight", "signal", "rank", "side"]
        )
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    price_table = data.pivot(index="trade_date", columns="ts_code", values=price_col)

    tradable_table = None
    if tradable_col and tradable_col in data.columns:
        tradable_table = data.pivot(index="trade_date", columns="ts_code", values=tradable_col)
        tradable_table = tradable_table.fillna(False).astype(bool)

    results: list[dict[str, object]] = []
    prev_holdings: Optional[set[str]] = None
    prev_short_holdings: Optional[set[str]] = None

    for reb_date in rebalance_dates:
        if reb_date not in date_to_idx:
            continue
        entry_idx = date_to_idx[reb_date] + shift_days
        if entry_idx >= len(trade_dates):
            continue
        entry_date = trade_dates[entry_idx]
        day = data[data["trade_date"] == reb_date]
        if day.empty:
            continue

        k = min(int(top_k), len(day))
        if k <= 0:
            continue

        if long_only:
            holdings, _ = select_holdings(
                day,
                entry_date,
                k,
                pred_col,
                ascending=False,
                price_table=price_table,
                tradable_table=tradable_table,
                prev_holdings=prev_holdings,
                buffer_exit=buffer_exit,
                buffer_entry=buffer_entry,
            )
            if not holdings:
                continue
            weight = 1.0 / len(holdings)
            ranked_codes = day.sort_values(pred_col, ascending=False)["ts_code"].tolist()
            rank_map = {code: idx + 1 for idx, code in enumerate(ranked_codes)}
            signal_map = day.set_index("ts_code")[pred_col].to_dict()
            for code in holdings:
                results.append(
                    {
                        "rebalance_date": reb_date.strftime("%Y%m%d"),
                        "entry_date": entry_date.strftime("%Y%m%d"),
                        "ts_code": code,
                        "weight": weight,
                        "signal": float(signal_map.get(code, np.nan)),
                        "rank": int(rank_map.get(code, 0)),
                        "side": "long",
                    }
                )
            prev_holdings = set(holdings)
            continue

        short_k_final = short_k if short_k is not None else k
        short_k_final = min(int(short_k_final), len(day) - k)
        if short_k_final <= 0:
            continue

        long_holdings, _ = select_holdings(
            day,
            entry_date,
            k,
            pred_col,
            ascending=False,
            price_table=price_table,
            tradable_table=tradable_table,
            prev_holdings=prev_holdings,
            buffer_exit=buffer_exit,
            buffer_entry=buffer_entry,
        )
        short_holdings, _ = select_holdings(
            day,
            entry_date,
            short_k_final,
            pred_col,
            ascending=True,
            price_table=price_table,
            tradable_table=tradable_table,
            prev_holdings=prev_short_holdings,
            buffer_exit=buffer_exit,
            buffer_entry=buffer_entry,
        )
        if not long_holdings or not short_holdings:
            continue

        long_weight = 1.0 / len(long_holdings)
        short_weight = -1.0 / len(short_holdings)
        long_ranked = day.sort_values(pred_col, ascending=False)["ts_code"].tolist()
        short_ranked = day.sort_values(pred_col, ascending=True)["ts_code"].tolist()
        long_rank_map = {code: idx + 1 for idx, code in enumerate(long_ranked)}
        short_rank_map = {code: idx + 1 for idx, code in enumerate(short_ranked)}
        signal_map = day.set_index("ts_code")[pred_col].to_dict()

        for code in long_holdings:
            results.append(
                {
                    "rebalance_date": reb_date.strftime("%Y%m%d"),
                    "entry_date": entry_date.strftime("%Y%m%d"),
                    "ts_code": code,
                    "weight": long_weight,
                    "signal": float(signal_map.get(code, np.nan)),
                    "rank": int(long_rank_map.get(code, 0)),
                    "side": "long",
                }
            )
        for code in short_holdings:
            results.append(
                {
                    "rebalance_date": reb_date.strftime("%Y%m%d"),
                    "entry_date": entry_date.strftime("%Y%m%d"),
                    "ts_code": code,
                    "weight": short_weight,
                    "signal": float(signal_map.get(code, np.nan)),
                    "rank": int(short_rank_map.get(code, 0)),
                    "side": "short",
                }
            )

        prev_holdings = set(long_holdings)
        prev_short_holdings = set(short_holdings)

    if not results:
        return pd.DataFrame(
            columns=["rebalance_date", "entry_date", "ts_code", "weight", "signal", "rank", "side"]
        )

    output = pd.DataFrame(results)
    output.sort_values(["entry_date", "side", "rank", "ts_code"], inplace=True)
    return output.reset_index(drop=True)
