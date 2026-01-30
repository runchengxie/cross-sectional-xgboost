from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def summarize_period_returns(
    returns: pd.Series,
    period_info: list[dict],
    trading_days_per_year: int,
) -> dict:
    if returns is None or returns.empty:
        return {
            "periods": 0,
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "avg_holding": np.nan,
            "periods_per_year": np.nan,
        }

    nav = (1 + returns).cumprod()
    total_return = nav.iloc[-1] - 1.0
    max_drawdown = (nav / nav.cummax() - 1.0).min()

    total_days = np.nan
    if period_info:
        entry_first = period_info[0]["entry_idx"]
        exit_last = period_info[-1]["exit_idx"]
        total_days = exit_last - entry_first
    if np.isfinite(total_days) and total_days > 0:
        ann_return = (1 + total_return) ** (trading_days_per_year / total_days) - 1.0
    else:
        ann_return = np.nan

    holding_lengths = [info["exit_idx"] - info["entry_idx"] for info in period_info]
    avg_holding = np.mean(holding_lengths) if holding_lengths else np.nan
    periods_per_year = (
        trading_days_per_year / avg_holding
        if np.isfinite(avg_holding) and avg_holding > 0
        else np.nan
    )
    period_vol = returns.std(ddof=1)
    if np.isfinite(period_vol) and period_vol > 0 and np.isfinite(periods_per_year):
        ann_vol = period_vol * np.sqrt(periods_per_year)
        sharpe = returns.mean() / period_vol * np.sqrt(periods_per_year)
    else:
        ann_vol = np.nan
        sharpe = np.nan

    return {
        "periods": len(returns),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_holding": avg_holding,
        "periods_per_year": periods_per_year,
    }


def backtest_topk(
    data: pd.DataFrame,
    pred_col: str,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    top_k: int,
    shift_days: int,
    cost_bps: float,
    trading_days_per_year: int,
    exit_mode: Literal["rebalance", "label_horizon"] = "rebalance",
    exit_horizon_days: Optional[int] = None,
    long_only: bool = True,
    short_k: Optional[int] = None,
):
    trade_dates = sorted(data["trade_date"].unique())
    if len(trade_dates) < 2:
        return None
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    price_table = data.pivot(index="trade_date", columns="ts_code", values=price_col)

    net_returns = []
    gross_returns = []
    turnovers = []
    costs = []
    period_info = []
    prev_holdings = None
    prev_entry_date = None
    prev_entry_prices = None
    prev_short_holdings = None
    prev_short_entry_date = None
    prev_short_entry_prices = None
    prev_exit_idx = None

    def _compute_turnover(
        prev_set: Optional[set],
        prev_prices: Optional[pd.Series],
        prev_date: Optional[pd.Timestamp],
        current_holdings: list[str],
        entry_date: pd.Timestamp,
    ) -> float:
        if prev_set is None:
            return 1.0
        drift_turnover = np.nan
        if prev_prices is not None and prev_date is not None:
            prev_holdings_list = list(prev_set)
            prev_prices = prev_prices.reindex(prev_holdings_list)
            prev_prices = prev_prices[prev_prices.notna()]
            if not prev_prices.empty and prev_date in price_table.index:
                current_prices = price_table.loc[entry_date, prev_prices.index]
                valid_prev = current_prices.notna()
                prev_prices = prev_prices[valid_prev]
                current_prices = current_prices[valid_prev]
                if not prev_prices.empty:
                    prev_weights = np.repeat(1.0 / len(prev_prices), len(prev_prices))
                    drift = prev_weights * (current_prices / prev_prices).to_numpy()
                    drift_sum = drift.sum()
                    if drift_sum > 0:
                        drift_weights = pd.Series(drift / drift_sum, index=prev_prices.index)
                        target_weights = pd.Series(1.0 / len(current_holdings), index=current_holdings)
                        all_ids = drift_weights.index.union(target_weights.index)
                        drift_aligned = drift_weights.reindex(all_ids).fillna(0.0)
                        target_aligned = target_weights.reindex(all_ids).fillna(0.0)
                        drift_turnover = 0.5 * float(np.abs(target_aligned - drift_aligned).sum())
        if np.isfinite(drift_turnover):
            return drift_turnover
        overlap = len(set(current_holdings) & prev_set)
        return 1 - overlap / len(current_holdings)

    for i, reb_date in enumerate(rebalance_dates):
        if reb_date not in date_to_idx:
            continue
        if exit_mode == "rebalance":
            if i >= len(rebalance_dates) - 1:
                break
            next_reb = rebalance_dates[i + 1]
            if next_reb not in date_to_idx:
                continue
            entry_idx = date_to_idx[reb_date] + shift_days
            exit_idx = date_to_idx[next_reb] + shift_days
        else:
            if exit_horizon_days is None:
                raise ValueError("exit_horizon_days is required for exit_mode='label_horizon'.")
            entry_idx = date_to_idx[reb_date] + shift_days
            exit_idx = entry_idx + exit_horizon_days
            if prev_exit_idx is not None and entry_idx < prev_exit_idx:
                raise ValueError(
                    "exit_mode='label_horizon' overlaps with rebalance_dates. "
                    "Increase rebalance_frequency or use exit_mode='rebalance'."
                )

        if entry_idx >= len(trade_dates) or exit_idx >= len(trade_dates) or entry_idx >= exit_idx:
            continue

        entry_date = trade_dates[entry_idx]
        exit_date = trade_dates[exit_idx]
        day = data[data["trade_date"] == reb_date]
        if day.empty:
            continue

        k = min(top_k, len(day))
        if k <= 0:
            continue

        cost_per_side = cost_bps / 10000.0
        if long_only:
            holdings = list(day.nlargest(k, pred_col)["ts_code"])
            entry_prices = price_table.loc[entry_date, holdings]
            exit_prices = price_table.loc[exit_date, holdings]
            valid = entry_prices.notna() & exit_prices.notna()
            if valid.sum() == 0:
                continue

            entry_prices = entry_prices[valid]
            exit_prices = exit_prices[valid]
            holdings = list(entry_prices.index)
            k = len(holdings)
            if k == 0:
                continue

            period_returns = (exit_prices / entry_prices) - 1.0
            gross = period_returns.mean()
            turnover = _compute_turnover(prev_holdings, prev_entry_prices, prev_entry_date, holdings, entry_date)
            if prev_holdings is None:
                cost = cost_per_side
            else:
                cost = 2 * cost_per_side * turnover
            net = gross - cost

            prev_holdings = set(holdings)
            prev_entry_date = entry_date
            prev_entry_prices = entry_prices
        else:
            short_k_final = short_k if short_k is not None else k
            short_k_final = min(int(short_k_final), len(day) - k)
            if short_k_final <= 0:
                continue

            long_holdings = list(day.nlargest(k, pred_col)["ts_code"])
            short_holdings = list(day.nsmallest(short_k_final, pred_col)["ts_code"])

            long_entry = price_table.loc[entry_date, long_holdings]
            long_exit = price_table.loc[exit_date, long_holdings]
            long_valid = long_entry.notna() & long_exit.notna()
            long_entry = long_entry[long_valid]
            long_exit = long_exit[long_valid]
            long_holdings = list(long_entry.index)

            short_entry = price_table.loc[entry_date, short_holdings]
            short_exit = price_table.loc[exit_date, short_holdings]
            short_valid = short_entry.notna() & short_exit.notna()
            short_entry = short_entry[short_valid]
            short_exit = short_exit[short_valid]
            short_holdings = list(short_entry.index)

            if not long_holdings or not short_holdings:
                continue

            long_returns = (long_exit / long_entry) - 1.0
            short_returns = (short_exit / short_entry) - 1.0
            long_gross = long_returns.mean()
            short_gross = -short_returns.mean()
            gross = long_gross + short_gross

            turnover_long = _compute_turnover(
                prev_holdings, prev_entry_prices, prev_entry_date, long_holdings, entry_date
            )
            turnover_short = _compute_turnover(
                prev_short_holdings,
                prev_short_entry_prices,
                prev_short_entry_date,
                short_holdings,
                entry_date,
            )
            cost_long = cost_per_side if prev_holdings is None else 2 * cost_per_side * turnover_long
            cost_short = (
                cost_per_side if prev_short_holdings is None else 2 * cost_per_side * turnover_short
            )
            cost = cost_long + cost_short
            net = gross - cost
            turnover = turnover_long + turnover_short

            prev_holdings = set(long_holdings)
            prev_entry_date = entry_date
            prev_entry_prices = long_entry
            prev_short_holdings = set(short_holdings)
            prev_short_entry_date = entry_date
            prev_short_entry_prices = short_entry

        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)
        costs.append(cost)
        period_info.append(
            {
                "rebalance_date": reb_date,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_date": entry_date,
                "exit_date": exit_date,
            }
        )
        prev_exit_idx = exit_idx

    if not net_returns:
        return None

    index = [info["exit_date"] for info in period_info]
    net_series = pd.Series(net_returns, index=index, name="net_return")
    gross_series = pd.Series(gross_returns, index=index, name="gross_return")
    turnover_series = pd.Series(turnovers, index=index, name="turnover")

    stats = summarize_period_returns(net_series, period_info, trading_days_per_year)
    avg_turnover = turnover_series.dropna().mean() if turnover_series.notna().any() else np.nan
    avg_cost = float(np.mean(costs)) if costs else np.nan
    stats.update(
        {
            "avg_turnover": avg_turnover,
            "avg_cost_drag": avg_cost,
            "mode": "long_only" if long_only else "long_short",
            "long_k": int(top_k),
            "short_k": int(short_k) if (not long_only and short_k is not None) else None,
        }
    )
    return stats, net_series, gross_series, turnover_series, period_info
