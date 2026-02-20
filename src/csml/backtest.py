from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from .portfolio import select_holdings
from .execution import BpsCostModel, ExecutionModel, ExitPolicy


def _drawdown_timing(nav: pd.Series) -> dict[str, float]:
    if nav is None:
        return {
            "drawdown_duration": np.nan,
            "recovery_time": np.nan,
            "drawdown_duration_days": np.nan,
            "recovery_time_days": np.nan,
        }
    nav = nav.dropna()
    if nav.empty:
        return {
            "drawdown_duration": np.nan,
            "recovery_time": np.nan,
            "drawdown_duration_days": np.nan,
            "recovery_time_days": np.nan,
        }

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    trough_date = drawdown.idxmin()
    try:
        trough_pos = nav.index.get_loc(trough_date)
    except KeyError:
        trough_pos = int(drawdown.values.argmin())
        trough_date = nav.index[trough_pos]

    peak_value = float(running_max.loc[trough_date])
    pre_peak = nav.loc[:trough_date]
    peak_candidates = pre_peak[pre_peak == peak_value]
    if peak_candidates.empty:
        peak_date = nav.index[0]
        peak_pos = 0
    else:
        peak_date = peak_candidates.index[-1]
        peak_pos = nav.index.get_loc(peak_date)

    drawdown_duration = float(trough_pos - peak_pos)

    post_nav = nav.loc[trough_date:]
    recovery_candidates = post_nav[post_nav >= peak_value]
    if recovery_candidates.empty:
        recovery_time = np.nan
        recovery_days = np.nan
    else:
        recovery_date = recovery_candidates.index[0]
        recovery_pos = nav.index.get_loc(recovery_date)
        recovery_time = float(recovery_pos - trough_pos)
        if isinstance(nav.index, pd.DatetimeIndex):
            recovery_days = float((recovery_date - trough_date).days)
        else:
            recovery_days = np.nan

    if isinstance(nav.index, pd.DatetimeIndex):
        drawdown_days = float((trough_date - peak_date).days)
    else:
        drawdown_days = np.nan

    return {
        "drawdown_duration": drawdown_duration,
        "recovery_time": recovery_time,
        "drawdown_duration_days": drawdown_days,
        "recovery_time_days": recovery_days,
    }

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
            "sortino": np.nan,
            "calmar": np.nan,
            "drawdown_duration": np.nan,
            "recovery_time": np.nan,
            "drawdown_duration_days": np.nan,
            "recovery_time_days": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
            "avg_exit_lag_days": np.nan,
            "max_exit_lag_days": np.nan,
            "periods_with_delayed_exit": 0,
            "delayed_exit_ratio": np.nan,
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

    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else np.nan
    if np.isfinite(downside_std) and downside_std > 0 and np.isfinite(periods_per_year):
        sortino = float(returns.mean() / downside_std * np.sqrt(periods_per_year))
    else:
        sortino = np.nan

    if np.isfinite(max_drawdown) and max_drawdown < 0 and np.isfinite(ann_return):
        calmar = float(ann_return / abs(max_drawdown))
    else:
        calmar = np.nan

    timing = _drawdown_timing(nav)

    exit_lags: list[float] = []
    for info in period_info:
        lag_raw = info.get("exit_delay_steps")
        if lag_raw is None:
            planned_idx = info.get("planned_exit_idx")
            exit_idx = info.get("exit_idx")
            if planned_idx is not None and exit_idx is not None:
                lag_raw = int(exit_idx) - int(planned_idx)
        if lag_raw is None:
            continue
        try:
            lag = float(lag_raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(lag):
            exit_lags.append(max(0.0, lag))

    if exit_lags:
        avg_exit_lag = float(np.mean(exit_lags))
        max_exit_lag = float(np.max(exit_lags))
        delayed_periods = int(sum(lag > 0 for lag in exit_lags))
        delayed_ratio = delayed_periods / float(len(exit_lags))
    else:
        avg_exit_lag = np.nan
        max_exit_lag = np.nan
        delayed_periods = 0
        delayed_ratio = np.nan

    skew = float(returns.skew()) if returns.shape[0] > 2 else np.nan
    kurtosis = float(returns.kurtosis()) if returns.shape[0] > 3 else np.nan
    if returns.shape[0] > 0:
        var_95 = float(np.nanpercentile(returns, 5))
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if not tail.empty else np.nan
    else:
        var_95 = np.nan
        cvar_95 = np.nan

    return {
        "periods": len(returns),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_holding": avg_holding,
        "periods_per_year": periods_per_year,
        "sortino": sortino,
        "calmar": calmar,
        "drawdown_duration": timing["drawdown_duration"],
        "recovery_time": timing["recovery_time"],
        "drawdown_duration_days": timing["drawdown_duration_days"],
        "recovery_time_days": timing["recovery_time_days"],
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "avg_exit_lag_days": avg_exit_lag,
        "max_exit_lag_days": max_exit_lag,
        "periods_with_delayed_exit": delayed_periods,
        "delayed_exit_ratio": delayed_ratio,
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
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    tradable_col: Optional[str] = None,
    exit_price_policy: Literal["strict", "ffill", "delay"] = "strict",
    exit_fallback_policy: Literal["ffill", "none"] = "ffill",
    execution: Optional[ExecutionModel] = None,
):
    if execution is None:
        if exit_price_policy not in {"strict", "ffill", "delay"}:
            raise ValueError("exit_price_policy must be one of: strict, ffill, delay.")
        if exit_fallback_policy not in {"ffill", "none"}:
            raise ValueError("exit_fallback_policy must be one of: ffill, none.")
        exit_policy = ExitPolicy(exit_price_policy, exit_fallback_policy)
        cost_model = BpsCostModel(cost_bps)
    else:
        exit_policy = execution.exit_policy
        cost_model = execution.cost_model
    trade_dates = sorted(data["trade_date"].unique())
    if len(trade_dates) < 2:
        return None
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    price_table = data.pivot(index="trade_date", columns="ts_code", values=price_col)
    tradable_table = None
    if tradable_col and tradable_col in data.columns:
        tradable_table = data.pivot(index="trade_date", columns="ts_code", values=tradable_col)
        tradable_table = tradable_table.fillna(False).astype(bool)

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

    def _resolve_exit_prices(
        holdings: list[str],
        planned_exit_idx: int,
    ) -> tuple[pd.Series, int]:
        return exit_policy.resolve_exit_prices(
            holdings,
            planned_exit_idx,
            price_table=price_table,
            tradable_table=tradable_table,
            trade_dates=trade_dates,
            date_to_idx=date_to_idx,
        )

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

        if prev_exit_idx is not None and entry_idx < prev_exit_idx:
            continue

        if entry_idx >= len(trade_dates) or exit_idx >= len(trade_dates) or entry_idx >= exit_idx:
            continue

        entry_date = trade_dates[entry_idx]
        planned_exit_idx = exit_idx
        planned_exit_date = trade_dates[planned_exit_idx]
        day = data[data["trade_date"] == reb_date]
        if day.empty:
            continue

        k = min(top_k, len(day))
        if k <= 0:
            continue

        if long_only:
            holdings, entry_prices = select_holdings(
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
            exit_prices, period_exit_idx = _resolve_exit_prices(holdings, exit_idx)
            if exit_prices.empty:
                continue
            entry_prices = entry_prices.reindex(exit_prices.index)
            holdings = list(exit_prices.index)
            k = len(holdings)
            if k == 0:
                continue
            exit_idx = period_exit_idx
            exit_date = trade_dates[exit_idx]
            period_returns = (exit_prices / entry_prices) - 1.0
            gross = period_returns.mean()
            turnover = _compute_turnover(prev_holdings, prev_entry_prices, prev_entry_date, holdings, entry_date)
            cost = cost_model.cost(turnover, is_initial=prev_holdings is None, side="long")
            net = gross - cost

            prev_holdings = set(holdings)
            prev_entry_date = entry_date
            prev_entry_prices = entry_prices
        else:
            short_k_final = short_k if short_k is not None else k
            short_k_final = min(int(short_k_final), len(day) - k)
            if short_k_final <= 0:
                continue

            long_holdings, long_entry = select_holdings(
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
            short_holdings, short_entry = select_holdings(
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

            long_exit, period_exit_idx_long = _resolve_exit_prices(long_holdings, exit_idx)
            short_exit, period_exit_idx_short = _resolve_exit_prices(short_holdings, exit_idx)
            if long_exit.empty or short_exit.empty:
                continue
            long_entry = long_entry.reindex(long_exit.index)
            short_entry = short_entry.reindex(short_exit.index)
            long_holdings = list(long_exit.index)
            short_holdings = list(short_exit.index)
            if not long_holdings or not short_holdings:
                continue
            exit_idx = max(exit_idx, period_exit_idx_long, period_exit_idx_short)
            exit_date = trade_dates[exit_idx]

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
            cost_long = cost_model.cost(turnover_long, is_initial=prev_holdings is None, side="long")
            cost_short = cost_model.cost(turnover_short, is_initial=prev_short_holdings is None, side="short")
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
                "planned_exit_idx": planned_exit_idx,
                "exit_idx": exit_idx,
                "entry_date": entry_date,
                "planned_exit_date": planned_exit_date,
                "exit_date": exit_date,
                "exit_delay_steps": int(exit_idx - planned_exit_idx),
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
