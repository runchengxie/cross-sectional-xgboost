from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return x_rank.corr(y_rank)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    return x.corr(y)


def daily_ic_series(
    data: pd.DataFrame,
    target_col: str,
    pred_col: str,
    *,
    method: str = "spearman",
) -> pd.Series:
    method = str(method).strip().lower()
    if method == "spearman":
        corr_fn = spearman_corr
    elif method == "pearson":
        corr_fn = pearson_corr
    else:
        raise ValueError("method must be one of: spearman, pearson.")
    records: list[tuple[pd.Timestamp, float]] = []
    for date, group in data.groupby("trade_date"):
        if group[target_col].nunique() < 2:
            continue
        ic = corr_fn(group[pred_col], group[target_col])
        if not np.isnan(ic):
            records.append((pd.to_datetime(date), float(ic)))
    if not records:
        return pd.Series(dtype=float, name="ic")
    records.sort(key=lambda x: x[0])
    return pd.Series(
        [value for _, value in records],
        index=pd.Index([date for date, _ in records], name="trade_date"),
        name="ic",
    )


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    if ic_series is None or ic_series.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    values = ic_series.dropna()
    n = int(values.shape[0])
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    ir = mean / std if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p_value = np.nan
    if scipy_stats is not None and np.isfinite(t_stat) and n > 1:
        p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def quantile_returns(
    data: pd.DataFrame,
    pred_col: str,
    target_col: str,
    n_quantiles: int,
) -> pd.DataFrame:
    def _add_quantile(values: pd.Series) -> pd.Series:
        if len(values) < n_quantiles:
            return pd.Series([np.nan] * len(values), index=values.index)
        ranks = values.rank(method="first")
        return pd.qcut(ranks, n_quantiles, labels=False)

    data = data.copy()
    quantile = data.groupby("trade_date")[pred_col].apply(_add_quantile)
    data["quantile"] = quantile.reset_index(level=0, drop=True)
    data = data.dropna(subset=["quantile"])  # drop dates with insufficient symbols

    q_ret = data.groupby(["trade_date", "quantile"])[target_col].mean().unstack()
    q_ret.index = pd.to_datetime(q_ret.index)
    return q_ret


def regression_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if aligned.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    y_true = aligned["y_true"].to_numpy()
    y_pred = aligned["y_pred"].to_numpy()
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(err**2) / denom) if denom > 0 else np.nan
    return {"n": int(aligned.shape[0]), "mae": mae, "rmse": rmse, "r2": r2}


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if aligned.empty:
        return {"n": 0, "hit_rate": np.nan}
    sign_true = np.sign(aligned["y_true"].to_numpy())
    sign_pred = np.sign(aligned["y_pred"].to_numpy())
    hit = float(np.mean(sign_true == sign_pred)) if aligned.shape[0] > 0 else np.nan
    return {"n": int(aligned.shape[0]), "hit_rate": hit}


def topk_positive_ratio(
    data: pd.DataFrame,
    pred_col: str,
    target_col: str,
    k: int,
    *,
    date_col: str = "trade_date",
) -> dict[str, float]:
    if data is None or data.empty or k <= 0:
        return {"n_dates": 0, "topk_positive_ratio": np.nan}
    ratios = []
    for _, group in data.groupby(date_col):
        if len(group) < k:
            continue
        top = group.nlargest(k, pred_col)
        if top.empty:
            continue
        ratios.append(float((top[target_col] > 0).mean()))
    if not ratios:
        return {"n_dates": 0, "topk_positive_ratio": np.nan}
    return {"n_dates": int(len(ratios)), "topk_positive_ratio": float(np.mean(ratios))}


def assign_daily_quantile_bucket(
    data: pd.DataFrame,
    value_col: str,
    n_bins: int,
    *,
    date_col: str = "trade_date",
) -> pd.Series:
    def _bucket(values: pd.Series) -> pd.Series:
        if values.nunique() < n_bins:
            return pd.Series([np.nan] * len(values), index=values.index)
        ranks = values.rank(method="first")
        return pd.qcut(ranks, n_bins, labels=False, duplicates="drop")

    buckets = data.groupby(date_col, sort=False)[value_col].apply(_bucket)
    buckets = buckets.reset_index(level=0, drop=True)
    buckets.name = f"{value_col}_bucket"
    return buckets


def bucket_ic_summary(
    data: pd.DataFrame,
    target_col: str,
    pred_col: str,
    bucket_col: str,
    *,
    method: str = "spearman",
    min_count: int = 0,
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    subset = data.dropna(subset=[bucket_col, target_col, pred_col])
    if subset.empty:
        return pd.DataFrame()
    records: list[dict[str, float | str]] = []
    for bucket_value, group in subset.groupby(bucket_col, sort=False):
        if min_count and group.shape[0] < min_count:
            continue
        ic_series = daily_ic_series(group, target_col, pred_col, method=method)
        stats = summarize_ic(ic_series)
        stats["bucket"] = str(bucket_value)
        stats["bucket_col"] = str(bucket_col)
        records.append(stats)
    return pd.DataFrame(records)


def estimate_turnover(
    data: pd.DataFrame,
    pred_col: str,
    k: int,
    rebalance_dates: list[pd.Timestamp],
    buffer_exit: int = 0,
    buffer_entry: int = 0,
) -> pd.Series:
    prev = None
    turnovers: list[tuple[pd.Timestamp, float]] = []
    for date in rebalance_dates:
        day = data[data["trade_date"] == date]
        if len(day) < k:
            continue
        ranked = day.sort_values(pred_col, ascending=False)["ts_code"].tolist()
        if prev is None or (buffer_exit <= 0 and buffer_entry <= 0):
            holdings = set(ranked[:k])
        else:
            keep_limit = min(len(ranked), k + max(0, buffer_exit))
            entry_limit = min(len(ranked), max(0, k - max(0, buffer_entry)))
            keep_set = set(ranked[:keep_limit]) & prev
            holdings_list: list[str] = [code for code in ranked if code in keep_set]
            preferred = set(ranked[:entry_limit]) if entry_limit > 0 else set()
            for code in ranked:
                if len(holdings_list) >= k:
                    break
                if code in holdings_list:
                    continue
                if preferred and code not in preferred:
                    continue
                holdings_list.append(code)
            if len(holdings_list) < k:
                for code in ranked:
                    if len(holdings_list) >= k:
                        break
                    if code in holdings_list:
                        continue
                    holdings_list.append(code)
            holdings = set(holdings_list[:k])
        if prev is not None:
            overlap = len(holdings & prev)
            turnovers.append((pd.to_datetime(date), 1 - overlap / k))
        prev = holdings
    if not turnovers:
        return pd.Series(dtype=float, name="turnover")
    turnovers.sort(key=lambda x: x[0])
    return pd.Series(
        [value for _, value in turnovers],
        index=pd.Index([date for date, _ in turnovers], name="trade_date"),
        name="turnover",
    )


def summarize_active_returns(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
) -> tuple[dict[str, float], pd.Series]:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if aligned.empty:
        empty = pd.Series(dtype=float, name="active_return")
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "beta": np.nan,
            "alpha": np.nan,
            "corr": np.nan,
            "active_total_return": np.nan,
        }, empty

    strategy = aligned["strategy"]
    benchmark = aligned["benchmark"]
    active = strategy - benchmark

    mean = float(active.mean())
    std = float(active.std(ddof=1)) if active.shape[0] > 1 else np.nan
    if np.isfinite(std) and std > 0 and np.isfinite(periods_per_year):
        tracking_error = std * np.sqrt(periods_per_year)
        information_ratio = mean / std * np.sqrt(periods_per_year)
    else:
        tracking_error = np.nan
        information_ratio = np.nan

    bench_var = float(benchmark.var(ddof=1)) if benchmark.shape[0] > 1 else np.nan
    if np.isfinite(bench_var) and bench_var > 0:
        beta = float(strategy.cov(benchmark) / bench_var)
    else:
        beta = np.nan
    if np.isfinite(beta) and np.isfinite(periods_per_year):
        alpha = float((strategy.mean() - beta * benchmark.mean()) * periods_per_year)
    else:
        alpha = np.nan

    corr = float(strategy.corr(benchmark)) if strategy.shape[0] > 1 else np.nan

    strat_total = float((1 + strategy).prod() - 1.0)
    bench_total = float((1 + benchmark).prod() - 1.0)
    if np.isfinite(strat_total) and np.isfinite(bench_total):
        active_total = (1 + strat_total) / (1 + bench_total) - 1.0
    else:
        active_total = np.nan

    return {
        "n": int(active.shape[0]),
        "mean": mean,
        "std": std,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "corr": corr,
        "active_total_return": float(active_total) if np.isfinite(active_total) else np.nan,
    }, active.rename("active_return")
