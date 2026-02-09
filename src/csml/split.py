from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import daily_ic_series
from .modeling import build_model, fit_model, resolve_model_spec


def build_sample_weight(
    data: pd.DataFrame,
    mode: str | None,
    *,
    date_col: str = "trade_date",
) -> np.ndarray | None:
    if mode is None:
        return None
    mode_text = str(mode).strip().lower()
    if mode_text in {"", "none", "null"}:
        return None
    if mode_text in {"date_equal", "date"}:
        counts = data.groupby(date_col, sort=False)[date_col].transform("count")
        return (1.0 / counts).to_numpy()
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")


def time_series_cv_ic(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_splits: int,
    embargo_days: int,
    purge_days: int,
    model_cfg: Mapping[str, object] | None = None,
    signal_direction: float = 1.0,
    sample_weight_mode: str | None = None,
    date_col: str = "trade_date",
    *,
    model_params: Mapping[str, object] | None = None,
):
    if model_cfg is not None and model_params is not None:
        raise ValueError("Provide either model_cfg or model_params, not both.")
    if model_params is not None:
        resolved_type, resolved_params = resolve_model_spec(
            {"type": "xgb_regressor", "params": dict(model_params)}
        )
    elif model_cfg is None:
        resolved_type, resolved_params = resolve_model_spec({})
    elif "type" in model_cfg or "params" in model_cfg:
        resolved_type, resolved_params = resolve_model_spec(model_cfg)
    else:
        resolved_type, resolved_params = resolve_model_spec(
            {"type": "xgb_regressor", "params": dict(model_cfg)}
        )

    sorted_data = data.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    date_values = sorted_data[date_col].to_numpy()
    if date_values.size == 0:
        return []

    dates, date_start_rows = np.unique(date_values, return_index=True)
    date_end_rows = np.empty_like(date_start_rows)
    if len(date_start_rows) > 1:
        date_end_rows[:-1] = date_start_rows[1:]
    date_end_rows[-1] = len(sorted_data)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    gap = max(int(embargo_days), int(purge_days))
    for train_idx, val_idx in tscv.split(dates):
        if gap > 0:
            cutoff = val_idx[0] - gap
            train_idx = train_idx[train_idx < cutoff]
            if len(train_idx) == 0:
                continue

        tr_start = date_start_rows[train_idx[0]]
        tr_end = date_end_rows[train_idx[-1]]
        va_start = date_start_rows[val_idx[0]]
        va_end = date_end_rows[val_idx[-1]]
        tr_df = sorted_data.iloc[tr_start:tr_end]
        va_df = sorted_data.iloc[va_start:va_end].copy()

        model = build_model(resolved_type, resolved_params)
        sample_weight = build_sample_weight(tr_df, sample_weight_mode, date_col=date_col)
        fit_model(
            model,
            resolved_type,
            tr_df,
            features=features,
            target_col=target_col,
            sample_weight=sample_weight,
            date_col=date_col,
        )
        va_df["pred"] = model.predict(va_df[features])
        if signal_direction != 1.0:
            va_df["pred"] = va_df["pred"] * signal_direction

        if date_col == "trade_date":
            ic_input = va_df
        else:
            ic_input = va_df.rename(columns={date_col: "trade_date"})
        ic_values = daily_ic_series(ic_input, target_col, "pred")
        scores.append(float(ic_values.mean()) if not ic_values.empty else np.nan)
    return scores
