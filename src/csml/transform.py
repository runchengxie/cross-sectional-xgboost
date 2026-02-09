from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_cross_sectional_transform(
    data: pd.DataFrame,
    features: list[str],
    method: str,
    winsorize_pct: Optional[float],
) -> pd.DataFrame:
    if method == "none":
        return data

    out = data.copy()
    values = out[features].copy()
    date_index = out["trade_date"]

    if winsorize_pct:
        grouped = values.groupby(date_index, sort=False)
        lower = grouped.transform(lambda s: s.quantile(winsorize_pct))
        upper = grouped.transform(lambda s: s.quantile(1 - winsorize_pct))
        values = values.clip(lower=lower, upper=upper, axis=0)

    if method == "zscore":
        grouped = values.groupby(date_index, sort=False)
        mean = grouped.transform("mean")
        std = grouped.transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
        values = (values - mean) / std
        values = values.fillna(0.0)
    elif method == "rank":
        values = values.groupby(date_index, sort=False).rank(method="average", pct=True) - 0.5

    out[features] = values
    return out
