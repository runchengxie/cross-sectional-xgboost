import numpy as np
import pandas as pd

from csxgb.pipeline import build_walk_forward_windows


def test_walk_forward_anchor_end_with_gap():
    dates = np.array(pd.date_range("2020-01-01", periods=10, freq="D"))
    windows = build_walk_forward_windows(
        dates,
        test_size=0.2,
        n_windows=2,
        step_size=None,
        gap_days=1,
        anchor_end=True,
    )
    assert len(windows) == 2
    first = windows[0]
    second = windows[1]
    assert first["train_end"] == dates[4]
    assert first["test_start"] == dates[6]
    assert first["test_end"] == dates[7]
    assert second["train_end"] == dates[6]
    assert second["test_start"] == dates[8]
    assert second["test_end"] == dates[9]


def test_walk_forward_step_size_float_anchor_end():
    dates = np.array(pd.date_range("2020-01-01", periods=12, freq="D"))
    windows = build_walk_forward_windows(
        dates,
        test_size=0.25,
        n_windows=3,
        step_size=0.25,
        gap_days=0,
        anchor_end=True,
    )
    assert len(windows) == 3
    assert windows[0]["test_start"] == dates[3]
    assert windows[1]["test_start"] == dates[6]
    assert windows[2]["test_start"] == dates[9]
    assert windows[2]["test_end"] == dates[11]
