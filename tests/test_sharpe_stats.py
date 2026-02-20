import math

import pytest

from csml.sharpe_stats import (
    annualized_sharpe_to_periodic,
    annualized_variance_to_periodic,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)


def test_annualized_conversion_helpers():
    assert annualized_sharpe_to_periodic(2.0, 4.0) == pytest.approx(1.0)
    assert annualized_variance_to_periodic(0.8, 4.0) == pytest.approx(0.2)
    assert math.isnan(annualized_sharpe_to_periodic(2.0, 0.0))
    assert math.isnan(annualized_variance_to_periodic(0.8, 0.0))


def test_expected_max_sharpe_increases_with_trials():
    small_n = expected_max_sharpe(n_trials=10, var_sharpe=0.02)
    large_n = expected_max_sharpe(n_trials=50, var_sharpe=0.02)
    assert large_n > small_n > 0.0


def test_deflated_sharpe_ratio_behaviour():
    higher_dsr, sr0 = deflated_sharpe_ratio(
        sharpe=0.6,
        periods=60,
        skew=0.0,
        kurtosis_excess=0.0,
        n_trials=20,
        var_sharpe=0.05,
    )
    lower_dsr, _ = deflated_sharpe_ratio(
        sharpe=0.3,
        periods=60,
        skew=0.0,
        kurtosis_excess=0.0,
        n_trials=20,
        var_sharpe=0.05,
    )
    assert 0.0 <= higher_dsr <= 1.0
    assert sr0 > 0.0
    assert higher_dsr > lower_dsr
