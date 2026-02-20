from __future__ import annotations

import math

try:
    from scipy.stats import norm as scipy_norm
except Exception:  # pragma: no cover - optional dependency
    scipy_norm = None

_SQRT_TWO = math.sqrt(2.0)
_EULER_GAMMA = getattr(math, "euler_gamma", 0.5772156649015329)


def norm_cdf(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    if scipy_norm is not None:
        return float(scipy_norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / _SQRT_TWO))


def _acklam_ppf(p: float) -> float:
    # Coefficients from Peter J. Acklam's inverse-normal approximation.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def norm_ppf(p: float) -> float:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return math.nan
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if scipy_norm is not None:
        return float(scipy_norm.ppf(p))
    return _acklam_ppf(p)


def annualized_sharpe_to_periodic(sharpe_annualized: float, periods_per_year: float) -> float:
    try:
        sharpe_annualized = float(sharpe_annualized)
        periods_per_year = float(periods_per_year)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(sharpe_annualized):
        return math.nan
    if not math.isfinite(periods_per_year) or periods_per_year <= 0:
        return math.nan
    return sharpe_annualized / math.sqrt(periods_per_year)


def annualized_variance_to_periodic(variance_annualized: float, periods_per_year: float) -> float:
    try:
        variance_annualized = float(variance_annualized)
        periods_per_year = float(periods_per_year)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(variance_annualized):
        return math.nan
    if not math.isfinite(periods_per_year) or periods_per_year <= 0:
        return math.nan
    return variance_annualized / periods_per_year


def sharpe_standard_error(
    sharpe: float,
    periods: float,
    skew: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    try:
        sharpe = float(sharpe)
        periods = float(periods)
        skew = float(skew)
        kurtosis_excess = float(kurtosis_excess)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(sharpe):
        return math.nan
    if not math.isfinite(periods) or periods <= 1:
        return math.nan
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurtosis_excess):
        kurtosis_excess = 0.0

    variance_term = (
        1.0
        - skew * sharpe
        + 0.25 * (kurtosis_excess + 2.0) * sharpe * sharpe
    )
    if not math.isfinite(variance_term) or variance_term <= 0:
        return math.nan
    return math.sqrt(variance_term / (periods - 1.0))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    try:
        n_trials = int(n_trials)
        var_sharpe = float(var_sharpe)
    except (TypeError, ValueError):
        return math.nan
    if n_trials < 2:
        return math.nan
    if not math.isfinite(var_sharpe) or var_sharpe <= 0:
        return math.nan
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    if not math.isfinite(z1) or not math.isfinite(z2):
        return math.nan
    return math.sqrt(var_sharpe) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    periods: float,
    skew: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    try:
        sharpe = float(sharpe)
        benchmark_sharpe = float(benchmark_sharpe)
    except (TypeError, ValueError):
        return math.nan
    se = sharpe_standard_error(
        sharpe=sharpe,
        periods=periods,
        skew=skew,
        kurtosis_excess=kurtosis_excess,
    )
    if not math.isfinite(se) or se <= 0:
        return math.nan
    z_score = (sharpe - benchmark_sharpe) / se
    return norm_cdf(z_score)


def deflated_sharpe_ratio(
    sharpe: float,
    periods: float,
    skew: float,
    kurtosis_excess: float,
    n_trials: int,
    var_sharpe: float,
) -> tuple[float, float]:
    sr0 = expected_max_sharpe(n_trials=n_trials, var_sharpe=var_sharpe)
    if not math.isfinite(sr0):
        return math.nan, math.nan
    dsr = probabilistic_sharpe_ratio(
        sharpe=sharpe,
        benchmark_sharpe=sr0,
        periods=periods,
        skew=skew,
        kurtosis_excess=kurtosis_excess,
    )
    return dsr, sr0
