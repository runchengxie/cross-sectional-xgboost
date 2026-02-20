import pytest
import pandas as pd

from csml.project_tools import build_hk_connect_universe as hk_universe
from csml.project_tools import fetch_index_components as index_components


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("today", "today"),
        ("t", "today"),
        ("t-1", "t-1"),
        ("yesterday", "t-1"),
        ("last_trading_day", "last_trading_day"),
        ("last_completed_trading_day", "last_completed_trading_day"),
        ("20260131", "20260131"),
    ],
)
def test_normalize_date_token(token, expected):
    assert hk_universe.normalize_date_token(token, "end-date") == expected


def test_normalize_date_token_rejects_invalid_date():
    with pytest.raises(SystemExit, match="end-date must be in YYYYMMDD format."):
        hk_universe.normalize_date_token("2026-01-31", "end-date")


def test_format_output_path_appends_date_tag():
    out = hk_universe.format_output_path("out/universe/universe_by_date.csv", "20260131", append_date=True)
    assert str(out) == "out/universe/universe_by_date_20260131.csv"


def test_format_output_path_supports_template():
    out = hk_universe.format_output_path("out/universe/{as_of}/symbols.txt", "20260131", append_date=True)
    assert str(out) == "out/universe/20260131/symbols.txt"


def test_extract_universe_config_normalizes_nested_keys():
    cfg = {
        "hk_connect_universe": {
            "start-date": "20250101",
            "rqdata": {"username": "u", "password": "p"},
        }
    }
    normalized = hk_universe.extract_universe_config(cfg)
    assert normalized["start_date"] == "20250101"
    assert normalized["rqdata_user"] == "u"
    assert normalized["rqdata_pass"] == "p"


def test_resolve_as_of_date_respects_last_trading_variants(monkeypatch):
    calls = []

    def fake_resolve_last_trading_date(rqdatac, as_of, market, include_today):
        calls.append(include_today)
        return pd.Timestamp("2026-01-31")

    monkeypatch.setattr(hk_universe, "resolve_last_trading_date", fake_resolve_last_trading_date)

    hk_universe.resolve_as_of_date(object(), "last_trading_day", "hk")
    hk_universe.resolve_as_of_date(object(), "last_completed_trading_day", "hk")

    assert calls == [True, False]


def test_month_bounds_handles_leap_year():
    assert index_components.month_bounds("202402") == ("20240201", "20240229")


def test_month_bounds_rejects_invalid_month():
    with pytest.raises(SystemExit, match="month must be between 01 and 12."):
        index_components.month_bounds("202413")
