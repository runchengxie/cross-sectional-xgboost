from csml import pipeline


def test_warn_if_purge_too_small_emits_warning(caplog):
    caplog.set_level("WARNING", logger="csml")
    pipeline._warn_if_purge_too_small(
        purge_days_cfg=0,
        purge_days=0,
        label_horizon_effective=5,
        label_shift_days=1,
    )
    assert any(
        "eval.purge_days=0 is smaller than label span (6 = horizon_effective 5 + shift_days 1)"
        in record.getMessage()
        for record in caplog.records
    )


def test_warn_if_purge_too_small_skips_for_default_behavior(caplog):
    caplog.set_level("WARNING", logger="csml")
    pipeline._warn_if_purge_too_small(
        purge_days_cfg=None,
        purge_days=6,
        label_horizon_effective=5,
        label_shift_days=1,
    )
    assert not any("label leakage" in record.getMessage() for record in caplog.records)


def test_warn_if_delay_exit_lag_emits_warning(caplog):
    caplog.set_level("WARNING", logger="csml")
    pipeline._warn_if_delay_exit_lag(
        label_prefix="[test] ",
        exit_price_policy="delay",
        stats={
            "periods": 10,
            "periods_with_delayed_exit": 3,
            "avg_exit_lag_days": 1.4,
            "max_exit_lag_days": 4,
        },
    )
    assert any(
        "Delay exit policy produced lagged exits in 3/10 periods" in record.getMessage()
        for record in caplog.records
    )


def test_warn_if_delay_exit_lag_skips_non_delay_policy(caplog):
    caplog.set_level("WARNING", logger="csml")
    pipeline._warn_if_delay_exit_lag(
        label_prefix="",
        exit_price_policy="strict",
        stats={
            "periods": 10,
            "periods_with_delayed_exit": 3,
            "avg_exit_lag_days": 1.4,
            "max_exit_lag_days": 4,
        },
    )
    assert not any("Delay exit policy produced lagged exits" in record.getMessage() for record in caplog.records)
