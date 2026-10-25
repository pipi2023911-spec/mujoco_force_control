import numpy as np
import pytest

from controllers.fuzzy_pi import FuzzyAdaptivePI
from controllers.force_pi import ForcePI
from experiments.m1_force_pi import load_config
from experiments.m1_force_pi import run
import pandas as pd


def test_rules_and_membership_partition():
    pi = FuzzyAdaptivePI(load_config())
    for x in np.linspace(-2, 2, 41):
        assert pi.memberships(x).sum() == pytest.approx(1)
        assert (pi.memberships(x) >= 0).all()
    np.testing.assert_array_equal(pi.memberships(-1), [1, 0, 0])
    np.testing.assert_array_equal(pi.memberships(0), [0, 1, 0])
    # 正误差继续扩大时更积极；向目标接近时抑制积分。
    assert pi.KI_RULES[2, 2] > pi.KI_RULES[2, 0]


def test_gain_update_is_applied_and_bumpless():
    c = load_config()
    pi = FuzzyAdaptivePI(c)
    pi.reset(.5)
    old_p_plus_i = pi.kp * .5 + pi.integral
    output, info = pi.step(5, 4.5, .005)
    assert pi.kp != c["pi_kp_m_per_n"]
    assert pi.ki != c["pi_ki_m_per_ns"]
    assert info["pi_p_m"] == pytest.approx(pi.kp * .5)
    assert output == pytest.approx(old_p_plus_i + pi.ki * .5 * .005)


def test_bounded_gains_saturation_and_reversal():
    c = load_config()
    pi = FuzzyAdaptivePI(c)
    previous = 0.
    for force in [-100.] * 1000 + [100.] * 1000:
        output, info = pi.step(5, force, .005)
        assert pi.kp_bounds[0] <= pi.kp <= pi.kp_bounds[1]
        assert pi.ki_bounds[0] <= pi.ki <= pi.ki_bounds[1]
        assert abs(output) <= pi.offset_limit
        assert abs(pi.integral) <= pi.integral_limit
        assert abs(output - previous) <= pi.speed * .005 + 1e-15
        previous = output
    assert output == pytest.approx(-pi.offset_limit)
    assert info["pi_integral_blocked"]


def test_fixed_bounds_reproduce_fixed_pi():
    c = load_config()
    fuzzy = FuzzyAdaptivePI({**c, "fuzzy_kp_bounds": [c["pi_kp_m_per_n"]] * 2,
                            "fuzzy_ki_bounds": [c["pi_ki_m_per_ns"]] * 2})
    fixed = ForcePI(c)
    fuzzy.reset(.5)
    fixed.reset(.5)
    for force in np.linspace(4.5, 5.5, 100):
        a, _ = fuzzy.step(5, force, .005)
        b, _ = fixed.step(5, force, .005)
        assert a == b


@pytest.mark.parametrize("change", [{"fuzzy_error_scale_n": 0}, {"fuzzy_rate_scale_n_s": np.nan},
                                   {"fuzzy_gain_tau_s": -1}, {"fuzzy_kp_bounds": [3e-5, 4e-5]}])
def test_invalid_parameters(change):
    with pytest.raises(ValueError):
        FuzzyAdaptivePI({**load_config(), **change})


def test_invalid_measurement_does_not_change_state():
    pi = FuzzyAdaptivePI(load_config())
    before = (pi.kp, pi.ki, pi.integral, pi.previous_error)
    with pytest.raises(ValueError):
        pi.step(5, np.nan, .005)
    assert (pi.kp, pi.ki, pi.integral, pi.previous_error) == before


def test_real_fuzzy_loop_uses_online_gains_and_keeps_limits(tmp_path):
    c = {**load_config(), "pi_algorithm": "fuzzy"}
    output = run(c, tmp_path, mode="pi")
    commands = pd.read_csv(output / "pi/run_001/control.csv")
    active = commands[commands.pi_active]
    assert np.ptp(active.pi_kp_m_per_n) > 1e-9
    assert np.ptp(active.pi_ki_m_per_ns) > 1e-7
    np.testing.assert_allclose(active.pi_p_m, active.pi_kp_m_per_n * active.force_error_n, atol=1e-15)
    assert active.pi_kp_m_per_n.between(1e-5, 3e-5).all()
    assert active.pi_ki_m_per_ns.between(1.5e-4, 4.5e-4).all()
    assert commands.pi_offset_m.abs().max() <= c["pi_offset_limit_m"]
    assert commands.pi_i_m.abs().max() <= c["pi_integral_limit_m"]
    assert commands.pi_offset_m.diff().abs().max() <= c["pi_offset_speed_m_s"] * c["control_dt_s"] + 1e-12
