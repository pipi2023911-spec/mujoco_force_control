import numpy as np
import pandas as pd
import pytest

from controllers.admittance import Admittance
from experiments.m2b_admittance_force import analyze, load_config, run


def test_target_force_sign_and_k_zero_ideal_spring():
    c = {**load_config(), "admittance_stiffness_n_m": 0., "admittance_damping_ns_m": 3000.}
    controller = Admittance(c)
    offset, state = controller.step(6., .005)
    assert offset > 0  # 外力过大，沿正法向退让。
    assert state["admittance_a_m_s2"] == pytest.approx(1. / c["admittance_mass_kg"])
    assert state["admittance_force_error_n"] == 1.
    for _ in range(2000):
        offset, _ = controller.step(10. - 20000. * controller.position, .005)
    assert 10. - 20000. * offset == pytest.approx(5., abs=1e-9)


@pytest.mark.parametrize("change", [{"admittance_stiffness_n_m": -1.},
                                  {"admittance_force_target_n": np.nan}])
def test_invalid_force_config(change):
    with pytest.raises(ValueError):
        Admittance({**load_config(), **change})


def test_real_admittance_target_and_spring_balance(tmp_path):
    config = load_config()
    output = run(config, tmp_path)
    table = pd.read_csv(output / "comparison.csv")
    assert table.passed.all()
    for index in range(4):
        c = {**config, **list(config["cases"].values())[index],
             "force_target_n": config["admittance_force_target_n"],
             "force_control_start_s": config["admittance_start_s"], "control_mode": "admittance_force"}
        frame = pd.read_csv(output / f"c{index}/run_001/contact_force.csv")
        commands = pd.read_csv(output / f"c{index}/run_001/control.csv")
        active = commands[commands.admittance_active]
        old_x = np.r_[0., active.admittance_x_m.to_numpy()[:-1]]
        old_v = np.r_[0., active.admittance_v_m_s.to_numpy()[:-1]]
        expected = (active.external_force_n - c["admittance_force_target_n"]
                    - c["admittance_damping_ns_m"] * old_v
                    - c["admittance_stiffness_n_m"] * old_x) / c["admittance_mass_kg"]
        np.testing.assert_allclose(active.admittance_a_m_s2, expected, atol=1e-10)
        np.testing.assert_allclose(active.admittance_force_error_n,
                                   active.external_force_n - c["admittance_force_target_n"], atol=1e-12)
        row = analyze(frame, commands, c)
        assert row["平衡与接触检查通过"]
        if c["admittance_stiffness_n_m"] == 0:
            assert row["目标力跟踪通过"]
        else:
            assert not row["目标力跟踪通过"]
            assert abs(-row["末秒平均误差_N"] - row["末秒Kx_N"]) < .01
        corrupted = commands.copy()
        corrupted.loc[corrupted.index[-1], "admittance_limited"] = True
        assert not analyze(frame, corrupted, c)["passed"]
