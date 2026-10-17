import json

import numpy as np
import pandas as pd
import pytest

from controllers.admittance import Admittance
from experiments.m2_admittance import load_config, run, analyze


def test_admittance_constant_force_analytic_convergence():
    c = load_config()
    mass, damping, stiffness = c["admittance_mass_kg"], c["admittance_damping_ns_m"], c["admittance_stiffness_n_m"]
    r1, r2 = np.roots([mass, damping, stiffness])
    duration = .1
    exact = (1 + (r2*np.exp(r1*duration)-r1*np.exp(r2*duration))/(r1-r2)) / stiffness
    errors = []
    for dt in [.005, .0025]:
        controller = Admittance(c)
        for _ in range(round(duration/dt)):
            x, state = controller.step(1., dt)  # 单元测试的已知输入，不作机器人实验结果。
            assert not state["admittance_limited"]
        errors.append(abs(x-exact))
    assert errors[1] < errors[0] < 5e-6


def test_admittance_limits_and_unstable_step_rejected():
    c = load_config()
    controller = Admittance(c)
    for _ in range(500):
        x, state = controller.step(1000, .005)
    assert x == pytest.approx(c["admittance_offset_limit_m"])
    assert state["admittance_limited"] and controller.velocity == 0
    x2, _ = controller.step(-1000, .005)
    assert x2 < x
    with pytest.raises(ValueError, match="稳定条件"):
        controller.step(1, .1)
    with pytest.raises(ValueError):
        controller.step(np.nan, .005)


def test_real_admittance_feedback_and_discrete_equation(tmp_path):
    c = load_config()
    output = run(c, tmp_path, case="reference")
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["passed"]
    path = output / "reference/run_001"
    frame, command = pd.read_csv(path/"contact_force.csv"), pd.read_csv(path/"control.csv")
    assert len(frame) == 12001 and len(command) == 2400
    np.testing.assert_allclose(command.external_force_n, frame.set_index("time_s").loc[command.time_s,"normal_force_n"], atol=1e-12)
    np.testing.assert_allclose(command.feedback_time_s, command.time_s, atol=1e-12)
    active = command[command.admittance_active]
    old_x = np.r_[0., active.admittance_x_m.to_numpy()[:-1]]
    old_v = np.r_[0., active.admittance_v_m_s.to_numpy()[:-1]]
    expected = (active.external_force_n - c["admittance_damping_ns_m"]*old_v - c["admittance_stiffness_n_m"]*old_x)/c["admittance_mass_kg"]
    np.testing.assert_allclose(active.admittance_a_m_s2, expected, atol=1e-10)
    np.testing.assert_allclose(active.admittance_x_m, old_x + active.admittance_v_m_s*c["control_dt_s"], atol=1e-12)
    assert not active.admittance_limited.any()
    assert report["对照结果"][0]["末秒参考退让_mm"] > 0
    corrupted = command.copy()
    corrupted.loc[corrupted.index[-1],"admittance_limited"] = True
    with pytest.raises(ValueError, match="限幅"):
        analyze(frame, corrupted, c, "reference")
