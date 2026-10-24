import json

import numpy as np
import pandas as pd
import pytest

from controllers.force_pi import ForcePI
from experiments.m1_force_pi import ForceReference, load_config, run


def test_pi_saturation_rate_limit_and_reversal():
    c = load_config()
    pi = ForcePI(c)
    outputs = [0.0]
    for _ in range(1000):
        output, info = pi.step(5, -100, c["control_dt_s"])
        outputs.append(output)
    assert output == pytest.approx(c["pi_offset_limit_m"])
    assert pi.integral == 0  # 持续饱和没有积累隐藏积分。
    assert info["pi_integral_blocked"]
    assert np.max(np.abs(np.diff(outputs))) <= c["pi_offset_speed_m_s"] * c["control_dt_s"] + 1e-15
    for _ in range(1000):
        output, _ = pi.step(5, 100, c["control_dt_s"])
    assert output == pytest.approx(-c["pi_offset_limit_m"])
    assert abs(pi.integral) <= c["pi_integral_limit_m"]


def test_pi_bumpless_sign_and_invalid_measurement():
    c = load_config()
    pi = ForcePI(c)
    pi.reset(-1)
    assert pi.kp * -1 + pi.integral == pytest.approx(0)
    output, _ = pi.step(5, 6, c["control_dt_s"])
    assert output < 0  # 力过大则退出；不足则压入。
    pi.reset(1)
    assert pi.step(5, 4, c["control_dt_s"])[0] > 0
    with pytest.raises(ValueError):
        pi.step(5, np.nan, c["control_dt_s"])


def test_no_contact_cannot_enable_force_control():
    c = load_config()
    reference = ForceReference(c, "pi")
    with pytest.raises(ValueError, match="持续接触"):
        reference(c["force_control_start_s"], np.zeros(3), np.array([0,0,1]), c,
                  {"normal_force_n": 0., "time_s": c["force_control_start_s"]})


def test_real_pi_vs_baseline_and_feedback_trace(tmp_path):
    c = load_config()
    output = run(c, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    base = pd.read_csv(output / "baseline/run_001/contact_force.csv")
    pi = pd.read_csv(output / "pi/run_001/contact_force.csv")
    commands = pd.read_csv(output / "pi/run_001/control.csv")
    assert len(pi) == 12001 and len(commands) == 2400
    pre = base.time_s < c["force_control_start_s"]
    np.testing.assert_allclose(base.loc[pre, "normal_force_n"], pi.loc[pre, "normal_force_n"], atol=1e-12)
    assert (commands.loc[commands.time_s < c["force_control_start_s"] - 1e-10, "pi_offset_m"] == 0).all()
    # 证明反馈来自同一时间戳的真实物理日志，不是目标力或人为曲线。
    np.testing.assert_allclose(commands.feedback_time_s, commands.time_s, atol=1e-12)
    logged = pi.set_index("time_s").loc[commands.time_s, "normal_force_n"]
    np.testing.assert_allclose(commands.measured_force_n, logged.to_numpy(), atol=1e-12)
    np.testing.assert_array_equal(commands.sensor_true_force_n, commands.sensor_filtered_force_n)
    np.testing.assert_allclose(commands.sensor_source_time_s, commands.time_s, atol=1e-12)
    np.testing.assert_allclose(commands.force_error_n, c["force_target_n"] - commands.measured_force_n, atol=1e-12)
    assert commands.pi_offset_m.abs().max() <= c["pi_offset_limit_m"]
    assert commands.pi_i_m.abs().max() <= c["pi_integral_limit_m"]
    assert commands.pi_offset_m.diff().abs().max() <= c["pi_offset_speed_m_s"] * c["control_dt_s"] + 1e-12
    assert summary["PI与Baseline末秒RMSE比值"] < c["comparison_rmse_ratio_limit"]


def test_sensor_channel_reaches_pi_and_preserves_truth(tmp_path):
    c = {**load_config(), "sensor_noise_std_n": .01, "sensor_bias_n": .01,
         "sensor_delay_steps": 2, "sensor_lpf_cutoff_hz": 20., "sensor_seed": 42}
    output = run(c, tmp_path, mode="pi")
    commands = pd.read_csv(output / "pi/run_001/control.csv")
    truth = pd.read_csv(output / "pi/run_001/contact_force.csv")
    np.testing.assert_allclose(commands.measured_force_n, commands.sensor_filtered_force_n, atol=1e-12)
    logged = truth.set_index("time_s").loc[commands.time_s, "normal_force_n"]
    np.testing.assert_allclose(commands.sensor_true_force_n, logged.to_numpy(), atol=1e-12)
    expected = commands.sensor_true_force_n.to_numpy()[np.maximum(np.arange(len(commands)) - 2, 0)]
    np.testing.assert_allclose(commands.sensor_delayed_force_n, expected, atol=1e-12)
    np.testing.assert_allclose(commands.sensor_actual_delay_s.iloc[2:], .01, atol=1e-12)
    assert not np.allclose(commands.measured_force_n, commands.sensor_true_force_n)
    np.testing.assert_allclose(truth.force_error_n, c["force_target_n"] - truth.normal_force_n, atol=1e-12)
    assert (output / "sensor_force.png").exists()
