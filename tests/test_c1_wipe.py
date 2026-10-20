"""C1-01 最小验证：路径、法向、PI 动态方向及真实仿真反馈链路。"""
import json

import numpy as np
import pandas as pd
import pytest

from common.contact_scene import build_scene
from controllers.force_pi import ForcePI
from experiments.c1_wipe import load_config, motion_metrics, run, wipe_displacement


def test_quintic_round_trip_and_speed():
    c = load_config()
    start, duration, length = c["wipe_start_s"], c["wipe_leg_duration_s"], c["wipe_length_m"]
    for time, position in [(0, 0), (start, 0), (start+duration, length),
                           (start+2*duration, 0), (c["duration_s"], 0)]:
        assert wipe_displacement(time, c) == pytest.approx((position, 0))
    for time, sign in [(start+duration/2, 1), (start+1.5*duration, -1)]:
        assert wipe_displacement(time, c)[1] == pytest.approx(sign*1.875*length/duration)
        h = 1e-5
        derivative = (wipe_displacement(time+h, c)[0]-wipe_displacement(time-h, c)[0])/(2*h)
        assert derivative == pytest.approx(wipe_displacement(time, c)[1], abs=1e-10)
    assert np.hypot(1.875*length/duration, c["pi_offset_speed_m_s"]) < c["max_reference_speed_m_s"]


def test_actual_plate_normal_and_path_coverage(tmp_path):
    c = load_config()
    model, data, _ = build_scene(c, tmp_path/"scene.xml")
    plate = model.geom("contact_plane").id
    rotation = data.geom_xmat[plate].reshape(3, 3)
    normal, tangent = rotation[:, 2], rotation[:, 0]
    assert np.linalg.norm(normal) == pytest.approx(1)
    assert tangent @ normal == pytest.approx(0)
    np.testing.assert_allclose(normal, c["normal_world"])
    np.testing.assert_allclose(tangent, [1, 0, 0])
    assert c["wipe_length_m"]+c["tool_radius_m"] < model.geom_size[plate, 0]


def test_pi_direction_with_bumpless_takeover_and_history():
    c = load_config()
    pi = ForcePI(c)
    pi.reset(1)
    assert pi.kp+pi.integral == pytest.approx(0)
    offsets = [pi.step(5, 4, c["control_dt_s"])[0] for _ in range(50)]
    assert offsets[-1] > offsets[0] > 0  # 低力：增大压入，世界 Z 参考降低。
    before = pi.offset
    offsets = [pi.step(5, 6, c["control_dt_s"])[0] for _ in range(100)]
    assert offsets[0] < before and offsets[-1] < 0  # 高力：先消除积分历史，再向外退让。


def test_real_short_round_trip_and_feedback(tmp_path):
    c = load_config()
    try:
        output = run(c, tmp_path)
    except ValueError as exc:
        if "C1-01 验收未通过" not in str(exc):
            raise
        output = tmp_path/"run_001"  # 保留失败数据，先检查反馈真实性，最后明确报告验收失败。
    frame = pd.read_csv(output/"contact_force.csv")
    commands = pd.read_csv(output/"control.csv")
    combined = pd.read_csv(output/"tracking.csv")
    metrics = json.loads((output/"c1_metrics.json").read_text(encoding="utf-8"))
    assert len(frame) == 17001
    assert metrics["运动采样数"] == 6000
    # 控制输入使用相同时间戳的物理反馈，不是目标力。
    measured = frame.set_index("time_s").loc[commands.time_s, "normal_force_n"]
    np.testing.assert_allclose(commands.measured_force_n, measured, atol=1e-12)
    np.testing.assert_allclose(commands.feedback_time_s, commands.time_s, atol=1e-12)
    np.testing.assert_allclose(commands.target_x_m, commands.path_x_m, atol=1e-12)
    np.testing.assert_allclose(commands.target_z_m, commands.path_z_m-commands.pi_offset_m, atol=1e-12)
    # 每个物理样本匹配严格更早的控制时刻，验证记录的确是已执行输入。
    applied = np.searchsorted(commands.time_s, frame.time_s.iloc[1:], side="left")-1
    np.testing.assert_allclose(combined.target_z_m.iloc[1:], commands.target_z_m.to_numpy()[applied], atol=1e-12)
    np.testing.assert_allclose(combined.pi_offset_m.iloc[1:], commands.pi_offset_m.to_numpy()[applied], atol=1e-12)
    actual = frame[["tool_x_m", "tool_y_m"]].to_numpy()
    path = frame[["path_x_m", "path_y_m"]].to_numpy()
    np.testing.assert_allclose(frame.tangential_error_m, np.linalg.norm(actual-path, axis=1), atol=1e-12)
    assert frame.other_contact_count.max() == 0 and frame.inside_plate_footprint.all()
    assert frame.lateral_drift_m.max() > c["max_lateral_drift_m"]
    assert commands[[f"ctrl_{i}" for i in range(7)]].diff().abs().max().max() <= c["max_joint_reference_speed_rad_s"]*c["control_dt_s"]+1e-12
    # 反例：仅命令在移动、机器人未移动时，验收必须拒绝。
    frozen = frame.copy()
    frozen[["tool_x_m", "tool_y_m"]] = actual[0]
    assert not motion_metrics(frozen, c)["连续接触与实际往返验收"]
    assert metrics["passed"], f"实际连续接触任务未达标：{metrics}"
