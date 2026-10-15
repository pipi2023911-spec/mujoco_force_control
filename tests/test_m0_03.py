import json

import mujoco
import numpy as np
import pandas as pd
import pytest

from common.contact_feedback import contact_force
from common.kinematics import approach_reference
from experiments.m0_03_contact_force import load_config, run


def test_zero_contact_and_multiple_points_ignore_other_body():
    # 两个独立物体都与地面接触；只允许累计选中的方块，不混入另一个球。
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body pos="0 0 .2"><freejoint/><geom name="tool" type="box" size=".05 .05 .05" mass="1"/></body>
      <body pos=".5 0 .2"><freejoint/><geom name="other" size=".05" mass="2"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tool, floor = model.geom("tool").id, model.geom("floor").id
    empty = contact_force(model, data, tool, floor, [0, 0, 1])
    np.testing.assert_array_equal(empty["force_world_n"], np.zeros(3))
    assert empty["tool_contact_count"] == 0 and empty["points"] == []
    mujoco.mj_step(model, data, nstep=2000)
    mujoco.mj_forward(model, data)
    measured = contact_force(model, data, tool, floor, [0, 0, 1])
    assert measured["tool_contact_count"] >= 2
    assert data.ncon > measured["tool_contact_count"]
    assert measured["normal_force_n"] == pytest.approx(9.81, abs=0.01)
    assert sum(p["tool_force_z_n"] for p in measured["points"]) == pytest.approx(measured["normal_force_n"])
    excluded = contact_force(model, data, tool, model.geom("other").id, [0, 0, 1])
    assert excluded["normal_force_n"] == 0


def test_contact_frame_rotation_and_action_reaction():
    # 用真实倾斜平面和沿其法向的重力检验旋转，避免只测单位矩阵。
    normal = np.array([np.sqrt(0.5), 0, np.sqrt(0.5)])
    gravity = " ".join(map(str, -9.81 * normal))
    position = " ".join(map(str, 0.2 * normal))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".001" gravity="{gravity}"/>
      <worldbody><geom name="plane" type="plane" size="2 2 .1" quat=".9238795325 0 .3826834324 0"/>
      <body pos="{position}"><freejoint/><geom name="ball" size=".05" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=2000)
    mujoco.mj_forward(model, data)
    ball, plane = model.geom("ball").id, model.geom("plane").id
    force = contact_force(model, data, ball, plane, normal)
    np.testing.assert_allclose(force["force_world_n"], 9.81 * normal, atol=0.01)
    opposite = contact_force(model, data, plane, ball, normal)
    np.testing.assert_allclose(opposite["force_world_n"], -force["force_world_n"], atol=1e-12)


def test_reference_is_smooth_bounded_and_independent_of_force():
    config = load_config()
    start = np.array([0.5, 0.0, 0.4])
    times = np.arange(0, config["duration_s"], config["control_dt_s"])
    positions = np.array([approach_reference(t, start, np.array([0, 0, 1]), config)[0] for t in times])
    np.testing.assert_allclose(positions[0], start)
    np.testing.assert_allclose(positions[-1], start - [0, 0, config["approach_distance_m"]])
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) / config["control_dt_s"] < config["max_reference_speed_m_s"]


def test_real_panda_contact_run_and_raw_force_reconstruction(tmp_path):
    config = load_config()
    output = run(config, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    force = pd.read_csv(output / "contact_force.csv")
    raw = pd.read_csv(output / "raw_contacts.csv")
    commands = pd.read_csv(output / "control.csv")
    assert summary["passed"] and len(force) == 10001 and len(commands) == 2000
    np.testing.assert_allclose(np.diff(force.time_s), .001, atol=1e-12)
    np.testing.assert_allclose(np.diff(commands.time_s), .005, atol=1e-12)
    assert (force.loc[force.time_s < config["settle_duration_s"], "normal_force_n"] == 0).all()
    assert (force.loc[force.tool_contact_count == 0, "normal_force_n"] == 0).all()
    assert (force.other_contact_count == 0).all()
    assert summary["peak_force_n"] < config["max_measured_force_n"]
    assert summary["hold_mean_force_n"] > config["minimum_hold_mean_force_n"]
    assert summary["hold_contact_fraction"] >= config["minimum_hold_contact_fraction"]
    for _, row in raw.iloc[::500].iterrows():
        frame = np.array([row[f"frame_{i}{j}"] for i in range(3) for j in range(3)]).reshape(3, 3)
        local = row[["local_force_normal_n", "local_force_tangent1_n", "local_force_tangent2_n"]].to_numpy(dtype=float)
        expected = row.tool_force_sign * frame.T @ local
        np.testing.assert_allclose(expected, row[[f"tool_force_{a}_n" for a in "xyz"]].to_numpy(dtype=float), atol=1e-12)
    summed = raw.groupby("time_s").tool_force_z_n.sum()
    actual = force.set_index("time_s").loc[summed.index, "normal_force_n"]
    np.testing.assert_allclose(summed.to_numpy(), actual.to_numpy(), atol=1e-12)
    assert (output / "contact_force.png").exists()
    assert (output / "summary_中文.json").exists()


def test_force_limit_failure_keeps_measured_sample(tmp_path):
    config = load_config()
    config["max_measured_force_n"] = 0.01  # 故障注入：故意设置很小的力上限。
    with pytest.raises(ValueError, match="真实接触力超过"):
        run(config, tmp_path)
    output = tmp_path / "run_001"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert not summary["passed"]
    frame = pd.read_csv(output / "contact_force.csv")
    assert frame.normal_force_n.iloc[-1] > config["max_measured_force_n"]
    assert (output / "raw_contacts.csv").exists()
