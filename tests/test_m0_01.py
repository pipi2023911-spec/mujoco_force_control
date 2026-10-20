import json

import mujoco
import numpy as np
import pandas as pd
import pytest

from common.robot_state import check_state, inspect_model
from common.contact_scene import load_model
from experiments.m0_01_model_inspection import ROOT, new_run, run


@pytest.fixture
def loaded():
    config = json.loads((ROOT / "configs/m0_01.json").read_text(encoding="utf-8"))
    model = load_model(ROOT / config["model"])  # 与入口一致，兼容 Windows 中文工作目录。
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data, config


def test_actual_panda_mapping(loaded):
    m, d, config = loaded
    report = inspect_model(m, d, config)
    assert (m.njnt, m.nq, m.nv, m.nu) == (9, 9, 9, 8)
    assert [j["qpos_address"] for j in report["joints"]] == list(range(9))
    assert [j["qvel_address"] for j in report["joints"]] == list(range(9))
    assert [a["target"] for a in report["actuators"]] == config["arm_joints"] + ["split"]
    assert report["tendons"][0]["terms"] == [
        {"joint": "finger_joint1", "coefficient": 0.5},
        {"joint": "finger_joint2", "coefficient": 0.5}]
    assert m.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert {m.joint(int(m.eq_obj1id[0])).name, m.joint(int(m.eq_obj2id[0])).name} == set(config["finger_joints"])
    np.testing.assert_allclose(d.ctrl[:7], d.qpos[:7])
    assert d.ctrl[7] == 255
    for a in range(7):
        assert m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
        assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
        assert m.actuator_gainprm[a, 0] == -m.actuator_biasprm[a, 1]


@pytest.mark.parametrize("fault", ["nan", "joint_limit", "ctrl_limit", "warning"])
def test_bad_state_is_rejected(loaded, fault):
    m, d, config = loaded
    if fault == "nan":
        d.qvel[0] = np.nan
    elif fault == "joint_limit":
        d.qpos[0] = m.jnt_range[0, 1] + 0.1
    elif fault == "ctrl_limit":
        d.ctrl[0] = m.actuator_ctrlrange[0, 1] + 0.1
    else:
        d.warning[0].number = 1
    with pytest.raises(ValueError):
        check_state(m, d, config["joint_limit_tolerance"])


def test_two_second_simulation_and_logs(loaded, tmp_path):
    _, _, config = loaded
    output = run(config, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    frame = pd.read_csv(output / "joint_state.csv")
    commands = pd.read_csv(output / "control.csv")
    assert len(frame) == 2001 and len(commands) == 400
    np.testing.assert_allclose(np.diff(frame.time_s), 0.001, atol=1e-12)
    np.testing.assert_allclose(np.diff(commands.time_s), 0.005, atol=1e-12)
    assert frame.time_s.iloc[-1] == pytest.approx(2.0)
    assert np.isfinite(frame.to_numpy()).all()
    assert new_run(tmp_path) != output
    assert (output / "joint_state.csv").exists()


def test_failure_is_preserved(loaded, tmp_path):
    _, _, config = loaded
    config["home_key"] = "missing"
    with pytest.raises(ValueError, match="home"):
        run(config, tmp_path)
    summary = json.loads((tmp_path / "run_001" / "summary.json").read_text(encoding="utf-8"))
    assert not summary["passed"] and "home" in summary["error"]
