import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pandas as pd
import pytest

from common.contact_scene import build_scene, check_geometry, geometry_state, clearance_audit, load_model
from experiments.m0_02_contact_scene import load_config, run


@pytest.fixture
def scene(tmp_path):
    config = load_config()
    model, data, report = build_scene(config, tmp_path / "scene.xml")
    return config, model, data, report


def test_geometry_fk_and_fixed_bodies(scene):
    config, m, d, report = scene
    assert (m.njnt, m.nq, m.nv, m.nu) == (9, 9, 9, 8)
    hand, tool, plate = (m.body(name).id for name in ("hand", "contact_tool", "contact_surface"))
    assert m.body_parentid[tool] == hand
    assert m.body_jntnum[tool] == 0 and m.body_jntnum[plate] == 0
    assert m.body_mocapid[plate] == -1 and m.body_parentid[plate] == 0
    assert m.body_mass[tool] == pytest.approx(0.07)
    expected = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ config["tool_center_local_m"]
    np.testing.assert_allclose(d.site("tool_tip").xpos, expected, atol=1e-12)
    np.testing.assert_allclose(d.geom("tool_sphere").xpos, expected, atol=1e-12)
    state = geometry_state(m, d, config)
    assert state["normal_gap_m"] == pytest.approx(0.02, abs=1e-12)
    assert state["geom_distance_m"] == pytest.approx(0.02, abs=1e-12)
    check_geometry(state, config)
    audit = clearance_audit(m, d, config)
    assert all(row["distance_m"] >= 0 for row in audit)
    assert any(row["body2"] == "left_finger" for row in audit)
    for name in ("tool_sphere", "tool_shaft", "contact_plane"):
        g = m.geom(name).id
        assert m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 1


def test_surface_follows_fk_not_hardcoded_height(tmp_path):
    config = load_config()
    config["tool_center_local_m"] = [0, 0, 0.18]
    config["initial_gap_m"] = 0.03
    m, d, report = build_scene(config, tmp_path / "scene.xml")
    assert geometry_state(m, d, config)["normal_gap_m"] == pytest.approx(0.03)
    expected_z = report["tool_center_home_world_m"][2] - config["tool_radius_m"] - 0.03
    assert d.site("plane_origin").xpos[2] == pytest.approx(expected_z)
    # Saved scene must reload with all external meshes present.
    reloaded = load_model(tmp_path / "scene.xml")
    assert reloaded.ngeom == m.ngeom


def test_collision_is_active_and_penetration_rejected(scene):
    config, m, d, _ = scene
    # Static geometry fault injection only; do not run contact motion or read force.
    m.body_pos[m.body("contact_surface").id, 2] += 0.025
    mujoco.mj_forward(m, d)
    state = geometry_state(m, d, config)
    assert state["normal_gap_m"] < 0 and state["ncon"] > 0
    pair = {m.geom("tool_sphere").id, m.geom("contact_plane").id}
    assert any({c.geom1, c.geom2} == pair for c in d.contact[:d.ncon])
    with pytest.raises(ValueError, match="clearance"):
        check_geometry(state, config)


def test_tool_outside_plate_is_rejected(scene):
    config, m, d, _ = scene
    m.body_pos[m.body("contact_surface").id, 0] += 0.5
    mujoco.mj_forward(m, d)
    with pytest.raises(ValueError, match="footprint"):
        check_geometry(geometry_state(m, d, config), config)


def test_two_second_hold_and_fresh_fk_logs(tmp_path):
    config = load_config()
    output = run(config, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] and not summary["snapshot_requested"]
    geometry = pd.read_csv(output / "geometry.csv")
    states = pd.read_csv(output / "joint_state.csv")
    commands = pd.read_csv(output / "control.csv")
    assert len(geometry) == len(states) == 2001 and len(commands) == 400
    np.testing.assert_allclose(np.diff(geometry.time_s), 0.001, atol=1e-12)
    np.testing.assert_allclose(np.diff(commands.time_s), 0.005, atol=1e-12)
    assert (geometry.ncon == 0).all() and geometry.normal_gap_m.min() > config["minimum_gap_m"]
    m = load_model(output / "contact_scene.xml")
    d = mujoco.MjData(m)
    # Independent FK recomputation detects a one-step timestamp/geometry lag.
    for row_index in (0, 100, 500, 2000):
        d.qpos[:] = states.iloc[row_index][[f"qpos_{i}" for i in range(m.nq)]].to_numpy()
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.site("tool_tip").xpos,
                                   geometry.iloc[row_index][["tool_x_m", "tool_y_m", "tool_z_m"]].to_numpy(dtype=float), atol=1e-12)


def test_failure_saved_and_run_not_overwritten(tmp_path):
    config = load_config()
    config["initial_gap_m"] = -0.001
    for _ in range(2):
        with pytest.raises(ValueError, match="initial_gap"):
            run(config, tmp_path)
    for name in ("run_001", "run_002"):
        summary = json.loads((tmp_path / name / "summary.json").read_text(encoding="utf-8"))
        assert not summary["passed"] and "initial_gap" in summary["error"]
