"""Derive a fixed-tool scene without changing the upstream Panda files."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def reset_home(model, key_name):
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key < 0:
        raise ValueError("Required home keyframe missing")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def build_scene(config, output_xml):
    """Return model, home data, placement report; save portable derived MJCF."""
    radius = config["tool_radius_m"]
    gap = config["initial_gap_m"]
    half_size = np.asarray(config["plate_half_size_m"], dtype=float)
    normal = np.asarray(config["normal_world"], dtype=float)
    if not np.array_equal(normal, [0, 0, 1]):
        raise ValueError("M0-02 supports the horizontal surface with +Z normal only")
    if radius <= 0 or gap <= config["minimum_gap_m"] or config["minimum_gap_m"] <= 0:
        raise ValueError("Require radius > 0 and initial_gap > minimum_gap > 0")
    if np.any(half_size <= 0) or np.any(half_size[:2] <= radius):
        raise ValueError("Plate must have positive thickness and cover the tool")
    source = ROOT / config["robot_xml"]
    original = mujoco.MjModel.from_xml_path(str(source))
    home = reset_home(original, config["home_key"])
    parent = original.body(config["tool_parent"]).id
    rotation = home.xmat[parent].reshape(3, 3)
    # FK maps the configured hand-local sphere center into world coordinates.
    center = home.xpos[parent] + rotation @ np.asarray(config["tool_center_local_m"])
    surface = center - normal * (radius + gap)
    plate_center = surface - normal * half_size[2]

    root = ET.parse(source).getroot()
    root.insert(0, ET.Comment("Derived from Menagerie Panda: added tool, fixed plate and sites; upstream source unchanged. See upstream LICENSE and assets/MODEL_SOURCE.json."))
    root.set("model", "panda_m0_contact_scene")
    output_xml = Path(output_xml)
    # Keep saved XML portable when the complete project is copied to Linux.
    try:
        meshdir = Path(os.path.relpath(source.parent / "assets", output_xml.parent)).as_posix()
    except ValueError:
        # Windows test directories can be on a different drive. Normal project
        # runs always use relative references; external temporary runs cannot.
        meshdir = (source.parent / "assets").as_posix()
    root.find("compiler").set("meshdir", meshdir)
    root.find("option").set("timestep", str(config["physics_dt_s"]))
    world = root.find("worldbody")
    hand = world.find(f".//body[@name='{config['tool_parent']}']")
    tool = ET.SubElement(hand, "body", name="contact_tool")
    contact = dict(contype="1", conaffinity="1", condim="3",
                   friction=numbers(config["contact_friction"]),
                   solref=numbers(config["contact_solref"]),
                   solimp=numbers(config["contact_solimp"]))
    ET.SubElement(tool, "geom", name="tool_shaft", type="capsule",
                  fromto=numbers(config["shaft_start_local_m"] + config["tool_center_local_m"]),
                  size=str(config["shaft_radius_m"]), mass=str(config["shaft_mass_kg"]),
                  rgba="0.45 0.5 0.55 1", **contact)
    ET.SubElement(tool, "geom", name="tool_sphere", type="sphere",
                  pos=numbers(config["tool_center_local_m"]), size=str(radius),
                  mass=str(config["tool_mass_kg"]), rgba="0.95 0.45 0.05 1", **contact)
    # TCP is the sphere center, not its orientation-dependent lowest material point.
    ET.SubElement(tool, "site", name="tool_tip", type="sphere",
                  pos=numbers(config["tool_center_local_m"]), size="0.003", rgba="1 0 0 1")
    plate = ET.SubElement(world, "body", name="contact_surface", pos=numbers(plate_center))
    ET.SubElement(plate, "geom", name="contact_plane", type="box", size=numbers(half_size),
                  rgba="0.2 0.6 0.7 1", **contact)
    ET.SubElement(plate, "site", name="plane_origin", pos=numbers([0, 0, half_size[2]]),
                  size="0.004", rgba="0 1 0 1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05",
                  rgba="0.22 0.25 0.3 1", contype="1", conaffinity="1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(config["render_width"]), offheight=str(config["render_height"]))
    ET.indent(root)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an earlier run's resolved model.
    with output_xml.open("x", encoding="utf-8") as stream:
        stream.write(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    data = reset_home(model, config["home_key"])
    report = {
        "tool_parent": config["tool_parent"], "hand_position_world_m": home.xpos[parent].tolist(),
        "hand_rotation_world_from_local": rotation.tolist(),
        "tool_center_home_world_m": center.tolist(), "tool_tip_definition": "sphere center TCP",
        "tool_radius_m": radius, "added_tool_mass_kg": float(model.body_mass[model.body("contact_tool").id]),
        "surface_origin_world_m": surface.tolist(), "plate_center_world_m": plate_center.tolist(),
        "normal_world": normal.tolist(), "approach_direction_world": (-normal).tolist(),
        "surface_definition": "top face of a fixed finite box, not an infinite plane",
        "plate_half_size_m": half_size.tolist(), "expected_initial_gap_m": gap,
        "tool_geom_id": model.geom("tool_sphere").id,
        "plane_geom_id": model.geom("contact_plane").id,
        "tool_site_id": model.site("tool_tip").id,
    }
    return model, data, report


def geometry_state(model, data, config):
    """Current geometry only: no force measurement and no fabricated force."""
    sphere, plate = model.geom("tool_sphere").id, model.geom("contact_plane").id
    center = data.geom_xpos[sphere].copy()
    surface = data.site("plane_origin").xpos.copy()
    normal = data.geom_xmat[plate].reshape(3, 3)[:, 2].copy()
    radius = float(model.geom_size[sphere, 0])
    gap = float(normal @ (center - surface) - radius)
    distance = mujoco.mj_geomDistance(model, data, sphere, plate,
                                     config["distance_query_max_m"], None)
    local = data.geom_xmat[plate].reshape(3, 3).T @ (center - surface)
    covered = bool(np.all(np.abs(local[:2]) + radius < model.geom_size[plate, :2]))
    return {"time_s": float(data.time),
            **dict(zip(("tool_x_m", "tool_y_m", "tool_z_m"), center.tolist())),
            "normal_gap_m": gap, "geom_distance_m": float(distance),
            "inside_plate_footprint": covered, "ncon": int(data.ncon)}


def check_geometry(state, config):
    if not state["inside_plate_footprint"]:
        raise ValueError("Tool is outside the plate's usable footprint")
    if state["normal_gap_m"] < config["minimum_gap_m"]:
        raise ValueError("Insufficient tool-plane clearance")
    if abs(state["normal_gap_m"] - state["geom_distance_m"]) > config["geometry_tolerance_m"]:
        raise ValueError("Analytic gap disagrees with MuJoCo geometry distance")
    if state["ncon"] != 0:
        raise ValueError("Unexpected contact in the non-contact M0-02 experiment")


def clearance_audit(model, data, config):
    """Query sphere/robot and plate/robot distances, including fixed robot links."""
    rows = []
    tool_body = model.body("contact_tool").id
    sphere, plate = model.geom("tool_sphere").id, model.geom("contact_plane").id
    for first in (sphere, plate):
        for other in range(model.ngeom):
            if other == first or not (model.geom_contype[other] or model.geom_conaffinity[other]):
                continue
            # Shaft-sphere overlap is intentional: both belong to one rigid tool.
            if first == sphere and model.geom_bodyid[other] == tool_body:
                continue
            distance = float(mujoco.mj_geomDistance(model, data, first, other,
                                                   config["distance_query_max_m"], None))
            rows.append({"time_s": float(data.time), "geom1": model.geom(first).name,
                         "geom2": model.geom(other).name or f"geom_{other}",
                         "body2": model.body(int(model.geom_bodyid[other])).name,
                         "distance_m": distance})
    return rows
