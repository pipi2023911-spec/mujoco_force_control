"""M0-02: FK-based contact scene placement and non-contact hold verification."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np
import pandas as pd

from common.contact_scene import build_scene, geometry_state, check_geometry, clearance_audit
from common.robot_state import check_state, inspect_model
from experiments.m0_01_model_inspection import ROOT, new_run, save_json


def load_config(path=Path("configs/m0_02.json")):
    scene = json.loads((ROOT / path).read_text(encoding="utf-8"))
    base = json.loads((ROOT / scene["base_config"]).read_text(encoding="utf-8"))
    return {**base, **scene}  # Snapshot the resolved values, not just a base-file reference.


def configure_camera(camera, config):
    camera.lookat[:] = config["camera_lookat_m"]
    camera.distance = config["camera_distance_m"]
    camera.azimuth = config["camera_azimuth_deg"]
    camera.elevation = config["camera_elevation_deg"]


def run(config, output_root=None, viewer=False, snapshot=False):
    output = new_run(output_root or ROOT / "results" / "M0-02")
    save_json(output / "config.json", config)
    summary = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "pandas")},
               "viewer_requested": viewer, "snapshot_requested": snapshot}
    geometry, states, commands, audit = [], [], [], []
    sim_time = 0.0
    log = (output / "run.log").open("x", encoding="utf-8")
    def message(text):
        log.write(f"sim_time_s={sim_time:.6f} {text}\n")
        log.flush()
    try:
        dt, period, duration = config["physics_dt_s"], config["control_dt_s"], config["duration_s"]
        if not all(np.isfinite([dt, period, duration])) or dt <= 0 or period < dt or duration <= 0:
            raise ValueError("Invalid time configuration")
        steps, substeps = round(duration / dt), round(period / dt)
        if not np.isclose(steps * dt, duration) or not np.isclose(substeps * dt, period):
            raise ValueError("Time periods must be integer multiples of physics timestep")
        model, data, placement = build_scene(config, output / "contact_scene.xml")
        save_json(output / "scene_report.json", placement)
        save_json(output / "model_report.json", inspect_model(model, data, config))
        summary["git_head_at_run"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        summary["git_worktree_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
        summary["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for folder in ("common", "experiments") for p in sorted((ROOT / folder).glob("*.py"))}
        asset_dir = (ROOT / config["robot_xml"]).parent
        summary["asset_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(asset_dir.rglob("*")) if p.is_file()}
        summary["scene_sha256"] = hashlib.sha256((output / "contact_scene.xml").read_bytes()).hexdigest()
        summary["model_source"] = json.loads((ROOT / "assets/MODEL_SOURCE.json").read_text())
        summary["model_dimensions"] = {"njnt": model.njnt, "nq": model.nq, "nv": model.nv, "nu": model.nu}
        arm = [model.joint(n).id for n in config["arm_joints"]]
        qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
        initial_q, command = data.qpos.copy(), data.ctrl.copy()
        state_columns = (["time_s"] + [f"qpos_{i}" for i in range(model.nq)] +
                         [f"qvel_{i}" for i in range(model.nv)] + [f"ctrl_{i}" for i in range(model.nu)])
        def record_and_check():
            state = geometry_state(model, data, config)
            geometry.append(state)
            states.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist(), *data.ctrl.tolist()])
            check_state(model, data, config["joint_limit_tolerance"])
            check_geometry(state, config)
            if np.max(np.abs(data.qpos[qa] - initial_q[qa])) > config["max_arm_drift_rad"]:
                raise ValueError("Arm drift exceeds acceptance threshold")
            if np.max(np.abs(data.qvel[va])) > config["max_arm_speed_rad_s"]:
                raise ValueError("Arm speed exceeds acceptance threshold")
        record_and_check()
        if abs(geometry[0]["normal_gap_m"] - config["initial_gap_m"]) > config["geometry_tolerance_m"]:
            raise ValueError("Initial gap does not match FK placement")
        audit.extend(clearance_audit(model, data, config))
        if any(r["distance_m"] < -config["geometry_tolerance_m"] for r in audit):
            raise ValueError("Initial geometry penetration")
        message(f"FK placement verified; initial gap={geometry[0]['normal_gap_m']:.9f} m; ncon={data.ncon}")
        context = nullcontext(None)
        if viewer:
            import mujoco.viewer as mj_viewer
            context = mj_viewer.launch_passive(model, data)
        with context as view:
            if view is not None:
                with view.lock():
                    configure_camera(view.cam, config)
            start = time.perf_counter()
            for step in range(steps):
                if step % substeps == 0:
                    data.ctrl[:] = command
                    commands.append([float(data.time), *command.tolist()])
                mujoco.mj_step(model, data)
                sim_time = float(data.time)
                if not np.isclose(sim_time, (step + 1) * dt, atol=1e-10, rtol=0):
                    raise ValueError("Unexpected simulation time reset")
                # mj_step leaves some derived geometry at the pre-integration state.
                # Refresh before logging sites, distances and contacts at data.time.
                mujoco.mj_forward(model, data)
                record_and_check()
                if (step + 1) % substeps == 0:
                    message(f"gap={geometry[-1]['normal_gap_m']:.9f} m; ncon={data.ncon}")
                    if view is not None:
                        if not view.is_running():
                            raise RuntimeError("Viewer closed before experiment finished")
                        view.sync()
                        time.sleep(max(0, start + sim_time - time.perf_counter()))
        audit.extend(clearance_audit(model, data, config))
        if any(r["distance_m"] < -config["geometry_tolerance_m"] for r in audit):
            raise ValueError("Geometry penetration after holding")
        final_speed = float(np.max(np.abs(data.qvel[va])))
        if final_speed > config["max_final_arm_speed_rad_s"]:
            raise ValueError("Final speed exceeds settling threshold")
        summary.update({"simulated_duration_s": sim_time, "physics_steps": steps,
                        "state_samples": len(states), "command_samples": len(commands),
                        "initial_gap_m": geometry[0]["normal_gap_m"],
                        "minimum_gap_m": min(r["normal_gap_m"] for r in geometry),
                        "final_gap_m": geometry[-1]["normal_gap_m"],
                        "max_ncon": max(r["ncon"] for r in geometry),
                        "max_arm_drift_rad": float(np.max(np.abs(np.asarray(states)[:, 1 + qa] - initial_q[qa]))),
                        "final_arm_speed_rad_s": final_speed,
                        "warning_counts": data.warning.number.tolist(),
                        "max_gap_distance_disagreement_m": max(abs(r["normal_gap_m"] - r["geom_distance_m"]) for r in geometry)})
        if snapshot:
            from PIL import Image
            camera = mujoco.MjvCamera()
            configure_camera(camera, config)
            with mujoco.Renderer(model, height=config["render_height"], width=config["render_width"]) as renderer:
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(output / "scene.png")
            summary["snapshot_saved"] = True
        summary["passed"] = True
        message("PASS: fixed tool and surface, positive clearance throughout, zero contacts")
    except Exception as exc:
        summary.update(error=f"{type(exc).__name__}: {exc}", failed_at_sim_time_s=sim_time)
        message(summary["error"])
        raise
    finally:
        if states:
            pd.DataFrame(states, columns=state_columns).to_csv(output / "joint_state.csv", index=False)
        if geometry:
            pd.DataFrame(geometry).to_csv(output / "geometry.csv", index=False)
        if commands:
            pd.DataFrame(commands, columns=["time_s"] + [f"ctrl_{i}" for i in range(len(commands[0])-1)]).to_csv(output / "control.csv", index=False)
        if audit:
            pd.DataFrame(audit).to_csv(output / "clearance_audit.csv", index=False)
        save_json(output / "summary.json", summary)
        log.close()
        print(f"Results: {output}; passed={summary['passed']}")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_02.json"))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--snapshot", action="store_true", help="Requires OpenGL; default headless run does not")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, snapshot=args.snapshot)
