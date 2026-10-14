"""M0-01: hold the upstream home command, record actual MuJoCo integration."""
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

from common.robot_state import check_state, inspect_model

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def new_run(parent):
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = parent / f"run_{index:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def run(config, output_root=None, viewer=False):
    output = new_run(output_root or ROOT / "results" / "M0-01")
    save_json(output / "config.json", config)
    rows, commands = [], []
    summary = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in
                            ("mujoco", "numpy", "pandas", "matplotlib", "pytest")}}
    log = (output / "run.log").open("w", encoding="utf-8")
    sim_time = 0.0
    def message(text):
        log.write(f"sim_time_s={sim_time:.6f} {text}\n")
        log.flush()
    try:
        dt, control_dt = config["physics_dt_s"], config["control_dt_s"]
        duration = config["duration_s"]
        if dt <= 0 or control_dt < dt or duration <= 0:
            raise ValueError("Invalid time configuration")
        substeps, steps = round(control_dt / dt), round(duration / dt)
        if not np.isclose(substeps * dt, control_dt) or not np.isclose(steps * dt, duration):
            raise ValueError("Periods must be integer multiples of physics timestep")
        model_path = ROOT / config["model"]
        model = mujoco.MjModel.from_xml_path(str(model_path))
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_key"])
        if key < 0:
            raise ValueError("Required home keyframe missing")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        report = inspect_model(model, data, config)
        save_json(output / "model_report.json", report)
        print(json.dumps(report, indent=2))
        summary["model_dimensions"] = {k: report[k] for k in ("njnt", "nq", "nv", "nu")}
        summary["asset_sha256"] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_path.parent.rglob("*")) if p.is_file()}
        summary["source_sha256"] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("common", "experiments") for p in sorted((ROOT / folder).glob("*.py"))}
        summary["model_source"] = json.loads((ROOT / "assets" / "MODEL_SOURCE.json").read_text())
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        summary["git_head_at_run"] = git.stdout.strip() if git.returncode == 0 else "uncommitted initial repository"
        arm = [model.joint(n).id for n in config["arm_joints"]]
        qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
        q0, command = data.qpos.copy(), data.ctrl.copy()
        columns = (["time_s"] + [f"qpos_{i}" for i in range(model.nq)] +
                   [f"qvel_{i}" for i in range(model.nv)] +
                   [f"ctrl_{i}" for i in range(model.nu)] + ["ncon"])
        def record():
            rows.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist(),
                         *data.ctrl.tolist(), int(data.ncon)])
        check_state(model, data, config["joint_limit_tolerance"])
        record()
        message("Loaded home; constant position-servo reference; no force control")
        context = nullcontext(None)
        if viewer:
            import mujoco.viewer as mj_viewer
            context = mj_viewer.launch_passive(model, data)
        with context as view:
            wall_start = time.perf_counter()
            for step in range(steps):
                if step % substeps == 0:
                    data.ctrl[:] = command
                    commands.append([float(data.time), *command.tolist()])
                mujoco.mj_step(model, data)
                sim_time = float(data.time)
                record()  # Keep even the failing sample as evidence.
                check_state(model, data, config["joint_limit_tolerance"])
                if not np.isclose(data.time, (step + 1) * dt, atol=1e-10, rtol=0):
                    raise ValueError("Unexpected time reset or integration step")
                if np.max(np.abs(data.qpos[qa] - q0[qa])) > config["max_arm_drift_rad"]:
                    raise ValueError("Arm drift exceeds configured acceptance threshold")
                if np.max(np.abs(data.qvel[va])) > config["max_arm_speed_rad_s"]:
                    raise ValueError("Arm speed exceeds configured acceptance threshold")
                if (step + 1) % substeps == 0:
                    message("State checks passed")
                    if view is not None:
                        if not view.is_running():
                            raise RuntimeError("Viewer closed before experiment finished")
                        view.sync()
                        time.sleep(max(0, wall_start + data.time - time.perf_counter()))
        frame = pd.DataFrame(rows, columns=columns)
        arm_q = frame[[f"qpos_{i}" for i in qa]].to_numpy()
        arm_v = frame[[f"qvel_{i}" for i in va]].to_numpy()
        final_speed = float(np.max(np.abs(arm_v[-1])))
        summary.update({"simulated_duration_s": sim_time, "physics_steps": steps,
                        "state_samples": len(rows), "command_samples": len(commands),
                        "max_arm_drift_rad": float(np.max(np.abs(arm_q - q0[qa]))),
                        "max_arm_speed_rad_s": float(np.max(np.abs(arm_v))),
                        "final_arm_speed_rad_s": final_speed,
                        "max_ncon": int(frame.ncon.max()),
                        "warning_counts": data.warning.number.tolist()})
        if final_speed > config["max_final_arm_speed_rad_s"]:
            raise ValueError("Final arm speed exceeds settling threshold")
        # Agg produces file plots without a display or OpenGL context.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i, name in enumerate(config["arm_joints"]):
            axes[0].plot(frame.time_s, arm_q[:, i] - q0[qa[i]], label=name)
            axes[1].plot(frame.time_s, arm_v[:, i])
        axes[0].set_ylabel("Position drift (rad)")
        axes[1].set_ylabel("Velocity (rad/s)")
        axes[1].set_xlabel("Simulation time (s)")
        axes[0].legend(ncol=4)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "joint_state.png", dpi=150)
        plt.close(fig)
        summary["passed"] = True
        message("PASS: finite state, limits, drift, speed and settling checks")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["failed_at_sim_time_s"] = sim_time
        message(summary["error"])
        raise
    finally:
        if rows:
            pd.DataFrame(rows, columns=columns).to_csv(output / "joint_state.csv", index=False)
        if commands:
            pd.DataFrame(commands, columns=["time_s"] + [f"ctrl_{i}" for i in range(len(commands[0])-1)]).to_csv(output / "control.csv", index=False)
        save_json(output / "summary.json", summary)
        log.close()
        print(f"Results: {output}; passed={summary['passed']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_01.json"))
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text(encoding="utf-8")), viewer=args.viewer)
