"""M0-02：根据正运动学建立接触场景，验证无接触保持。"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import mujoco
import numpy as np
import pandas as pd

from common.contact_scene import build_scene, geometry_state, check_geometry, clearance_audit
from common.experiment_view import ExperimentView
from common.chinese_output import chinese, print_summary, save_csv, configure_chinese_font
from common.robot_state import check_state, inspect_model
from experiments.m0_01_model_inspection import ROOT, new_run, save_json


def load_config(path=Path("configs/m0_02.json")):
    scene = json.loads((ROOT / path).read_text(encoding="utf-8"))
    base = json.loads((ROOT / scene["base_config"]).read_text(encoding="utf-8"))
    return {**base, **scene}  # 合并并保存完整参数，避免后续配置修改使旧实验无法追溯。


def configure_camera(camera, config):
    camera.lookat[:] = config["camera_lookat_m"]  # 相机观察中心使用世界坐标。
    camera.distance = config["camera_distance_m"]
    camera.azimuth = config["camera_azimuth_deg"]
    camera.elevation = config["camera_elevation_deg"]


def run(config, output_root=None, viewer=False, snapshot=False, speed=1.0):
    output = new_run(output_root or ROOT / "results" / "M0-02")
    save_json(output / "config.json", config)
    summary = {"passed": False, "display_speed": speed, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "pandas")},
               "viewer_requested": viewer, "snapshot_requested": snapshot}
    geometry, states, commands, audit = [], [], [], []  # 分别保存几何、关节状态、控制输入和距离审计。
    sim_time = 0.0
    log = (output / "run.log").open("x", encoding="utf-8")
    def message(text):
        log.write(f"仿真时间_秒={sim_time:.6f} {text}\n")
        log.flush()
    try:
        dt, period, duration = config["physics_dt_s"], config["control_dt_s"], config["duration_s"]
        if not all(np.isfinite([dt, period, duration])) or dt <= 0 or period < dt or duration <= 0:
            raise ValueError("时间参数无效")
        steps, substeps = round(duration / dt), round(period / dt)
        if not np.isclose(steps * dt, duration) or not np.isclose(substeps * dt, period):
            raise ValueError("周期必须为物理步长的整数倍")
        model, data, placement = build_scene(config, output / "contact_scene.xml")  # 由 FK 确定平面，不修改上游模型。
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
        qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]  # 按模型地址选七个臂关节，排除夹爪。
        initial_q, command = data.qpos.copy(), data.ctrl.copy()  # 保存独立副本，供漂移检查和固定指令使用。
        state_columns = (["time_s"] + [f"qpos_{i}" for i in range(model.nq)] +
                         [f"qvel_{i}" for i in range(model.nv)] + [f"ctrl_{i}" for i in range(model.nu)])
        def record_and_check():
            state = geometry_state(model, data, config)
            geometry.append(state)
            states.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist(), *data.ctrl.tolist()])
            check_state(model, data, config["joint_limit_tolerance"])
            check_geometry(state, config)
            if np.max(np.abs(data.qpos[qa] - initial_q[qa])) > config["max_arm_drift_rad"]:
                raise ValueError("机械臂位置偏差超过验收阈值")
            if np.max(np.abs(data.qvel[va])) > config["max_arm_speed_rad_s"]:
                raise ValueError("机械臂速度超过验收阈值")
        record_and_check()
        if abs(geometry[0]["normal_gap_m"] - config["initial_gap_m"]) > config["geometry_tolerance_m"]:
            raise ValueError("初始间隙与正运动学计算不一致")
        audit.extend(clearance_audit(model, data, config))
        if any(r["distance_m"] < -config["geometry_tolerance_m"] for r in audit):
            raise ValueError("初始几何体穿透")
        message(f"正运动学定位已验证；初始间隙={geometry[0]['normal_gap_m']:.9f} m；接触数={data.ncon}")
        with ExperimentView(model, viewer, speed) as view:
            if view.enabled:
                with view.handle.lock():
                    configure_camera(view.handle.cam, config)
            view.wait_start(data)  # 准备期间不推进仿真时间。
            for step in range(steps):
                if step % substeps == 0:
                    data.ctrl[:] = command  # 整段写入原控制数组，不直接改写关节位置。
                    commands.append([float(data.time), *command.tolist()])
                mujoco.mj_step(model, data)  # 推进一个物理步长，运动由动力学求解得到。
                sim_time = float(data.time)
                if not np.isclose(sim_time, (step + 1) * dt, atol=1e-10, rtol=0):
                    raise ValueError("仿真时间意外重置")
                # mj_step 后部分几何量仍对应积分前状态。
                # 刷新后再记录，确保坐标、间隙和接触数对应当前仿真时间。
                mujoco.mj_forward(model, data)
                record_and_check()
                if (step + 1) % substeps == 0:
                    message(f"间隙={geometry[-1]['normal_gap_m']:.9f} m；接触数={data.ncon}")
                    view.update(data)  # 实时或慢速展示；物理步长保持不变。
            view.wait_end(data)  # 物理积分已停止，关闭窗口后继续生成报告。
        audit.extend(clearance_audit(model, data, config))
        if any(r["distance_m"] < -config["geometry_tolerance_m"] for r in audit):
            raise ValueError("保持结束后出现几何体穿透")
        final_speed = float(np.max(np.abs(data.qvel[va])))
        if final_speed > config["max_final_arm_speed_rad_s"]:
            raise ValueError("末态速度超过稳定验收阈值")
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
        message("验收通过：工具与接触面固定，全程间隙为正且接触数为零")
    except Exception as exc:
        summary.update(error=f"{type(exc).__name__}: {exc}", failed_at_sim_time_s=sim_time)
        message(summary["error"])
        raise
    finally:
        if states:
            save_csv(pd.DataFrame(states, columns=state_columns), output / "joint_state.csv")
        if geometry:
            save_csv(pd.DataFrame(geometry), output / "geometry.csv")
        if commands:
            save_csv(pd.DataFrame(commands, columns=["time_s"] + [f"ctrl_{i}" for i in range(len(commands[0])-1)]), output / "control.csv")
        if audit:
            save_csv(pd.DataFrame(audit), output / "clearance_audit.csv")
        save_json(output / "summary.json", summary)
        log.close()
        print(f"结果目录：{output}；验收：{'通过' if summary['passed'] else '未通过'}")
    print_summary(summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_02.json"))
    parser.add_argument("--viewer", action="store_true", help="展示完整过程：空格开始，结束后关闭窗口")
    parser.add_argument("--speed", type=float, default=1.0, help="图形播放速度；0.25 为四倍慢放，不改变物理步长")
    parser.add_argument("--snapshot", action="store_true", help="保存实际场景截图，需要 OpenGL；默认无图形模式无需 OpenGL")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, snapshot=args.snapshot, speed=args.speed)
