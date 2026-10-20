"""M0-03：慢速位置接近、真实接触力读取与固定位置保持；不进行力闭环。"""
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

from common.chinese_output import configure_chinese_font, print_summary, save_csv
from common.contact_feedback import contact_force
from common.contact_scene import build_scene, geometry_state, reset_home
from common.experiment_view import ExperimentView
from common.kinematics import approach_reference, orientation_error, solve_pose
from common.robot_state import check_state, inspect_model
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_02_contact_scene import configure_camera, load_config as load_scene_config


def load_config(path=Path("configs/m0_03.json")):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_scene_config(Path(config["scene_config"])), **config}


def plot_force(frame, output, config):
    import matplotlib
    matplotlib.use("Agg")  # 默认无图形环境也能导出真实数据曲线。
    import matplotlib.pyplot as plt
    chinese = configure_chinese_font()
    fig, ax = plt.subplots(figsize=(config["plot_width_in"], config["plot_height_in"]))
    ax.plot(frame.time_s, frame.normal_force_n, color="#1676a3", linewidth=1.5)
    if "force_target_n" in config:
        ax.axhline(config["force_target_n"], color="black", linestyle="--", label="目标力" if chinese else "Target")
        ax.legend()
    label = config.get("experiment_label", "M0-03")
    ax.set(xlabel="仿真时间（秒）" if chinese else "Simulation time (s)",
           ylabel="工具所受法向接触力（N）" if chinese else "Normal force on tool (N)",
           title=(f"{label}：真实接触力" if chinese else f"{label}: measured contact force"))
    ax.set_xlim(0, config["duration_s"])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "contact_force.png", dpi=config["plot_dpi"])
    plt.close(fig)


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False,
        reference_function=approach_reference, end_checks=None, feedback_reference=None,
        track_path=False):
    output = new_run(output_root or ROOT / "results" / "M0-03")
    save_json(output / "config.json", config)
    summary = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "pandas", "matplotlib")},
               "viewer_requested": viewer, "display_speed": speed, "snapshot_requested": snapshot,
               "force_definition": "平面对工具球的合力，向世界 +Z 为正；不取绝对值、不滤波"}
    samples, states, commands, contacts = [], [], [], []
    sim_time = 0.0
    log = (output / "run.log").open("x", encoding="utf-8")
    def message(text):
        log.write(f"仿真时间_秒={sim_time:.6f} {text}\n")
        log.flush()
    try:
        dt, period, duration = config["physics_dt_s"], config["control_dt_s"], config["duration_s"]
        if not np.isfinite([dt, period, duration]).all() or dt <= 0 or period < dt:
            raise ValueError("物理步长、控制周期或时长无效")
        steps, substeps = round(duration / dt), round(period / dt)
        if not np.isclose(steps * dt, duration) or not np.isclose(substeps * dt, period):
            raise ValueError("周期必须为物理步长的整数倍")
        hold_start = config["settle_duration_s"] + config["approach_duration_s"]
        if config["settle_duration_s"] <= 0 or config["approach_duration_s"] <= 0 or duration - hold_start < config["hold_metrics_window_s"]:
            raise ValueError("时间配置必须保留初始保持、接近和完整末段统计窗口")
        # 五次曲线的最大归一化速度为 1.875；在运行前验证接近速度约束。
        peak_reference_speed = 1.875 * (config["approach_distance_m"]) / config["approach_duration_s"]
        if peak_reference_speed > config["max_reference_speed_m_s"]:
            raise ValueError("名义接近速度超过配置上限")
        model, data, placement = build_scene(config, output / "contact_scene.xml")
        save_json(output / "scene_report.json", placement)
        save_json(output / "model_report.json", inspect_model(model, data, config))
        summary["model_dimensions"] = {"njnt": model.njnt, "nq": model.nq, "nv": model.nv, "nu": model.nu}
        summary["git_head_at_run"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        summary["git_worktree_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
        summary["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for folder in ("common", "experiments", "controllers") for p in sorted((ROOT / folder).glob("*.py"))}
        summary["asset_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT / config["robot_xml"]).parent.rglob("*")) if p.is_file()}
        summary["model_source"] = json.loads((ROOT / "assets/MODEL_SOURCE.json").read_text(encoding="utf-8"))
        summary["scene_sha256"] = hashlib.sha256((output / "contact_scene.xml").read_bytes()).hexdigest()
        reference = reset_home(model, config["home_key"])  # 独立运动学数据；实际 data.qpos 只由物理引擎推进。
        arm = np.array([model.joint(n).id for n in config["arm_joints"]])
        qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
        actuator_ids = [next(a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                             and model.actuator_trnid[a, 0] == j) for j in arm]
        tool, plate, site = model.geom("tool_sphere").id, model.geom("contact_plane").id, model.site("tool_tip").id
        normal = np.asarray(config["normal_world"], dtype=float)
        start_pos, start_rotation = data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()
        command, previous_arm = data.ctrl.copy(), data.qpos[qa].copy()
        target, phase = reference_function(0, start_pos, normal, config)
        state_columns = ["time_s"] + [f"qpos_{i}" for i in range(model.nq)] + [f"qvel_{i}" for i in range(model.nv)]

        def record_and_check():
            geometry = geometry_state(model, data, config)
            measured = contact_force(model, data, tool, plate, normal)
            lateral = float(np.linalg.norm((np.eye(3) - np.outer(normal, normal)) @ (data.site_xpos[site] - start_pos)))
            angle_error = float(np.linalg.norm(orientation_error(data.site_xmat[site].reshape(3, 3), start_rotation)))
            samples.append({**geometry, "phase": phase, "normal_force_n": measured["normal_force_n"],
                            "tool_contact_count": measured["tool_contact_count"], "other_contact_count": data.ncon - measured["tool_contact_count"],
                            "lateral_drift_m": lateral, "orientation_error_rad": angle_error,
                            **{f"tool_force_{a}_n": float(measured["force_world_n"][i]) for i, a in enumerate("xyz")}})
            if "force_target_n" in config:
                samples[-1].update(force_target_n=config["force_target_n"],
                                   force_error_n=config["force_target_n"] - measured["normal_force_n"])
            # 仅 C1 显式启用：相对于当前时刻的名义路径检查切向误差。
            # target 是当前仍在执行的零阶保持命令，不用未来参考替代实际命令。
            if track_path:
                path, _ = reference_function(float(data.time), start_pos, normal, config)
                error = data.site_xpos[site] - path
                tangential_error = float(np.linalg.norm(error - normal * (normal @ error)))
                samples[-1].update(
                    **{f"path_{a}_m": float(path[i]) for i, a in enumerate("xyz")},
                    **{f"target_{a}_m": float(target[i]) for i, a in enumerate("xyz")},
                    tangential_error_m=tangential_error,
                    normal_position_error_m=float(normal @ (data.site_xpos[site] - target)))
                if not np.isfinite(np.r_[path, target, tangential_error]).all():
                    raise ValueError("路径、执行参考或轨迹误差出现非有限数值")
                if tangential_error > config["max_tangential_error_m"]:
                    raise ValueError("实际切向位置相对名义路径的误差超限")
            contacts.extend(measured["points"])
            states.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist()])
            check_state(model, data, config["joint_limit_tolerance"])
            if not np.isfinite(measured["force_world_n"]).all():
                raise ValueError("真实接触力出现非有限数值")
            if not geometry["inside_plate_footprint"] or (not track_path and lateral > config["max_lateral_drift_m"]) or angle_error > config["max_orientation_error_rad"]:
                raise ValueError("工具板面覆盖、横向位置或姿态偏差超限")
            if geometry["normal_gap_m"] < -config["max_penetration_m"]:
                raise ValueError("工具球实际穿透量超过仿真验收上限")
            if data.ncon != measured["tool_contact_count"]:
                raise ValueError("出现工具球与薄板之外的意外接触")
            if np.linalg.norm(measured["force_world_n"]) > config["max_measured_force_n"]:
                raise ValueError("真实接触力超过实验上限；停止并保留失败结果")
            if measured["normal_force_n"] < -config["zero_force_tolerance_n"]:
                raise ValueError("工具法向受力符号异常")
            if np.max(np.abs(data.qvel[va])) > config["max_arm_speed_rad_s"]:
                raise ValueError("机械臂实际速度超限")
            if sim_time <= config["settle_duration_s"] and measured["tool_contact_count"]:
                raise ValueError("初始保持阶段不应产生接触")

        record_and_check()
        message("初始无接触；启用实验参考更新与真实反馈记录" if feedback_reference else "初始无接触；仅按时间生成位置参考")
        with ExperimentView(model, viewer, speed) as view:
            if view.enabled:
                with view.handle.lock():
                    configure_camera(view.handle.cam, config)
                    view.handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                    view.handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                view.model.vis.map.force = config["viewer_force_length_m_per_n"] * view.model.stat.meanmass
            view.wait_start(data)
            for step in range(steps):
                if step % substeps == 0:
                    target, phase = reference_function(float(data.time), start_pos, normal, config)
                    control_details = {}
                    if feedback_reference is not None:
                        target, phase, control_details = feedback_reference(float(data.time), start_pos, normal, config, samples[-1])
                    arm_command = solve_pose(model, reference, site, arm, target, start_rotation, config)
                    if np.max(np.abs(arm_command - previous_arm)) > config["max_joint_reference_speed_rad_s"] * period:
                        raise ValueError("关节位置参考变化率超限")
                    command[actuator_ids] = arm_command
                    previous_arm = arm_command.copy()
                    data.ctrl[:] = command  # 唯一施加给真实仿真的输入是执行器位置参考。
                    commands.append({"time_s": float(data.time), "phase": phase,
                                     **control_details,
                                     **{f"target_{a}_m": float(target[i]) for i, a in enumerate("xyz")},
                                     **{f"ctrl_{i}": float(v) for i, v in enumerate(command)}})
                mujoco.mj_step(model, data)
                sim_time = float(data.time)
                if not np.isclose(sim_time, (step + 1) * dt, atol=1e-10, rtol=0):
                    raise ValueError("仿真时间重置或积分步长异常")
                mujoco.mj_forward(model, data)  # 重新求解当前积分后状态，力、位姿、时间戳严格对齐。
                record_and_check()
                if (step + 1) % substeps == 0:
                    message(f"阶段={phase}；法向力={samples[-1]['normal_force_n']:.6f} N；间隙={samples[-1]['normal_gap_m']:.9f} m")
                    view.update(data)
                    if viewer and (step + 1) % max(1, round(config["display_interval_s"] / dt)) == 0:
                        print(f"仿真 {sim_time:.2f} 秒｜{phase}｜法向接触力 {samples[-1]['normal_force_n']:.3f} N", flush=True)
            view.wait_end(data)
        frame = pd.DataFrame(samples)
        hold_end = config.get("contact_hold_end_s", duration)  # 退出实验在退出之前统计接触保持。
        hold = frame[(frame.time_s > hold_end - config["hold_metrics_window_s"]) & (frame.time_s <= hold_end)]
        detected = frame[frame.normal_force_n > config["contact_detection_force_n"]]
        hold_fraction = float((hold.tool_contact_count > 0).mean())
        hold_mean = float(hold.normal_force_n.mean())
        summary.update({"simulated_duration_s": sim_time, "physics_steps": steps, "state_samples": len(samples),
                        "command_samples": len(commands), "raw_contact_rows": len(contacts),
                        "initial_gap_m": samples[0]["normal_gap_m"], "final_gap_m": samples[-1]["normal_gap_m"],
                        "peak_force_n": float(frame.normal_force_n.max()), "hold_mean_force_n": hold_mean,
                        "hold_std_force_n": float(hold.normal_force_n.std(ddof=0)), "hold_contact_fraction": hold_fraction,
                        "first_contact_time_s": None if detected.empty else float(detected.time_s.iloc[0]),
                        "max_lateral_drift_m": float(frame.lateral_drift_m.max()),
                        "max_orientation_error_rad": float(frame.orientation_error_rad.max()),
                        "maximum_penetration_m": max(0.0, float(-frame.normal_gap_m.min())),
                        "other_contact_count": int(frame.other_contact_count.max()),
                        "final_arm_speed_rad_s": float(np.max(np.abs(data.qvel[va]))),
                        "warning_counts": data.warning.number.tolist()})
        if detected.empty or hold_mean < config["minimum_hold_mean_force_n"] or hold_fraction < config["minimum_hold_contact_fraction"]:
            raise ValueError("未建立可持续检测的工具球与薄板接触")
        if summary["final_arm_speed_rad_s"] > config["max_final_arm_speed_rad_s"]:
            raise ValueError("末态关节速度未达到稳定阈值")
        if end_checks is not None:
            summary.update(end_checks(frame, config))  # 实验专属验收也在失败日志保护范围内。
        plot_force(frame, output, config)
        if snapshot:
            from PIL import Image
            camera = mujoco.MjvCamera()
            configure_camera(camera, config)
            with mujoco.Renderer(model, height=config["render_height"], width=config["render_width"]) as renderer:
                options = mujoco.MjvOption()
                options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                model.vis.map.force = config["viewer_force_length_m_per_n"] * model.stat.meanmass
                renderer.update_scene(data, camera=camera, scene_option=options)
                Image.fromarray(renderer.render()).save(output / "contact_scene.png")
        summary["passed"] = True
        message("验收通过：真实力接口和本实验检查通过")
    except Exception as exc:
        summary.update(error=f"{type(exc).__name__}: {exc}", failed_at_sim_time_s=sim_time)
        message(summary["error"])
        raise
    finally:
        if samples:
            save_csv(pd.DataFrame(samples), output / "contact_force.csv")
        if states:
            save_csv(pd.DataFrame(states, columns=state_columns), output / "joint_state.csv")
        if commands:
            save_csv(pd.DataFrame(commands), output / "control.csv")
        if contacts:
            save_csv(pd.DataFrame(contacts), output / "raw_contacts.csv")
        save_json(output / "summary.json", summary)
        log.close()
        scope = "基础循环检查（C1 任务验收另行计算）" if track_path else "验收"
        print(f"结果目录：{output}；{scope}：{'通过' if summary['passed'] else '未通过'}")
        print_summary(summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_03.json"), help="实验配置文件")
    parser.add_argument("--viewer", action="store_true", help="空格开始，完整展示接近与接触保持，结束后关闭窗口")
    parser.add_argument("--speed", type=float, default=1.0, help="图形播放速度倍率，不改变仿真步长")
    parser.add_argument("--snapshot", action="store_true", help="保存末态场景截图，需要 OpenGL")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, speed=args.speed, snapshot=args.snapshot)
