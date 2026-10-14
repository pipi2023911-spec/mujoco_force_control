"""M0-01：保持官方 home 指令，验证模型接口并记录真实仿真。"""
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

from common.experiment_view import ExperimentView
from common.chinese_output import chinese, print_summary, save_csv, configure_chinese_font
from common.robot_state import check_state, inspect_model

ROOT = Path(__file__).resolve().parents[1]  # 从当前脚本确定工程根目录，不依赖启动终端的位置。


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_name(path.stem + "_中文.json").write_text(
        json.dumps(chinese(value), indent=2, ensure_ascii=False), encoding="utf-8")


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


def run(config, output_root=None, viewer=False, speed=1.0):
    output = new_run(output_root or ROOT / "results" / "M0-01")
    save_json(output / "config.json", config)
    rows, commands = [], []  # 分别保存 1000 Hz 状态与 200 Hz 命令记录。
    summary = {"passed": False, "display_speed": speed, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "versions": {n: importlib.metadata.version(n) for n in
                            ("mujoco", "numpy", "pandas", "matplotlib", "pytest")}}
    log = (output / "run.log").open("w", encoding="utf-8")
    sim_time = 0.0
    def message(text):
        log.write(f"仿真时间_秒={sim_time:.6f} {text}\n")
        log.flush()
    try:
        dt, control_dt = config["physics_dt_s"], config["control_dt_s"]
        duration = config["duration_s"]
        if dt <= 0 or control_dt < dt or duration <= 0:
            raise ValueError("时间参数无效")
        substeps, steps = round(control_dt / dt), round(duration / dt)
        if not np.isclose(substeps * dt, control_dt) or not np.isclose(steps * dt, duration):
            raise ValueError("周期必须为物理步长的整数倍")
        model_path = ROOT / config["model"]
        model = mujoco.MjModel.from_xml_path(str(model_path))  # 编译模型结构和资源，不会自动开始仿真。
        model.opt.timestep = dt
        data = mujoco.MjData(model)  # 为该模型创建一份可变化的运行状态。
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config["home_key"])
        if key < 0:
            raise ValueError("缺少初始关键帧 home")
        mujoco.mj_resetDataKeyframe(model, data, key)  # 同时恢复 home 的关节位置和控制输入。
        mujoco.mj_forward(model, data)
        report = inspect_model(model, data, config)
        save_json(output / "model_report.json", report)
        print(json.dumps(chinese(report), ensure_ascii=False, indent=2))
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
        qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]  # 按模型地址选七个臂关节，排除夹爪。
        q0, command = data.qpos.copy(), data.ctrl.copy()  # 复制初始值，避免引用随仿真变化的内存。
        columns = (["time_s"] + [f"qpos_{i}" for i in range(model.nq)] +
                   [f"qvel_{i}" for i in range(model.nv)] +
                   [f"ctrl_{i}" for i in range(model.nu)] + ["ncon"])
        def record():
            rows.append([float(data.time), *data.qpos.tolist(), *data.qvel.tolist(),
                         *data.ctrl.tolist(), int(data.ncon)])
        check_state(model, data, config["joint_limit_tolerance"])
        record()
        message("已加载 home；保持固定位置指令；未启用力控制")
        with ExperimentView(model, viewer, speed) as view:
            view.wait_start(data)  # 先观察初始姿态，按空格才开始物理积分。
            for step in range(steps):
                if step % substeps == 0:
                    data.ctrl[:] = command  # 整段写入原控制数组，不直接改写关节位置。
                    commands.append([float(data.time), *command.tolist()])
                mujoco.mj_step(model, data)  # 推进一个物理步长，运动由动力学求解得到。
                sim_time = float(data.time)
                record()  # 即使此步失败也保留真实样本，便于定位原因。
                check_state(model, data, config["joint_limit_tolerance"])
                if not np.isclose(data.time, (step + 1) * dt, atol=1e-10, rtol=0):
                    raise ValueError("仿真时间重置或积分步长异常")
                if np.max(np.abs(data.qpos[qa] - q0[qa])) > config["max_arm_drift_rad"]:
                    raise ValueError("机械臂位置偏差超过验收阈值")
                if np.max(np.abs(data.qvel[va])) > config["max_arm_speed_rad_s"]:
                    raise ValueError("机械臂速度超过验收阈值")
                if (step + 1) % substeps == 0:
                    message("状态检查通过")
                    view.update(data)  # 实时或慢速展示；物理步长保持不变。
            view.wait_end(data)  # 物理积分已停止，关闭窗口后继续生成报告。
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
            raise ValueError("机械臂末态速度超过稳定验收阈值")
        # Agg 后端直接生成图片，无需桌面或 OpenGL。
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        use_chinese = configure_chinese_font()  # 自动使用系统中文字体，避免方框乱码。
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i, name in enumerate(config["arm_joints"]):
            axes[0].plot(frame.time_s, arm_q[:, i] - q0[qa[i]], label=f"关节{i+1}" if use_chinese else name)
            axes[1].plot(frame.time_s, arm_v[:, i])
        axes[0].set_ylabel("关节位置偏差（rad）" if use_chinese else "Position drift (rad)")
        axes[1].set_ylabel("关节速度（rad/s）" if use_chinese else "Velocity (rad/s)")
        axes[1].set_xlabel("仿真时间（秒）" if use_chinese else "Simulation time (s)")
        axes[0].legend(ncol=4)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "joint_state.png", dpi=150)
        plt.close(fig)
        summary["passed"] = True
        message("验收通过：数值、限位、位置偏差、速度及稳定状态检查通过")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["failed_at_sim_time_s"] = sim_time
        message(summary["error"])
        raise
    finally:
        if rows:
            save_csv(pd.DataFrame(rows, columns=columns), output / "joint_state.csv")
        if commands:
            save_csv(pd.DataFrame(commands, columns=["time_s"] + [f"ctrl_{i}" for i in range(len(commands[0])-1)]), output / "control.csv")
        save_json(output / "summary.json", summary)
        log.close()
        print(f"结果目录：{output}；验收：{'通过' if summary['passed'] else '未通过'}")
    print_summary(summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_01.json"))
    parser.add_argument("--viewer", action="store_true", help="展示完整过程：空格开始，结束后关闭窗口")
    parser.add_argument("--speed", type=float, default=1.0, help="图形播放速度；0.25 为四倍慢放，不改变物理步长")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text(encoding="utf-8")), viewer=args.viewer, speed=args.speed)
