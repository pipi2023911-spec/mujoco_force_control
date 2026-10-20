"""C1-01：复用 M1 力 PI，在水平面执行 10 mm 单程与 10 mm 返回。

世界系位置单位 m、接触力单位 N；5 ms 更新参考，1 ms 物理积分。
只向关节位置执行器写入 DLS 结果，实际位置与受力均由原仿真循环测量。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from common.kinematics import approach_reference
from experiments.m0_01_model_inspection import ROOT, save_json
from experiments.m0_03_contact_force import run as run_trial
from experiments.m1_force_pi import ForceReference, load_config as load_m1


def load_config(path=Path("configs/c1_wipe.json")):
    """继承原始 M1 参数，不复制或重新整定 PI 增益。"""
    extra = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_m1(Path(extra["base_config"])), **extra}


def wipe_displacement(time_s, config):
    """输入时间 s；输出世界 X 位移 m 和速度 m/s，往返各一段五次曲线。"""
    elapsed = time_s - config["wipe_start_s"]
    duration, length = config["wipe_leg_duration_s"], config["wipe_length_m"]
    returning = elapsed >= duration
    u = np.clip((elapsed - (duration if returning else 0)) / duration, 0, 1)
    blend = 10*u**3 - 15*u**4 + 6*u**5
    velocity = length / duration * 30*u**2*(1-u)**2
    return (length*(1-blend), -velocity) if returning else (length*blend, velocity)


def wipe_path(time_s, initial_position, normal, config):
    """名义球心路径：原接近参考加世界 X 切向位移，保持原末端姿态。"""
    path, phase = approach_reference(time_s, initial_position, normal, config)
    path = path + np.array([wipe_displacement(time_s, config)[0], 0, 0])
    start, duration = config["wipe_start_s"], config["wipe_leg_duration_s"]
    if time_s >= start + 2*duration:
        phase = "返回后保持"
    elif time_s >= start + duration:
        phase = "切向返回"
    elif time_s >= start:
        phase = "切向去程"
    return path, phase


class WipeReference:
    """M1 原样负责接触确认和一次性接管；本类只叠加切向路径。"""
    def __init__(self, config):
        self.force_reference = ForceReference(config, "pi")

    def __call__(self, time_s, initial_position, normal, config, measured):
        target, phase, details = self.force_reference(
            time_s, initial_position, normal, config, measured)
        path, path_phase = wipe_path(time_s, initial_position, normal, config)
        target = target + np.array([wipe_displacement(time_s, config)[0], 0, 0])
        if time_s >= config["wipe_start_s"]:
            phase = path_phase
        details.update({f"path_{a}_m": float(path[i]) for i, a in enumerate("xyz")})
        return target, phase, details


def motion_metrics(frame, config):
    """仅统计 [开始, 返回结束) 的物理采样；不混入接近和末段保持。"""
    start = config["wipe_start_s"]
    end = start + 2*config["wipe_leg_duration_s"]
    # 时间取整用于消除物理积分累计的浮点边界误差。
    time = frame.time_s.round(9)
    motion = frame[(time >= start) & (time < end)]
    error = config["force_target_n"] - motion.normal_force_n.to_numpy()
    at = lambda t: frame.iloc[int(np.argmin(np.abs(frame.time_s.to_numpy()-t)))]
    first, turn, last = at(start), at(start+config["wipe_leg_duration_s"]), at(end)
    positions = frame.loc[(time >= start) & (time <= end), ["tool_x_m", "tool_y_m"]].to_numpy()
    outward = float(turn.tool_x_m - first.tool_x_m)
    backward = float(turn.tool_x_m - last.tool_x_m)
    metrics = {
        "运动统计区间_秒": [start, end], "运动采样数": len(motion),
        "运动法向力RMSE_N": float(np.sqrt(np.mean(error**2))),
        "运动平均力误差_N": float(np.mean(error)),
        "运动峰值法向力_N": float(motion.normal_force_n.max()),
        "运动接触保持率": float((motion.tool_contact_count > 0).mean()),
        "运动切向误差RMSE_m": float(np.sqrt(np.mean(motion.tangential_error_m**2))),
        "运动最大切向误差_m": float(motion.tangential_error_m.max()),
        "指令累计路径_m": 2*config["wipe_length_m"],
        "实际累计切向路径_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "实际去程X位移_m": outward, "实际返回X位移_m": backward,
        "往返结束相对起点X位移_m": float(last.tool_x_m-first.tool_x_m),
        "去程端点切向误差_m": float(turn.tangential_error_m),
        "返回端点切向误差_m": float(last.tangential_error_m),
    }
    passed = (
        len(motion) == round((end-start)/config["physics_dt_s"])
        and metrics["运动接触保持率"] == 1.0
        and min(outward, backward) >= config["wipe_length_m"]-config["endpoint_tolerance_m"]
        and max(turn.tangential_error_m, last.tangential_error_m) <= config["endpoint_tolerance_m"]
        and metrics["运动最大切向误差_m"] <= config["max_tangential_error_m"])
    return {**metrics, "连续接触与实际往返验收": bool(passed)}


def plot_results(frame, commands, output, config):
    """只绘制日志实测量；名义路径和目标力以明确标签区分。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    start = config["wipe_start_s"]
    end = start + 2*config["wipe_leg_duration_s"]
    motion = frame[(frame.time_s.round(9) >= start) & (frame.time_s.round(9) <= end)]
    origin = motion[["path_x_m", "path_y_m"]].iloc[0].to_numpy()
    ax = axes[0, 0]
    ax.plot((motion.path_x_m-origin[0])*1000, (motion.path_y_m-origin[1])*1000, "--", label="名义路径")
    ax.plot((motion.tool_x_m-origin[0])*1000, (motion.tool_y_m-origin[1])*1000, label="实际球心轨迹")
    ax.set(xlabel="相对名义起点 X（mm）", ylabel="相对名义起点 Y（mm）", title="运动期间世界 XY 轨迹")
    ax = axes[0, 1]
    ax.plot(frame.time_s, frame.normal_force_n, label="实测法向力")
    ax.axhline(config["force_target_n"], linestyle="--", color="black", label="5 N 目标")
    ax.set(xlabel="时间（s）", ylabel="法向力（N）")
    ax = axes[1, 0]
    for field, name in [("pi_p_m", "比例项"), ("pi_i_m", "积分项"), ("pi_offset_m", "实际 PI 输出")]:
        ax.plot(commands.time_s, commands[field]*1000, label=name)
    ax.set(xlabel="时间（s）", ylabel="法向参考修正（mm）")
    ax = axes[1, 1]
    ax.plot(frame.time_s, frame.tangential_error_m*1000, label="实际切向误差")
    ax.set(xlabel="时间（s）", ylabel="切向位置误差（mm）")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvspan(start, end, alpha=.08, color="green")
    fig.tight_layout()
    fig.savefig(output / "c1_curves.png", dpi=config["plot_dpi"])
    plt.close(fig)


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False):
    """运行单个水平面 PI 实验；不启动倾斜、对照或增益扫描。"""
    peak = 1.875*config["wipe_length_m"]/config["wipe_leg_duration_s"]
    if (not np.isfinite(peak) or config["wipe_length_m"] <= 0
            or config["wipe_leg_duration_s"] <= 0
            or np.hypot(peak, config["pi_offset_speed_m_s"]) > config["max_reference_speed_m_s"]):
        raise ValueError("切向行程或合成参考峰值速度不满足限制")
    if (config["wipe_start_s"] < config["force_control_start_s"]
            or config["force_control_start_s"] < config["settle_duration_s"]+config["approach_duration_s"]+config["contact_confirm_s"]
            or config["duration_s"] < config["wipe_start_s"]+2*config["wipe_leg_duration_s"]+1):
        raise ValueError("时间轴未保留接触确认、PI 接管或至少一秒末态保持")
    output = run_trial(config, output_root or ROOT/"results"/"C1_Wipe",
                       viewer, speed, snapshot, reference_function=wipe_path,
                       feedback_reference=WipeReference(config), track_path=True)
    frame = pd.read_csv(output/"contact_force.csv")
    commands = pd.read_csv(output/"control.csv")
    # 同一物理时刻记录的是刚执行完的旧命令；按严格早于采样时刻匹配。
    # 因此不会把下一周期 PI 输出误标成已执行输入。
    details = ["time_s", "pi_p_m", "pi_i_m", "pi_offset_m", "pi_output_limited", "pi_active"]
    combined = pd.merge_asof(frame, commands[details], on="time_s", direction="backward", allow_exact_matches=False)
    combined.loc[0, details[1:]] = [0., 0., 0., False, False]
    save_csv(combined, output/"tracking.csv")
    metrics = motion_metrics(frame, config)
    metrics["PI参与参考修正"] = bool(commands.pi_active.any() and commands.pi_offset_m.abs().max() > 1e-9)
    metrics["passed"] = metrics["连续接触与实际往返验收"] and metrics["PI参与参考修正"]
    save_json(output/"c1_metrics.json", metrics)
    # 基础循环无异常并不代表任务达标；最终摘要必须反映连续接触验收。
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    summary["基础循环检查通过"] = summary["passed"]
    summary.update(metrics)
    if not metrics["passed"]:
        summary["error"] = "C1-01 连续接触或实际往返验收未通过，详见 c1_metrics.json"
    save_json(output/"summary.json", summary)
    plot_results(frame, commands, output, config)
    with (output/"run.log").open("a", encoding="utf-8") as log:
        log.write(f"C1-01 最终任务验收：{'通过' if metrics['passed'] else '未通过'}\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["passed"]:
        raise ValueError(f"C1-01 验收未通过，请检查真实日志：{output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/c1_wipe.json"), help="实验配置")
    parser.add_argument("--viewer", action="store_true", help="打开交互仿真窗口")
    parser.add_argument("--speed", type=float, default=1.0, help="显示播放倍率，不改变控制周期")
    parser.add_argument("--snapshot", action="store_true", help="保存实际仿真末态画面")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, speed=args.speed, snapshot=args.snapshot)
