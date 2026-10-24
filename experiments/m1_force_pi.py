"""M1：相同初态的固定位置 Baseline 与真实接触力 PI 对照。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from common.kinematics import approach_reference
from controllers.force_pi import ForcePI
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_03_contact_force import load_config as load_base, run as run_trial


def load_config(path=Path("configs/m1.json")):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_base(Path(config["base_config"])), **config}


class ForceReference:
    """将普通状态字典中的测量力送入独立 PI，再输出三维参考。"""
    def __init__(self, config, mode):
        self.pi = ForcePI(config)
        self.mode = mode
        self.active = False
        self.contact_duration = 0.0

    def __call__(self, time_s, initial_position, normal, config, measured):
        dt = config["control_dt_s"]
        self.contact_duration = self.contact_duration + dt if measured["normal_force_n"] > config["contact_detection_force_n"] else 0.0
        target, phase = approach_reference(time_s, initial_position, normal, config)
        error = config["force_target_n"] - measured["normal_force_n"]
        details = {"force_target_n": config["force_target_n"], "measured_force_n": measured["normal_force_n"],
                   "feedback_time_s": measured["time_s"], "force_error_n": error, "pi_p_m": 0.0,
                   "pi_i_m": 0.0, "pi_offset_m": 0.0, "pi_unsaturated_m": 0.0,
                   "pi_integral_blocked": False, "pi_output_limited": False, "pi_active": False}
        if time_s + 1e-10 >= config["force_control_start_s"]:
            if not self.active:
                if self.contact_duration < config["contact_confirm_s"]:
                    raise ValueError("预定接管时刻尚未建立持续接触")
                self.pi.reset(error)
                self.active = True
            if self.mode == "pi":
                offset, pi_details = self.pi.step(config["force_target_n"], measured["normal_force_n"], dt)
                target = target - normal * offset  # 力不足：误差为正，沿 -法向增加压入。
                details.update(pi_details, pi_active=True)
                phase = "力PI闭环"
            else:
                phase = "固定位置Baseline"
        return target, phase, details


def tracking_metrics(frame, config):
    controlled = frame[frame.time_s >= config["force_control_start_s"]]
    steady = frame[frame.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    error = config["force_target_n"] - steady.normal_force_n.to_numpy()
    outside = np.flatnonzero(np.abs(controlled.normal_force_n.to_numpy() - config["force_target_n"]) > config["tracking_tolerance_n"])
    first = int(outside[-1] + 1) if len(outside) else 0
    settling = None
    if first < len(controlled):
        candidate = float(controlled.time_s.iloc[first])
        if float(frame.time_s.iloc[-1]) - candidate >= config["settling_min_duration_s"]:
            settling = candidate - config["force_control_start_s"]
    result = {"末秒平均力_N": float(steady.normal_force_n.mean()), "末秒平均误差_N": float(np.mean(error)),
              "末秒RMSE_N": float(np.sqrt(np.mean(error**2))), "末秒最大绝对误差_N": float(np.max(np.abs(error))),
              "末秒力标准差_N": float(steady.normal_force_n.std(ddof=0)), "接管后稳定时间_秒": settling,
              "接管后峰值力_N": float(controlled.normal_force_n.max()),
              "接管后接触占比": float((controlled.tool_contact_count > 0).mean())}
    if config["control_mode"] == "pi":
        if result["末秒RMSE_N"] > config["steady_rmse_limit_n"] or result["末秒力标准差_N"] > config["steady_force_std_limit_n"] or settling is None:
            raise ValueError(f"PI 力跟踪验收未通过：{result}")
        if (controlled.tool_contact_count == 0).any():
            raise ValueError("PI 接管后发生接触丢失")
    return result


def plot_comparison(output, frames, commands, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(config["plot_width_in"], 2 * config["plot_height_in"]), sharex=True)
    for mode, frame in frames.items():
        name = ("力 PI" if mode == "pi" else "固定位置 Baseline") if zh else mode
        axes[0].plot(frame.time_s, frame.normal_force_n, label=name)
        axes[1].plot(frame.time_s, config["force_target_n"] - frame.normal_force_n, label=name)
    axes[0].axhline(config["force_target_n"], color="black", linestyle="--", label="目标力" if zh else "Target")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[0].set_ylabel("法向接触力（N）" if zh else "Normal force (N)")
    axes[1].set_ylabel("目标减实际力（N）" if zh else "Force error (N)")
    axes[1].set_xlabel("仿真时间（秒）" if zh else "Time (s)")
    for ax in axes:
        ax.axvline(config["force_control_start_s"], color="gray", linestyle=":")
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "force_comparison.png", dpi=config["plot_dpi"])
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(config["plot_width_in"], config["plot_height_in"]))
    for mode, frame in commands.items():
        ax.plot(frame.time_s, frame.pi_offset_m * 1000, label=mode)
    ax.set(xlabel="仿真时间（秒）" if zh else "Time (s)", ylabel="法向压入参考修正（mm）" if zh else "Reference correction (mm)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "position_command.png", dpi=config["plot_dpi"])
    plt.close(fig)
    if "pi" in commands:
        frame = commands["pi"]
        fig, ax = plt.subplots(figsize=(config["plot_width_in"], config["plot_height_in"]))
        for field, label in (("sensor_true_force_n", "真实力"), ("sensor_raw_force_n", "延迟与噪声后"),
                             ("sensor_filtered_force_n", "控制器测量力")):
            ax.plot(frame.time_s, frame[field], label=label if zh else field, alpha=.75)
        ax.set(xlabel="仿真时间（秒）" if zh else "Time (s)", ylabel="法向力（N）" if zh else "Force (N)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "sensor_force.png", dpi=config["plot_dpi"])
        plt.close(fig)


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False, mode="both"):
    output = new_run(output_root or ROOT / "results" / "M1")
    save_json(output / "config.json", {**config, "requested_mode": mode})
    summary = {"passed": False, "已完成实验": []}
    frames, commands, rows = {}, {}, []
    with (output / "run.log").open("x", encoding="utf-8") as log:
        try:
            if mode not in ("baseline", "pi", "both"):
                raise ValueError("未知实验模式")
            if (config["force_control_start_s"] < config["settle_duration_s"] + config["approach_duration_s"] + config["contact_confirm_s"] or
                    config["duration_s"] - config["force_control_start_s"] < config["hold_metrics_window_s"] or
                    not 0 < config["force_target_n"] < config["max_measured_force_n"] or
                    config["pi_offset_speed_m_s"] > config["max_reference_speed_m_s"]):
                raise ValueError("力目标、接管时间或参考速度配置不合理")
            for selected in (["baseline", "pi"] if mode == "both" else [mode]):
                trial = {**config, "control_mode": selected, "experiment_label": "M1 " + selected}
                name = "固定位置 Baseline" if selected == "baseline" else f"{config['force_target_n']:g} N 力 PI"
                print(f"开始 {name}，从相同 home 初态运行。", flush=True)
                log.write(f"模式={selected} 仿真时间_秒=0\n"); log.flush()
                folder = run_trial(trial, output / selected, viewer, speed, snapshot,
                                   feedback_reference=ForceReference(trial, selected), end_checks=tracking_metrics)
                frames[selected] = pd.read_csv(folder / "contact_force.csv")
                commands[selected] = pd.read_csv(folder / "control.csv")
                rows.append({"模式": selected, **tracking_metrics(frames[selected], trial), "数据目录": folder.relative_to(output).as_posix()})
                summary["已完成实验"].append(selected)
                save_csv(pd.DataFrame(rows), output / "comparison.csv")
                log.write(f"模式={selected} 仿真时间_秒={config['duration_s']} 验收通过\n")
            plot_comparison(output, frames, commands, config)
            if mode == "both":
                ratio = rows[1]["末秒RMSE_N"] / max(rows[0]["末秒RMSE_N"], np.finfo(float).eps)
                summary["PI与Baseline末秒RMSE比值"] = ratio
                if ratio > config["comparison_rmse_ratio_limit"]:
                    raise ValueError("PI 相对 Baseline 的误差改善不足")
            summary.update(passed=True, 对照结果=rows)
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            log.write(summary["error"] + "\n")
            raise
        finally:
            save_json(output / "summary.json", summary)
            print(f"M1 结果：{output}；验收：{'通过' if summary['passed'] else '未通过'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m1.json"), help="实验参数")
    parser.add_argument("--mode", choices=("both", "baseline", "pi"), default="both", help="默认先 Baseline 后 PI")
    parser.add_argument("--viewer", action="store_true", help="每组空格开始，结束后关窗进入下一组")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍率")
    parser.add_argument("--snapshot", action="store_true", help="保存末态截图，需要 OpenGL")
    parser.add_argument("--sensor-config", type=Path, help="可选的传感器参数 JSON，叠加到实验配置并保存")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sensor_config is not None:
        config.update(json.loads(args.sensor_config.read_text(encoding="utf-8")))
    run(config, viewer=args.viewer, speed=args.speed, snapshot=args.snapshot, mode=args.mode)
