"""M2：真实接触力驱动的法向导纳与质量、阻尼、刚度单因素对照。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from common.kinematics import approach_reference
from controllers.admittance import Admittance
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_03_contact_force import load_config as load_base, run as run_trial

NAMES = {"baseline": "固定位置", "reference": "基准导纳", "soft": "低刚度", "stiff": "高刚度",
         "low_damping": "低阻尼", "high_damping": "高阻尼", "light": "小质量", "heavy": "大质量"}


def load_config(path=Path("configs/m2.json")):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_base(Path(config["base_config"])), **config}


class AdmittanceReference:
    def __init__(self, config, case):
        self.controller = Admittance(config)
        self.case = case
        self.active = False
        self.contact_duration = 0.0

    def __call__(self, time_s, initial_position, normal, config, measured):
        dt = config["control_dt_s"]
        self.contact_duration = self.contact_duration + dt if measured["normal_force_n"] > config["contact_detection_force_n"] else 0.0
        target, phase = approach_reference(time_s, initial_position, normal, config)
        details = {"external_force_n": measured["normal_force_n"], "feedback_time_s": measured["time_s"],
                   "admittance_x_m": 0.0, "admittance_v_m_s": 0.0, "admittance_a_m_s2": 0.0,
                   "admittance_limited": False, "admittance_active": False}
        if time_s + 1e-10 >= config["admittance_start_s"]:
            if not self.active:
                if self.contact_duration < config["contact_confirm_s"]:
                    raise ValueError("导纳接管前未建立持续接触")
                self.active = True
            if self.case != "baseline":
                offset, state = self.controller.step(measured["normal_force_n"], dt)
                target = target + normal * offset  # 平面对球的力向+Z；顺着外力柔顺退让。
                details.update(state, admittance_active=True)
                phase = "导纳柔顺响应"
            else:
                phase = "固定位置对照"
        return target, phase, details


def analyze(frame, commands, config, case):
    active = frame[frame.time_s >= config["admittance_start_s"]]
    steady = active[active.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    command_tail = commands[commands.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    mean_force = float(steady.normal_force_n.mean())
    outside = np.flatnonzero(np.abs(active.normal_force_n.to_numpy() - mean_force) > config["settling_force_band_n"])
    first = int(outside[-1] + 1) if len(outside) else 0
    settling = None
    if first < len(active):
        candidate = float(active.time_s.iloc[first])
        if frame.time_s.iloc[-1] - candidate >= config["settling_min_duration_s"]:
            settling = candidate - config["admittance_start_s"]
    # 与本组最终平衡值比较，而不是拿5N跟踪误差评价柔顺性。
    residual = float(np.max(np.abs(config["admittance_stiffness_n_m"] * command_tail.admittance_x_m - command_tail.external_force_n)))
    normal = np.asarray(config["normal_world"])
    xyz = [f"tool_{a}_m" for a in "xyz"]
    actual_yield = float((steady[xyz].mean().to_numpy() - active[xyz].iloc[0].to_numpy()) @ normal)
    rows = {"工况": NAMES[case], "虚拟质量_kg": config["admittance_mass_kg"],
            "虚拟阻尼_Ns每m": config["admittance_damping_ns_m"], "虚拟刚度_N每m": config["admittance_stiffness_n_m"],
            "末秒平均力_N": mean_force, "接管后最小力_N": float(active.normal_force_n.min()),
            "接管后峰值力_N": float(active.normal_force_n.max()), "末秒力标准差_N": float(steady.normal_force_n.std(ddof=0)),
            "末秒参考退让_mm": float(command_tail.admittance_x_m.mean() * 1000),
            "末秒实际末端退让_mm": actual_yield * 1000,
            "峰值参考退让_mm": float(commands.admittance_x_m.max() * 1000),
            "峰值参考速度_mm每秒": float(commands.admittance_v_m_s.abs().max() * 1000),
            "达到自身平衡力的稳定时间_秒": settling,
            "虚拟弹簧平衡最大残差_N": residual if case != "baseline" else None,
            "限幅样本数": int(commands.admittance_limited.sum()),
            "接管后接触占比": float((active.tool_contact_count > 0).mean())}
    if case != "baseline":
        if rows["限幅样本数"]:
            raise ValueError(f"导纳触及限幅，不能作为未受限质量阻尼刚度对照验收：{rows}")
        if residual > config["equilibrium_residual_limit_n"] or rows["末秒力标准差_N"] > config["steady_force_std_limit_n"] or settling is None:
            raise ValueError(f"导纳平衡或稳定性验收未通过：{rows}")
        if (active.tool_contact_count == 0).any():
            raise ValueError("导纳接管后接触丢失")
    return rows


def plot_results(output, frames, commands, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    groups = [("刚度" if zh else "Stiffness", ["baseline", "reference", "soft", "stiff"]),
              ("阻尼" if zh else "Damping", ["reference", "low_damping", "high_damping"]),
              ("质量" if zh else "Mass", ["reference", "light", "heavy"])]
    for title, cases in groups:
        present = [case for case in cases if case in frames]
        if not present:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(config["plot_width_in"], 2 * config["plot_height_in"]), sharex=True)
        for case in present:
            label = NAMES[case] if zh else case
            axes[0].plot(frames[case].time_s, frames[case].normal_force_n, label=label)
            axes[1].plot(commands[case].time_s, commands[case].admittance_x_m * 1000, label=label)
        axes[0].set(title=f"M2 {title}", ylabel="实际法向力（N）" if zh else "Measured normal force (N)")
        axes[1].set(xlabel="仿真时间（秒）" if zh else "Time (s)", ylabel="参考退让（mm）" if zh else "Reference yielding (mm)")
        for ax in axes:
            ax.set_xlim(config["admittance_start_s"] - .2, config["duration_s"])
            ax.axvline(config["admittance_start_s"], linestyle=":", color="gray")
            ax.grid(alpha=.25); ax.legend()
        fig.tight_layout()
        name = {"刚度":"stiffness", "阻尼":"damping", "质量":"mass"}.get(title, title.lower())
        fig.savefig(output / f"{name}_comparison.png", dpi=config["plot_dpi"])
        plt.close(fig)


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False, case="all"):
    output = new_run(output_root or ROOT / "results" / "M2")
    save_json(output / "config.json", {**config, "selected_case": case})
    summary = {"passed": False, "已完成工况": []}
    rows, frames, commands = [], {}, {}
    with (output / "run.log").open("x", encoding="utf-8") as log:
        try:
            if config["admittance_start_s"] < config["settle_duration_s"] + config["approach_duration_s"] + config["contact_confirm_s"] or config["duration_s"] - config["admittance_start_s"] < config["hold_metrics_window_s"]:
                raise ValueError("导纳接管前后必须保留接触确认及末段统计时间")
            if config["admittance_speed_limit_m_s"] > config["max_reference_speed_m_s"]:
                raise ValueError("导纳速度上限超过实验参考速度上限")
            selected = list(config["cases"]) if case == "all" else [case]
            for name in selected:
                if name not in NAMES or name not in config["cases"]:
                    raise ValueError("未知导纳工况")
                c = {**config, **config["cases"][name], "experiment_label": "M2 " + NAMES[name]}
                print(f"开始 {NAMES[name]}：M={c['admittance_mass_kg']} kg，D={c['admittance_damping_ns_m']} Ns/m，K={c['admittance_stiffness_n_m']} N/m", flush=True)
                log.write(f"工况={name} 仿真时间_秒=0\n"); log.flush()
                folder = run_trial(c, output / name, viewer, speed, snapshot, feedback_reference=AdmittanceReference(c, name))
                frames[name] = pd.read_csv(folder / "contact_force.csv")
                commands[name] = pd.read_csv(folder / "control.csv")
                row = analyze(frames[name], commands[name], c, name)
                rows.append({**row, "数据目录": folder.relative_to(output).as_posix()})
                save_csv(pd.DataFrame(rows), output / "comparison.csv")
                summary["已完成工况"].append(name)
                log.write(f"工况={name} 仿真时间_秒={c['duration_s']} 平均力_N={row['末秒平均力_N']}\n")
            plot_results(output, frames, commands, config)
            summary.update(passed=True, 对照结果=rows)
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            log.write(summary["error"] + "\n")
            raise
        finally:
            save_json(output / "summary.json", summary)
            print(f"M2结果：{output}；验收：{'通过' if summary['passed'] else '未通过'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m2.json"), help="导纳实验配置")
    parser.add_argument("--case", default="all", choices=["all", *NAMES], help="默认运行全部八组；reference仅基准导纳")
    parser.add_argument("--viewer", action="store_true", help="每组空格开始，结束后关窗进入下一组")
    parser.add_argument("--speed", type=float, default=1.0, help="图形播放倍率")
    parser.add_argument("--snapshot", action="store_true", help="保存末态截图，需要OpenGL")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, speed=args.speed, snapshot=args.snapshot, case=args.case)
