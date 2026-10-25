"""M2b：K=0 导纳目标力跟踪，与 K>0 的非零平衡误差分开验收。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_03_contact_force import run as run_trial
from experiments.m1_force_pi import tracking_metrics
from experiments.m2_admittance import AdmittanceReference, load_config as load_m2


def load_config(path=Path("configs/m2b.json")):
    extra = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_m2(Path(extra["base_config"])), **extra}


def analyze(frame, commands, config):
    metrics = tracking_metrics(frame, config)
    active = frame[frame.time_s >= config["admittance_start_s"]]
    steady = commands[commands.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    residual = np.abs(steady.external_force_n - config["admittance_force_target_n"]
                      - config["admittance_stiffness_n_m"] * steady.admittance_x_m)
    tracking_passed = (metrics["末秒RMSE_N"] <= config["steady_rmse_limit_n"]
                       and metrics["接管后稳定时间_秒"] is not None)
    valid = (not commands.admittance_limited.any() and (active.tool_contact_count > 0).all()
             and metrics["末秒力标准差_N"] <= config["steady_force_std_limit_n"]
             and float(residual.max()) <= config["equilibrium_residual_limit_n"])
    return {**metrics, "目标力跟踪通过": bool(tracking_passed),
            "平衡与接触检查通过": bool(valid), "平衡最大残差_N": float(residual.max()),
            "末秒Kx_N": float(config["admittance_stiffness_n_m"] * steady.admittance_x_m.mean()),
            "限幅样本数": int(commands.admittance_limited.sum()),
            "passed": bool(valid and (tracking_passed if config["admittance_stiffness_n_m"] == 0 else True))}


def run(config, output_root=None):
    output = new_run(output_root or ROOT / "results/B")
    save_json(output / "config.json", config)
    rows, traces, offsets = [], {}, {}
    for index, (name, change) in enumerate(config["cases"].items()):
        trial = {**config, **change, "force_target_n": config["admittance_force_target_n"],
                 "force_control_start_s": config["admittance_start_s"], "control_mode": "admittance_force"}
        if not 0 < trial["force_target_n"] < trial["max_measured_force_n"]:
            raise ValueError("导纳目标力必须为正且低于真实力上限")
        folder = output / f"c{index}"
        print(f"导纳目标力实验：{name}", flush=True)
        try:
            result = run_trial(trial, folder, feedback_reference=AdmittanceReference(trial, name))
            frame = pd.read_csv(result / "contact_force.csv")
            commands = pd.read_csv(result / "control.csv")
            row = {"工况": name, **change, **analyze(frame, commands, trial)}
            traces[name], offsets[name] = frame, commands
        except ValueError as exc:
            row = {"工况": name, **change, "passed": False, "error": str(exc)}
        rows.append(row)
        save_json(folder / "task_result.json", row)
        save_csv(pd.DataFrame(rows), output / "comparison.csv")
    save_json(output / "summary.json", {"passed": all(row["passed"] for row in rows), "results": rows})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for name, frame in traces.items():
        axes[0].plot(frame.time_s, frame.normal_force_n, label=name)
        axes[1].plot(frame.time_s, config["admittance_force_target_n"] - frame.normal_force_n, label=name)
        axes[2].plot(offsets[name].time_s, offsets[name].admittance_x_m * 1000, label=name)
    axes[0].axhline(config["admittance_force_target_n"], color="black", linestyle="--")
    for ax, label, english in zip(axes, ["真实法向力（N）", "目标减真实力（N）", "退让参考（mm）"],
                                  ["Force (N)", "Error (N)", "Offset (mm)"]):
        ax.set_ylabel(label if zh else english)
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1].set_xlabel("仿真时间（秒）" if zh else "Time (s)")
    fig.tight_layout()
    fig.savefig(output / "admittance_force.png", dpi=config["plot_dpi"])
    plt.close(fig)
    if not all(row["passed"] for row in rows):
        raise ValueError(f"M2b 存在失败工况，原始数据已保存：{output}")
    print(f"M2b 验收完成：{output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m2b.json"))
    args = parser.parse_args()
    run(load_config(args.config))
