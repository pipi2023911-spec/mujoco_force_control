"""原模型、5 N、8秒单程：固定/模糊PI九工况配对，保留失败与真实力指标。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.c1_sensor_study import analyze
from experiments.c1_wipe import load_config, run as run_wipe
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m1_sensor_study import cases


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/CF")
    base = load_config()
    selected = cases()
    save_json(output / "study_config.json", {"base": base, "cases": selected,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows, traces, gains = [], {}, {}
    for index, (key, label, sensor) in enumerate(selected):
        for algorithm, short in [("fixed", "p"), ("fuzzy", "f")]:
            config = {**base, **sensor, "pi_algorithm": algorithm}
            parent = output / f"c{index}" / short
            error = ""
            print(f"C1配对 {index+1}/{len(selected)}：{label} / {algorithm}", flush=True)
            try:
                run_wipe(config, parent)
            except ValueError as exc:
                summary_path = parent / "run_001/summary.json"
                if not summary_path.exists() or json.loads(summary_path.read_text(encoding="utf-8"))["passed"]:
                    raise
                error = str(exc)
            folder = parent / "run_001"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "contact_force.csv")
            commands = pd.read_csv(folder / "control.csv")
            active = commands[commands.pi_active]
            row = {"工况": key, "说明": label, "算法": algorithm,
                   "任务验收通过": summary["passed"], "失败原因": error or summary.get("error", ""),
                   "数据目录": folder.relative_to(output).as_posix(), **analyze(frame, config)}
            if not active.empty:
                # 核对实际参与输出的增益，不以目标增益曲线代替实际生效值。
                np.testing.assert_allclose(active.pi_p_m, active.pi_kp_m_per_n * active.force_error_n, atol=1e-15)
                for field, nominal in [("pi_kp_m_per_n", base["pi_kp_m_per_n"]),
                                       ("pi_ki_m_per_ns", base["pi_ki_m_per_ns"])]:
                    assert active[field].between(.5*nominal, 1.5*nominal).all()
                    row[field + "_min"] = float(active[field].min())
                    row[field + "_max"] = float(active[field].max())
                assert commands.pi_offset_m.abs().max() <= base["pi_offset_limit_m"] + 1e-12
                assert commands.pi_i_m.abs().max() <= base["pi_integral_limit_m"] + 1e-12
                assert commands.pi_offset_m.diff().abs().max() <= base["pi_offset_speed_m_s"]*base["control_dt_s"] + 1e-12
                row["在线增益实际变化"] = bool(np.ptp(active.pi_kp_m_per_n) > 1e-12 and np.ptp(active.pi_ki_m_per_ns) > 1e-12)
                row["运动输出限幅占比"] = float(active.loc[active.time_s.between(
                    base["wipe_start_s"], base["wipe_start_s"]+2*base["wipe_leg_duration_s"], inclusive="left"), "pi_output_limited"].mean())
            rows.append(row)
            traces[key, algorithm], gains[key, algorithm] = frame, active
            save_csv(pd.DataFrame(rows), output / "comparison.csv")
        a, b = [traces[key, name] for name in ["fixed", "fuzzy"]]
        np.testing.assert_array_equal(a.loc[a.time_s < base["force_control_start_s"], "normal_force_n"],
                                      b.loc[b.time_s < base["force_control_start_s"], "normal_force_n"])
    table = pd.DataFrame(rows)
    combo = table[table["工况"].str.startswith("combo") & table["完整运动段"]]
    stats = combo.groupby("算法")["运动法向力RMSE_N"].agg(["mean", "std", "count"])
    save_csv(stats.reset_index(), output / "combo_statistics.csv")
    save_json(output / "study_summary.json", {"运行次数": len(rows),
              "任务通过数": int(table["任务验收通过"].sum()),
              "完整运动段数": int(table["完整运动段"].sum()),
              "模糊组增益均在线变化": bool(table.loc[table["算法"] == "fuzzy", "在线增益实际变化"].all()),
              "结论边界": "原模型、5N、8秒单程；任务通过不代表满足额外力误差阈值；测量参数为仿真假设，三种子不构成普适优势或稳定性证明。"})
    plot_results(table, traces, gains, output, base)
    print(f"C1固定/模糊对照已记录：{output}；任务通过{table['任务验收通过'].sum()}/{len(rows)}", flush=True)
    return output


def plot_results(table, traces, gains, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, key in zip(axes.flat, ["zero", "bias", "combo42", "stress"]):
        for algorithm in ["fixed", "fuzzy"]:
            frame = traces[key, algorithm]
            ax.plot(frame.time_s, config["force_target_n"]-frame.normal_force_n, label=algorithm)
        ax.set(title=key, xlim=(config["wipe_start_s"], config["wipe_start_s"]+2*config["wipe_leg_duration_s"]),
               xlabel="时间（s）" if zh else "Time (s)", ylabel="目标减真实力（N）" if zh else "True force error (N)")
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "force_errors.png", dpi=config["plot_dpi"])
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for (key, algorithm), frame in gains.items():
        if algorithm == "fuzzy":
            axes[0].plot(frame.time_s, frame.pi_kp_m_per_n, label=key)
            axes[1].plot(frame.time_s, frame.pi_ki_m_per_ns, label=key)
    for ax, field, label in zip(axes, ["pi_kp_m_per_n", "pi_ki_m_per_ns"], ["Kp (m/N)", "Ki (m/(N s))"]):
        ax.axhline(config[field], color="black", linestyle="--", label="fixed")
        ax.set_ylabel(label)
        ax.legend(ncol=5)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel("时间（s）" if zh else "Time (s)")
    fig.tight_layout()
    fig.savefig(output / "online_gains.png", dpi=config["plot_dpi"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="新建run_XXX的结果父目录")
    run(parser.parse_args().output)
