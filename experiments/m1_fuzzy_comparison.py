"""固定 PI / 在线模糊 PI 同条件对照；控制验收不放宽，真实力指标与增益独立记录。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m1_force_pi import load_config, run as run_pi
from experiments.m1_sensor_study import analyze


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/F")
    base = load_config()
    scenarios = [("zero", {}), *[(f"combo{s}", {"sensor_noise_std_n": .05,
                  "sensor_delay_steps": 2, "sensor_lpf_cutoff_hz": 20., "sensor_seed": s})
                  for s in (42, 43, 44)],
                 ("contact_soft", {"contact_solref": [.03, 1.]}),
                 ("contact_hard", {"contact_solref": [.01, 1.]})]
    save_json(output / "study_config.json", {"base": base, "scenarios": scenarios,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows, traces, gains = [], {}, {}
    for index, (key, overrides) in enumerate(scenarios):
        for algorithm, short in [("fixed", "p"), ("fuzzy", "f")]:
            config = {**base, **overrides, "pi_algorithm": algorithm}
            root = output / f"c{index}" / short
            print(f"同条件对照：{key} / {algorithm}", flush=True)
            error = ""
            try:
                run_pi(config, root, mode="pi")
            except ValueError as exc:
                error = str(exc)
            folder = root / "run_001"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            row = {"工况": key, "算法": algorithm, "验收通过": summary["passed"],
                   "失败原因": error, "数据目录": folder.relative_to(output).as_posix()}
            raw = folder / "pi/run_001/contact_force.csv"
            command = folder / "pi/run_001/control.csv"
            if raw.exists():
                frame = pd.read_csv(raw)
                traces[key, algorithm] = frame
                row.update(analyze(frame, config))
            if command.exists():
                commands = pd.read_csv(command)
                active = commands[commands.pi_active]
                if not active.empty:
                    gains[key, algorithm] = active
                    np.testing.assert_allclose(active.pi_p_m, active.pi_kp_m_per_n * active.force_error_n, atol=1e-15)
                    for field in ["pi_kp_m_per_n", "pi_ki_m_per_ns"]:
                        row[field + "_min"] = float(active[field].min())
                        row[field + "_max"] = float(active[field].max())
                    row["输出限幅占比"] = float(active.pi_output_limited.mean())
                    row["在线增益实际变化"] = bool(algorithm == "fuzzy" and
                        np.ptp(active.pi_kp_m_per_n) > 1e-12 and np.ptp(active.pi_ki_m_per_ns) > 1e-12)
            rows.append(row)
            save_csv(pd.DataFrame(rows), output / "comparison.csv")
        # 接管前必须相同；仅控制算法在接管后开始分叉。
        if all((key, a) in traces for a in ["fixed", "fuzzy"]):
            a, b = [traces[key, name] for name in ["fixed", "fuzzy"]]
            before = a.time_s < base["force_control_start_s"]
            np.testing.assert_array_equal(a.loc[before, "normal_force_n"], b.loc[before, "normal_force_n"])
    table = pd.DataFrame(rows)
    fuzzy = table[table["算法"] == "fuzzy"]
    save_json(output / "study_summary.json", {"运行次数": len(rows), "通过次数": int(table["验收通过"].sum()),
              "模糊组增益均在线变化": bool(fuzzy["在线增益实际变化"].all()),
              "结论边界": "规则型在线增益调度；接触参数变化未标定为物理刚度，不是稳定性证明"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for (key, algorithm), frame in gains.items():
        if algorithm == "fuzzy":
            axes[0].plot(frame.time_s, frame.pi_kp_m_per_n, label=key)
            axes[1].plot(frame.time_s, frame.pi_ki_m_per_ns, label=key)
    for ax, value, label in zip(axes, [base["pi_kp_m_per_n"], base["pi_ki_m_per_ns"]], ["Kp (m/N)", "Ki (m/(N s))"]):
        ax.axhline(value, color="black", linestyle="--", label="fixed")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(ncol=3)
    axes[-1].set_xlabel("仿真时间（秒）" if zh else "Time (s)")
    fig.tight_layout()
    fig.savefig(output / "online_gains.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, key in zip(axes.flat, ["zero", "combo42", "contact_soft", "contact_hard"]):
        for algorithm in ["fixed", "fuzzy"]:
            if (key, algorithm) in traces:
                frame = traces[key, algorithm]
                ax.plot(frame.time_s, base["force_target_n"] - frame.normal_force_n, label=algorithm)
        ax.set(title=key, xlim=(8, 12), ylabel="真实力误差（N）" if zh else "True error (N)", xlabel="Time (s)")
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "force_errors.png", dpi=180)
    plt.close(fig)
    print(f"对照与在线增益已记录：{output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="结果父目录")
    args = parser.parse_args()
    run(args.output)
