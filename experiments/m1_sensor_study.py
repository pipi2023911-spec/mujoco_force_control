"""固定 PI 测量扰动：单因素、小规模组合重复与大延迟压力工况。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m1_force_pi import load_config, run as run_pi, tracking_metrics


def cases():
    base = {"sensor_noise_std_n": 0., "sensor_bias_n": 0., "sensor_delay_steps": 0,
            "sensor_lpf_cutoff_hz": None, "sensor_seed": 42}
    overrides = [
        ("zero", "无扰动", {}),
        ("noise", "仅噪声", {"sensor_noise_std_n": .05}),
        ("bias", "仅偏置", {"sensor_bias_n": .1}),
        ("delay", "仅延迟", {"sensor_delay_steps": 4}),
        ("filter", "仅低通", {"sensor_lpf_cutoff_hz": 20.}),
        *[(f"combo{s}", f"组合种子{s}", {"sensor_noise_std_n": .05,
           "sensor_delay_steps": 2, "sensor_lpf_cutoff_hz": 20., "sensor_seed": s})
          for s in (42, 43, 44)],
        ("stress", "大延迟压力工况", {"sensor_delay_steps": 100}),
    ]
    return [(key, label, {**base, **change}) for key, label, change in overrides]


def analyze(frame, config):
    """复用真值指标定义，但不再次触发控制验收；FAIL 仍由原实验判定。"""
    complete = float(frame.time_s.iloc[-1]) >= config["duration_s"] - config["physics_dt_s"] / 2
    if not complete:
        return {"完整时长": False}  # 不给提前终止实验伪造末秒/稳定时间统计。
    values = tracking_metrics(frame, {**config, "control_mode": "baseline"})
    controlled = frame[frame.time_s >= config["force_control_start_s"]]
    error = config["force_target_n"] - controlled.normal_force_n.to_numpy()
    return {"完整时长": True, **values,
            "接管全程RMSE_N": float(np.sqrt(np.mean(error**2))),
            "正向峰值偏差_N": max(0., float(controlled.normal_force_n.max()) - config["force_target_n"]),
            "接管初始力_N": float(controlled.normal_force_n.iloc[0])}


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/S")
    config = load_config()
    selected = cases()
    save_json(output / "study_config.json", {"base": config, "cases": selected,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows, traces = [], {}
    for index, (key, label, sensor) in enumerate(selected):
        folder = output / f"c{index}"
        trial = {**config, **sensor}
        error = None
        print(f"测量扰动实验 {index + 1}/{len(selected)}：{label}", flush=True)
        try:
            run_pi(trial, folder, mode="pi")
        except ValueError as exc:
            error = str(exc)  # 保留原始 FAIL，不以批量继续运行掩盖失败。
        run_folder = folder / "run_001"
        summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
        row = {"工况": key, "说明": label, **sensor, "验收通过": summary["passed"],
               "失败原因": error or summary.get("error", ""), "数据目录": run_folder.relative_to(output).as_posix()}
        raw = run_folder / "pi/run_001/contact_force.csv"
        if raw.exists():
            frame = pd.read_csv(raw)
            row.update(analyze(frame, trial))
            traces[key] = frame
        rows.append(row)
        save_csv(pd.DataFrame(rows), output / "comparison.csv")
    table = pd.DataFrame(rows)
    combo = table[table["工况"].str.startswith("combo")]
    stats = {"重复次数": len(combo), "通过次数": int(combo["验收通过"].sum()),
             "末秒RMSE均值_N": float(combo["末秒RMSE_N"].mean()),
             "末秒RMSE样本标准差_N": float(combo["末秒RMSE_N"].std(ddof=1))}
    save_json(output / "study_summary.json", {"完成工况数": len(rows),
              "通过工况数": int(table["验收通过"].sum()), "组合统计": stats,
              "结论边界": "固定PI、单一接触参数、单一目标；不是模糊PI或多刚度验证"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for key, frame in traces.items():
        axes[0].plot(frame.time_s, config["force_target_n"] - frame.normal_force_n, label=key)
    axes[0].axvline(config["force_control_start_s"], color="gray", linestyle=":")
    axes[0].set(xlim=(7.5, config["duration_s"]), xlabel="仿真时间（秒）" if zh else "Time (s)",
                ylabel="目标减真实力（N）" if zh else "True force error (N)")
    axes[0].legend(ncol=3)
    axes[1].bar(table["工况"], table["末秒RMSE_N"],
                color=["#1676a3" if passed else "#ba3a32" for passed in table["验收通过"]])
    axes[1].set_ylabel("末秒真实力 RMSE（N）" if zh else "Last-second true RMSE (N)")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "sensor_study.png", dpi=180)
    plt.close(fig)
    print(f"全部工况已记录：{output}，通过 {table['验收通过'].sum()}/{len(rows)}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="新建 run_XXX 的结果父目录")
    args = parser.parse_args()
    run(args.output)
