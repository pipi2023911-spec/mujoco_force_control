"""C1原模型8秒基线的测量扰动实验；保留全部成功/失败，不改变原验收阈值。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.c1_wipe import load_config, motion_metrics, run as run_wipe
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m1_sensor_study import cases


def analyze(frame, config):
    """只对完整运动段计算指标；提前终止不生成虚假的往返和全段RMSE。"""
    start = config["wipe_start_s"]
    end = start + 2*config["wipe_leg_duration_s"]
    dt = config["physics_dt_s"]
    time = frame.time_s.round(9)
    motion = frame[(time >= start) & (time < end)]
    complete = (float(frame.time_s.iloc[-1]) >= end-dt/2
                and len(motion) == round((end-start)/dt))
    result = {"完整时长": bool(frame.time_s.iloc[-1] >= config["duration_s"]-dt/2),
              "完整运动段": bool(complete), "已记录运动样本数": len(motion)}
    if not complete:
        return result
    # 接触丢失按物理接触点数量统计，不以测量信号或滤波值替代。
    lost = (motion.tool_contact_count.to_numpy() == 0)
    edges = np.diff(np.r_[False, lost, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return {**result, **motion_metrics(frame, config),
            "接触丢失累计_s": float(lost.sum()*dt), "接触丢失次数": len(starts),
            "最长连续丢失_s": float(np.max(stops-starts)*dt) if len(starts) else 0.,
            "首次丢失时刻_s": float(motion.time_s.iloc[starts[0]]) if len(starts) else None,
            "运动正向峰值偏差_N": max(0., float(motion.normal_force_n.max())-config["force_target_n"])}


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/CS")
    config = {**load_config(), "pi_algorithm": "fixed"}
    selected = cases()  # 复用M1同一组九工况，避免两套传感器参数。
    save_json(output / "study_config.json", {"base": config, "cases": selected,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows, traces = [], {}
    for index, (key, label, sensor) in enumerate(selected):
        folder = output / f"c{index}"
        trial = {**config, **sensor}
        error = None
        print(f"C1测量扰动 {index+1}/{len(selected)}：{label}", flush=True)
        try:
            run_wipe(trial, folder)
        except ValueError as exc:
            summary_path = folder / "run_001/summary.json"
            if not summary_path.exists() or json.loads(summary_path.read_text(encoding="utf-8"))["passed"]:
                raise  # 未留下实际失败摘要的配置/绘图等异常不伪装成已完成实验。
            error = str(exc)
        run_folder = folder / "run_001"
        summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(run_folder / "contact_force.csv")
        row = {"工况": key, "说明": label, **sensor, "任务验收通过": summary["passed"],
               "失败原因": error or summary.get("error", ""),
               "数据目录": run_folder.relative_to(output).as_posix(), **analyze(frame, trial)}
        rows.append(row)
        traces[key] = frame
        save_csv(pd.DataFrame(rows), output / "comparison.csv")
    table = pd.DataFrame(rows)
    combo = table[table["工况"].str.startswith("combo")]
    finished = combo[combo["完整运动段"]]
    save_json(output / "study_summary.json", {
        "完成记录工况数": len(rows), "任务验收通过数": int(table["任务验收通过"].sum()),
        "完整运动段数": int(table["完整运动段"].sum()),
        "组合统计": {"运行次数": len(combo), "完整运动段次数": len(finished),
                     "任务通过次数": int(combo["任务验收通过"].sum()),
                     "运动RMSE均值_N": float(finished["运动法向力RMSE_N"].mean()) if len(finished) else None,
                     "运动RMSE样本标准差_N": float(finished["运动法向力RMSE_N"].std(ddof=1)) if len(finished)>1 else None},
        "结论边界": "原接触模型、5N、8秒单程、固定PI、仿真假设测量参数。任务验收只沿用连续接触/实际往返及基础限制；力RMSE另列，未新增或宣称满足力误差门槛。批量退出正常仅表示完成记录。"})
    plot_results(table, traces, output, config)
    print(f"C1九工况已记录：{output}；任务通过{table['任务验收通过'].sum()}/{len(rows)}", flush=True)
    return output


def plot_results(table, traces, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for row in table.to_dict("records"):
        key = row["工况"]
        frame = traces[key]
        axes[0].plot(frame.time_s, config["force_target_n"]-frame.normal_force_n,
                     label=key + ("" if row["任务验收通过"] else " FAIL"))
    axes[0].set(xlim=(config["wipe_start_s"]-1, config["duration_s"]),
                xlabel="时间（s）" if zh else "Time (s)",
                ylabel="目标减真实力（N）" if zh else "True force error (N)")
    axes[0].legend(ncol=3)
    for ax, key, scale, label in [
        (axes[1], "运动法向力RMSE_N", 1., "完整运动段RMSE（N）" if zh else "Complete-motion RMSE (N)"),
        (axes[2], "接触丢失累计_s", 1000., "完整运动段接触丢失（ms）" if zh else "Complete-motion lost contact (ms)")]:
        values = [row.get(key, np.nan)*scale for row in table.to_dict("records")]
        ax.bar(table["工况"], values, color=["#1676a3" if p else "#ba3a32" for p in table["任务验收通过"]])
        for i, complete in enumerate(table["完整运动段"]):
            if not complete:
                ax.text(i, 0, "未完整" if zh else "Incomplete", ha="center", va="bottom")
        ax.set_ylabel(label)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "c1_sensor_study.png", dpi=config["plot_dpi"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="新建run_XXX的结果父目录")
    args = parser.parse_args()
    run(args.output)
