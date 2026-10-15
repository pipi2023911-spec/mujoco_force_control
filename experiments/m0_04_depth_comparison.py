"""M0-04：独立复位的三组压入参考对照；真实力仅用于测量与验收。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_03_contact_force import load_config as load_base, run as run_trial


def load_config(path=Path("configs/m0_04.json")):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_base(Path(config["base_config"])), **config}


def analyze_trial(directory, config):
    """从实际逐点日志重建世界力，并区分参考行程与真实软接触穿透。"""
    frame = pd.read_csv(directory / "contact_force.csv")
    raw = pd.read_csv(directory / "raw_contacts.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    normal = np.asarray(config["normal_world"], dtype=float)
    matrices = raw[[f"frame_{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    local = raw[[f"local_force_{a}_n" for a in ("normal", "tangent1", "tangent2")]].to_numpy()
    # 每条记录：世界力 = 工具方向符号 × 接触矩阵转置 × 局部力。
    world = np.einsum("nji,nj->ni", matrices, local) * raw.tool_force_sign.to_numpy()[:, None]
    reconstructed = pd.DataFrame(world, columns=list("xyz"))
    reconstructed["time_s"] = raw.time_s.to_numpy()
    summed = reconstructed.groupby("time_s")[["x", "y", "z"]].sum()
    summed = summed.reindex(frame.time_s, fill_value=0).to_numpy()  # 无接触时必须是零。
    recorded = frame[[f"tool_force_{a}_n" for a in "xyz"]].to_numpy()
    error = float(max(np.max(np.abs(summed - recorded)),
                      np.max(np.abs(summed @ normal - frame.normal_force_n.to_numpy()))))
    hold = frame[frame.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    forces = hold[[f"tool_force_{a}_n" for a in "xyz"]].to_numpy()
    alignment = (forces @ normal) / np.maximum(np.linalg.norm(forces, axis=1), np.finfo(float).tiny)
    if (not summary["passed"] or not np.isfinite(summed).all() or
            error > config["force_reconstruction_tolerance_n"]):
        raise ValueError("逐点坐标转换与合力日志不一致")
    if (frame.normal_force_n.min() < -config["zero_force_tolerance_n"] or
            np.min(alignment) < config["minimum_normal_alignment"]):
        raise ValueError("法向力符号或末段受力方向异常")
    return {"参考接近行程_mm": config["approach_distance_m"] * 1000,
            "末段实际穿透_mm": float(-hold.normal_gap_m.mean() * 1000),
            "末段实际下降_mm": float((frame.normal_gap_m.iloc[0] - hold.normal_gap_m.mean()) * 1000),
            "末段平均法向力_N": float(hold.normal_force_n.mean()),
            "末段反向法向投影_N": float(np.mean(forces @ -normal)),
            "末段力标准差_N": float(hold.normal_force_n.std(ddof=0)),
            "峰值法向力_N": summary["peak_force_n"],
            "最小法向夹角余弦": float(np.min(alignment)),
            "坐标重建最大误差_N": error,
            "首次接触时间_s": summary["first_contact_time_s"],
            "末段接触占比": summary["hold_contact_fraction"]}


def check_trend(table, config):
    # 同一场景、相同初态，只有参考行程不同；要求变化量超过明确阈值。
    if not np.all(np.diff(table["末段平均法向力_N"]) > config["force_increment_min_n"]):
        raise ValueError("增加参考行程后，末段法向力未按阈值递增")
    if not np.all(np.diff(table["末段实际穿透_mm"]) / 1000 > config["penetration_increment_min_m"]):
        raise ValueError("增加参考行程后，实际穿透未按阈值递增")


def plot_comparison(output, table, traces, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chinese = configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for distance, frame in traces:
        axes[0].plot(frame.time_s, frame.normal_force_n, label=f"{distance * 1000:g} mm")
    axes[0].set(xlabel="仿真时间（秒）" if chinese else "Time (s)",
                ylabel="法向接触力（N）" if chinese else "Normal force (N)",
                title="不同参考行程的真实力" if chinese else "Measured force by reference travel")
    axes[0].legend(title="参考行程" if chinese else "Reference travel")
    axes[1].plot(table["末段实际穿透_mm"], table["末段平均法向力_N"], "o-")
    for _, row in table.iterrows():
        axes[1].annotate(f"{row['参考接近行程_mm']:g} mm", (row["末段实际穿透_mm"], row["末段平均法向力_N"]), xytext=(4, 4), textcoords="offset points")
    axes[1].set(xlabel="末段实际软接触穿透（mm）" if chinese else "Actual soft-contact penetration (mm)",
                ylabel="末段平均法向力（N）" if chinese else "Mean normal force (N)",
                title="实际穿透与力的关系" if chinese else "Penetration versus force")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.margins(0.15)
    fig.tight_layout()
    fig.savefig(output / "depth_comparison.png", dpi=config["plot_dpi"])
    plt.close(fig)


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False):
    output = new_run(output_root or ROOT / "results" / "M0-04")
    save_json(output / "config.json", config)
    report = {"passed": False, "实验": "M0-04：不同压入参考的力方向与变化", "已完成组数": 0}
    rows, traces = [], []
    with (output / "run.log").open("x", encoding="utf-8") as log:
        try:
            distances = np.asarray(config["approach_distances_m"], dtype=float)
            if distances.ndim != 1 or len(distances) < 2 or not np.isfinite(distances).all() or np.any(distances <= 0) or np.any(np.diff(distances) <= 0):
                raise ValueError("对照行程必须是至少两个递增的正数")
            for index, distance in enumerate(distances, start=1):
                trial_config = {**config, "approach_distance_m": float(distance), "experiment_label": "M0-04"}
                print(f"第 {index}/{len(distances)} 组：参考行程 {distance * 1000:g} mm；从同一 home 初态重新开始。", flush=True)
                log.write(f"组={index} 仿真时间_秒=0 参考行程_米={distance}\n")
                log.flush()
                directory = run_trial(trial_config, output / f"trial_{index:02d}", viewer=viewer, speed=speed, snapshot=snapshot)
                row = analyze_trial(directory, trial_config)
                row["数据目录"] = directory.relative_to(output).as_posix()
                rows.append(row)
                report["已完成组数"] = len(rows)
                traces.append((distance, pd.read_csv(directory / "contact_force.csv")))
                save_csv(pd.DataFrame(rows), output / "comparison.csv")
                log.write(f"组={index} 仿真时间_秒={config['duration_s']} 末段法向力_N={row['末段平均法向力_N']}\n")
            table = pd.DataFrame(rows)
            plot_comparison(output, table, traces, config)
            check_trend(table, config)
            report.update(passed=True, 对照结果=rows,
                          结论="参考行程增加，实际穿透和正法向接触力递增；反向法向投影为负。")
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            log.write(report["error"] + "\n")
            raise
        finally:
            save_json(output / "summary.json", report)
            print(f"M0-04 结果目录：{output}；验收：{'通过' if report['passed'] else '未通过'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_04.json"), help="对照实验配置")
    parser.add_argument("--viewer", action="store_true", help="各组按空格开始，完成后关闭窗口进入下一组")
    parser.add_argument("--speed", type=float, default=1.0, help="播放倍率，不改变物理步长")
    parser.add_argument("--snapshot", action="store_true", help="保存每组末态截图，需要 OpenGL")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, speed=args.speed, snapshot=args.snapshot)
