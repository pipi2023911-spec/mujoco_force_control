"""从参考位移—真实力数据估计局部有效刚度；不是材料标定或自动整定。"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_04_depth_comparison import load_config, run as run_depth
from experiments.m1_force_pi import load_config as load_pi


def fit_stiffness(displacement_m, force_n):
    x, y = np.asarray(displacement_m), np.asarray(force_n)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all() or np.ptp(x) <= 0:
        raise ValueError("局部拟合至少需要三个有限且参考位移不同的样本")
    centered = x - x.mean()
    stiffness = float(centered @ (y-y.mean()) / (centered @ centered))
    intercept = float(y.mean() - stiffness*x.mean())
    residual = y - (stiffness*x+intercept)
    return {"有效刚度_N每m": stiffness, "截距_N": intercept,
            "参考位移范围_m": [float(x.min()), float(x.max())],
            "拟合最大残差_N": float(np.max(np.abs(residual))), "点数": len(x)}


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/K")
    rows, curves = [], {}
    pi = load_pi()
    for i, (name, change) in enumerate([("original", {}), ("compliant", {"contact_solimp": [.5, .95, .001]})]):
        config = {**load_config(), **change}
        folder = run_depth(config, output / f"c{i}")
        table = pd.read_csv(folder / "comparison.csv")
        fit = fit_stiffness(table["参考接近行程_mm"].to_numpy()/1000, table["末段平均法向力_N"])
        stiffness = fit["有效刚度_N每m"]
        if stiffness <= 0:
            raise ValueError("拟合斜率非正，不能据此设计 PI")
        tau = (1+stiffness*pi["pi_kp_m_per_n"]) / (stiffness*pi["pi_ki_m_per_ns"])
        rows.append({"工况": name, **fit, "一阶估算时间常数_s": tau,
                     "四倍时间常数_仅近似_s": 4*tau,
                     "期望tau1秒的Ki_m每Ns": (1+stiffness*pi["pi_kp_m_per_n"])/stiffness,
                     "数据目录": folder.relative_to(output).as_posix()})
        curves[name] = table
        save_csv(pd.DataFrame(rows), output / "gain_design.csv")
    save_json(output / "summary.json", {"passed": True, "results": rows,
              "限制": "局部准静态有效斜率包含位置内环柔顺性；3点无重复，残差不是统计置信区间。4tau不是实测稳定时间。"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, table in curves.items():
        ax.plot(table["参考接近行程_mm"], table["末段平均法向力_N"], "o-", label=name)
    ax.set(xlabel="参考接近行程（mm）" if zh else "Reference approach (mm)",
           ylabel="末段真实法向力（N）" if zh else "Steady true force (N)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "effective_stiffness.png", dpi=180)
    plt.close(fig)
    print(f"局部有效刚度与增益估算已保存：{output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
