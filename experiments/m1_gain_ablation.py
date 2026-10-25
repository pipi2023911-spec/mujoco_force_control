"""区分固定增益选择与在线调度作用；固定PI、固定Ki=1.1倍、模糊PI配对。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.chinese_output import save_csv
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m1_force_pi import load_config, run as run_pi
from experiments.m1_sensor_study import analyze


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/A")
    base = load_config()
    scenarios = [("zero", {}), *[(f"combo{s}", {"sensor_noise_std_n": .05,
                  "sensor_delay_steps": 2, "sensor_lpf_cutoff_hz": 20., "sensor_seed": s}) for s in (42, 43, 44)]]
    methods = [("fixed", {}), ("fixed_ki110", {"pi_ki_m_per_ns": base["pi_ki_m_per_ns"]*1.1}),
               ("fuzzy", {"pi_algorithm": "fuzzy"})]
    save_json(output / "config.json", {"base": base, "scenarios": scenarios, "methods": methods,
              "说明": "1.1是模糊规则在零误差/零变化率时的Ki倍率，不是从本轮结果搜索出的最优值"})
    rows = []
    for i, (key, sensor) in enumerate(scenarios):
        for j, (method, change) in enumerate(methods):
            config = {**base, **sensor, **change}
            parent = output / f"c{i}" / f"m{j}"
            error = ""
            try:
                run_pi(config, parent, mode="pi")
            except ValueError as exc:
                error = str(exc)
            folder = parent / "run_001"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "pi/run_001/contact_force.csv")
            rows.append({"工况": key, "方法": method, "passed": summary["passed"], "error": error,
                         **analyze(frame, config), "数据目录": folder.relative_to(output).as_posix()})
            save_csv(pd.DataFrame(rows), output / "comparison.csv")
    table = pd.DataFrame(rows)
    stats = table[table["工况"].str.startswith("combo")].groupby("方法")["末秒RMSE_N"].agg(["mean", "std", "count"])
    save_csv(stats.reset_index(), output / "combo_statistics.csv")
    save_json(output / "summary.json", {"运行次数": len(rows), "通过次数": int(table.passed.sum()),
              "限制": "单一接触、单一目标、三种子；固定Ki110不是最优固定PI，也不是完整参数搜索"})
    print(f"固定增益消融已记录：{output}", flush=True)
    return output


if __name__ == "__main__":
    run()
