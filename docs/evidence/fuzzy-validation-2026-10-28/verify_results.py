"""从仓库根运行，独立重构本轮30次仿真的测量、控制输出和真值统计。"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from common.force_sensor import ForceSensor
from controllers.force_pi import ForcePI
from controllers.fuzzy_pi import FuzzyAdaptivePI
from experiments.c1_sensor_study import analyze as analyze_c1
from experiments.m1_sensor_study import analyze as analyze_m1

OUT = ROOT / "docs/evidence/fuzzy-validation-2026-10-28"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    OUT.mkdir(parents=True, exist_ok=True)
    checks, pairs, history = [], [], []
    immutable = sorted([*ROOT.glob("controllers/*.py"), *ROOT.glob("common/*.py"),
                        ROOT / "experiments/m0_03_contact_force.py", ROOT / "experiments/m1_force_pi.py",
                        ROOT / "experiments/c1_wipe.py"])
    for file in immutable:
        tracked = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                           "show", "HEAD:"+file.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert tracked.replace(b"\r\n", b"\n") == file.read_bytes().replace(b"\r\n", b"\n"), file
    for kind, count in [("M", 12), ("C", 18)]:
        source = ROOT / f"results/V20261028/{kind}/run_001"
        table = pd.read_csv(source / "comparison.csv")
        assert len(table) == count
        controls, frames, configs, scenes, assets = {}, {}, {}, {}, {}
        for row in table.to_dict("records"):
            folder = source / row["数据目录"]
            if kind == "M":
                folder = folder / "pi/run_001"
            config, summary = read(folder / "config.json"), read(folder / "summary.json")
            for name, digest in summary["source_sha256"].items():
                assert sha(ROOT / name) == digest, name
            assert sha(folder / "contact_scene.xml") == summary["scene_sha256"]
            for name, digest in summary["asset_sha256"].items():
                assert sha(ROOT / name) == digest, name
            command = pd.read_csv(folder / "control.csv")
            frame = pd.read_csv(folder / "contact_force.csv")
            physical = frame.set_index("time_s").loc[command.time_s, "normal_force_n"].to_numpy()
            np.testing.assert_allclose(physical, command.sensor_true_force_n, atol=1e-12, rtol=0)
            np.testing.assert_allclose(command.measured_force_n, command.sensor_filtered_force_n, atol=1e-12, rtol=0)
            sensor = ForceSensor(config, config["control_dt_s"])
            reconstructed, source_times = [], []
            for force, time in zip(physical, command.time_s):
                value, detail = sensor.step(force, time)
                reconstructed.append(value)
                source_times.append(detail["sensor_source_time_s"])
            np.testing.assert_allclose(reconstructed, command.measured_force_n, atol=1e-12, rtol=0)
            np.testing.assert_allclose(source_times, command.sensor_source_time_s, atol=1e-12, rtol=0)
            active = command[command.pi_active]
            controller = (FuzzyAdaptivePI if row["算法"] == "fuzzy" else ForcePI)(config)
            controller.reset(float(active.force_error_n.iloc[0]))
            replay = []
            for sample in active.itertuples():
                offset, detail = controller.step(sample.force_target_n, sample.measured_force_n, config["control_dt_s"])
                replay.append([controller.kp, controller.ki, detail["pi_p_m"], detail["pi_i_m"], offset])
            fields = ["pi_kp_m_per_n", "pi_ki_m_per_ns", "pi_p_m", "pi_i_m", "pi_offset_m"]
            np.testing.assert_allclose(replay, active[fields], atol=1e-12, rtol=0)
            if row["算法"] == "fuzzy":
                assert active.pi_kp_m_per_n.between(config["pi_kp_m_per_n"]*.5, config["pi_kp_m_per_n"]*1.5).all()
                assert active.pi_ki_m_per_ns.between(config["pi_ki_m_per_ns"]*.5, config["pi_ki_m_per_ns"]*1.5).all()
                assert np.ptp(active.pi_kp_m_per_n) > 1e-12 and np.ptp(active.pi_ki_m_per_ns) > 1e-12
            else:
                assert active.pi_kp_m_per_n.eq(config["pi_kp_m_per_n"]).all()
                assert active.pi_ki_m_per_ns.eq(config["pi_ki_m_per_ns"]).all()
            assert command.pi_offset_m.abs().max() <= config["pi_offset_limit_m"]+1e-12
            assert command.pi_i_m.abs().max() <= config["pi_integral_limit_m"]+1e-12
            assert command.pi_offset_m.diff().abs().max() <= config["pi_offset_speed_m_s"]*config["control_dt_s"]+1e-12
            assert command[[f"ctrl_{i}" for i in range(7)]].diff().abs().max().max() <= config["max_joint_reference_speed_rad_s"]*config["control_dt_s"]+1e-12
            metrics = (analyze_m1 if kind == "M" else analyze_c1)(frame, config)
            metric_key = "接管全程RMSE_N" if kind == "M" else "运动法向力RMSE_N"
            assert abs(metrics[metric_key]-row[metric_key]) < 1e-12
            passed_key = "验收通过" if kind == "M" else "任务验收通过"
            outer = read(source / row["数据目录"] / "summary.json")
            assert bool(row[passed_key]) == outer["passed"]
            checks.append({"kind": kind, "case": row["工况"], "algorithm": row["算法"],
                           "passed": bool(row[passed_key]), "active_samples": len(active),
                           "measurement_max_error_n": float(np.max(np.abs(np.array(reconstructed)-command.measured_force_n))),
                           "pi_replay_max_error": float(np.max(np.abs(np.array(replay)-active[fields].to_numpy()))),
                           "source_files_verified": len(summary["source_sha256"]),
                           "asset_files_verified": len(summary["asset_sha256"]),
                           "git_head_at_run": summary["git_head_at_run"]})
            key = row["工况"], row["算法"]
            controls[key], frames[key], configs[key] = command, frame, config
            scenes[key], assets[key] = summary["scene_sha256"], summary["asset_sha256"]
        for key in table["工况"].unique():
            a, b = [controls[key, name] for name in ["fixed", "fuzzy"]]
            ca, cb = [configs[key, name] for name in ["fixed", "fuzzy"]]
            assert {k for k in ca.keys() | cb.keys() if ca.get(k) != cb.get(k)} == {"pi_algorithm"}
            assert scenes[key, "fixed"] == scenes[key, "fuzzy"]
            assert assets[key, "fixed"] == assets[key, "fuzzy"]
            np.testing.assert_array_equal(a.time_s, b.time_s)
            np.testing.assert_allclose(a.sensor_raw_force_n-a.sensor_delayed_force_n,
                                       b.sensor_raw_force_n-b.sensor_delayed_force_n, atol=2e-14, rtol=0)
            fa, fb = [frames[key, name] for name in ["fixed", "fuzzy"]]
            before = fa.time_s < ca["force_control_start_s"]
            np.testing.assert_array_equal(fa.loc[before, "normal_force_n"], fb.loc[before, "normal_force_n"])
            pairs.append({"kind": kind, "case": key, "only_algorithm_changed": True,
                          "paired_noise_and_pre_takeover_force": "PASS", "same_scene_and_assets": True})
        old_folder = "fuzzy-pi-2026-10-25" if kind == "M" else "c1-sensor-study-2026-10-25"
        old = pd.read_csv(ROOT / "docs/evidence" / old_folder / "comparison.csv")
        for old_row in old.to_dict("records"):
            algorithm = old_row.get("算法", "fixed")
            new = table[(table["工况"] == old_row["工况"]) & (table["算法"] == algorithm)].iloc[0]
            metric = "末秒RMSE_N" if kind == "M" else "运动法向力RMSE_N"
            difference = abs(new[metric]-old_row[metric])
            assert difference < 1e-12
            history.append({"kind": kind, "case": old_row["工况"], "algorithm": algorithm, "rmse_difference_n": difference})
    for file in [*ROOT.glob("controllers/*.py"), *ROOT.glob("experiments/*.py")]:
        ast.parse(file.read_text(encoding="utf-8-sig"))
    result = {"date": "2026-10-28", "trial_checks": checks, "pair_checks": pairs,
              "historical_metric_reproduction": history,
              "unchanged_control_source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in immutable},
              "new_runner_sha256": sha(ROOT / "experiments/c1_fuzzy_comparison.py"),
              "test_results": {"initial_long_path": "9 PASS / 1 model-load FAIL at t=0",
                               "short_path_fuzzy": "10 PASS, 14 dependency warnings, 5.51 s",
                               "c1_metrics": "3 PASS, 0.41 s"},
              "not_run": ["full regression suite this round", "real robot", "real sensor", "interactive viewer"]}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"trials": len(checks), "pairs": len(pairs), "historical_metrics": len(history),
                      "measurement_max_error_n": max(r["measurement_max_error_n"] for r in checks),
                      "pi_replay_max_error": max(r["pi_replay_max_error"] for r in checks),
                      "status": "PASS"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
