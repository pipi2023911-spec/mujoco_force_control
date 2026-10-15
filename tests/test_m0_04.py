import json

import pandas as pd
import pytest

from experiments.m0_04_depth_comparison import analyze_trial, check_trend, load_config, run


def test_real_depth_comparison_and_corrupt_force_detection(tmp_path):
    config = load_config()
    output = run(config, tmp_path)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["已完成组数"] == 3
    table = pd.read_csv(output / "comparison.csv")
    assert (table["末段反向法向投影_N"] < 0).all()
    assert (table["末段实际穿透_mm"] > 0).all()
    assert (table["坐标重建最大误差_N"] < config["force_reconstruction_tolerance_n"]).all()
    assert (output / "depth_comparison.png").exists()
    # 三组初态必须相同；对照变量仅是参考行程。
    states = [pd.read_csv(output / path / "joint_state.csv").iloc[0] for path in table["数据目录"]]
    for state in states[1:]:
        pd.testing.assert_series_equal(states[0], state)
    # 篡改一条局部力记录，验证坐标审计确实发现不一致，而非只检查文件存在。
    trial = output / table["数据目录"].iloc[0]
    raw = pd.read_csv(trial / "raw_contacts.csv")
    raw.loc[0, "local_force_normal_n"] += 1
    raw.to_csv(trial / "raw_contacts.csv", index=False)
    with pytest.raises(ValueError, match="坐标转换"):
        analyze_trial(trial, config)


def test_reversed_force_trend_is_rejected():
    table = pd.DataFrame({"末段平均法向力_N": [6, 5, 4], "末段实际穿透_mm": [.01, .02, .03]})
    with pytest.raises(ValueError, match="法向力"):
        check_trend(table, load_config())


def test_invalid_sweep_keeps_failure_report(tmp_path):
    config = load_config()
    config["approach_distances_m"] = [.014, .013]
    with pytest.raises(ValueError, match="递增"):
        run(config, tmp_path)
    report = json.loads((tmp_path / "run_001" / "summary.json").read_text(encoding="utf-8"))
    assert not report["passed"] and report["已完成组数"] == 0
