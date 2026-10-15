import json

import numpy as np
import pandas as pd
import pytest

from experiments.m0_05_contact_exit import check_exit, exit_reference, load_config, run


def test_real_withdrawal_and_zero_force_audit(tmp_path):
    config = load_config()
    output = run(config, tmp_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] and summary["state_samples"] == 18001
    assert summary["hold_mean_force_n"] > 1
    assert summary["末段平均法向力_N"] == 0
    assert summary["解除后持续无接触时长_秒"] > 1
    frame = pd.read_csv(output / "contact_force.csv")
    with pytest.raises(ValueError, match="回碰"):
        check_exit(frame, {**config, "recontact_force_max_n": .01})
    after = frame[frame.time_s >= summary["稳定解除接触时间_秒"]]
    assert (after.tool_contact_count == 0).all()
    assert (after[[f"tool_force_{a}_n" for a in "xyz"]] == 0).all().all()
    # 只篡改无接触时的切向力，确认验收检查三维力，而非只检查法向力。
    frame.loc[frame.index[-1], "tool_force_x_n"] = .01
    with pytest.raises(ValueError, match="未归零"):
        check_exit(frame, config)


def test_exit_reference_continuous_and_speed_bounded():
    config = load_config()
    p, n = np.array([.5, 0, .4]), np.array([0., 0., 1.])
    times = np.arange(0, config["duration_s"], config["control_dt_s"])
    positions = np.array([exit_reference(t, p, n, config)[0] for t in times])
    np.testing.assert_allclose(positions[0], positions[-1], atol=1e-12)
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() / config["control_dt_s"] < config["max_reference_speed_m_s"]
    start = config["contact_hold_end_s"]
    np.testing.assert_allclose(exit_reference(start-1e-8,p,n,config)[0],exit_reference(start+1e-8,p,n,config)[0],atol=1e-10)


def test_invalid_exit_timing_preserves_failure(tmp_path):
    config = load_config()
    config["withdraw_duration_s"] = 30
    with pytest.raises(ValueError, match="离面保持"):
        run(config, tmp_path)
    report = json.loads((tmp_path / "run_001" / "summary.json").read_text(encoding="utf-8"))
    assert not report["passed"]
