"""实验统计不能把提前终止或不达标的真值曲线算作通过。"""
import pandas as pd
import pytest

from experiments.m1_force_pi import load_config, tracking_metrics
from experiments.m1_sensor_study import analyze


def test_incomplete_trial_has_no_steady_or_settling_metrics():
    result = analyze(pd.DataFrame({"time_s": [0., 1.]}), load_config())
    assert result == {"完整时长": False}


def test_failed_tracking_can_be_reported_without_relaxing_acceptance():
    config = {**load_config(), "control_mode": "pi"}
    frame = pd.DataFrame({"time_s": [8., 9., 10., 11., 12.],
                          "normal_force_n": [4.8] * 5, "tool_contact_count": [1] * 5})
    with pytest.raises(ValueError, match="验收未通过"):
        tracking_metrics(frame, config)
    result = analyze(frame, config)
    assert result["末秒RMSE_N"] == pytest.approx(.2)
    assert result["接管全程RMSE_N"] == pytest.approx(.2)
    assert result["接管后稳定时间_秒"] is None
