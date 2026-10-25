"""运动段指标必须排除接近/末态，失败与提前终止不能被统计掩盖。"""
import numpy as np
import pandas as pd
import pytest

from experiments.c1_sensor_study import analyze


def sample():
    config = {"duration_s": 9., "wipe_start_s": 2., "wipe_leg_duration_s": 2.,
              "physics_dt_s": 1., "force_target_n": 5., "wipe_length_m": .01,
              "endpoint_tolerance_m": .001, "max_tangential_error_m": .003}
    frame = pd.DataFrame({"time_s": np.arange(10.), "normal_force_n": [1000., 1000., 4., 5., 6., 7., 2000., 2000., 2000., 2000.],
                          "tool_contact_count": [1]*10, "tool_x_m": [0., 0., 0., .005, .01, .005, 0., 0., 0., 0.],
                          "tool_y_m": [0.]*10, "tangential_error_m": [0.]*10})
    return config, frame


def test_physical_dropout_intervals_and_motion_only_force():
    config, frame = sample()
    frame.loc[[3, 4], "tool_contact_count"] = 0
    result = analyze(frame, config)
    assert result["运动法向力RMSE_N"] == pytest.approx(np.sqrt(1.5))
    assert result["运动正向峰值偏差_N"] == 2.
    assert result["接触丢失累计_s"] == result["最长连续丢失_s"] == 2.
    assert result["接触丢失次数"] == 1 and result["首次丢失时刻_s"] == 3.
    assert result["运动接触保持率"] == .5
    assert not result["连续接触与实际往返验收"]


def test_incomplete_motion_has_no_complete_metrics():
    config, frame = sample()
    result = analyze(frame.iloc[:5], config)
    assert not result["完整运动段"] and not result["完整时长"]
    assert result["已记录运动样本数"] == 3
    assert "运动法向力RMSE_N" not in result and "接触丢失累计_s" not in result


def test_no_dropout_and_completed_motion_before_later_failure():
    config, frame = sample()
    result = analyze(frame.iloc[:7], config)
    assert result["完整运动段"] and not result["完整时长"]
    assert result["连续接触与实际往返验收"]  # 不代表外层完整任务通过。
    assert result["首次丢失时刻_s"] is None
    assert result["接触丢失次数"] == result["最长连续丢失_s"] == 0
