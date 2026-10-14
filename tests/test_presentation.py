import json

import mujoco
import mujoco.viewer
import pandas as pd
import pytest

from common.chinese_output import save_csv
from common.experiment_view import ExperimentView
from experiments.m0_01_model_inspection import save_json


def test_chinese_reports_keep_original_interface(tmp_path):
    save_json(tmp_path / "summary.json", {"passed": True, "initial_gap_m": 0.02})
    original = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    translated = json.loads((tmp_path / "summary_中文.json").read_text(encoding="utf-8"))
    assert original["passed"] is True
    assert translated == {"验收通过": "是", "初始间隙_米": 0.02}
    save_csv(pd.DataFrame({"time_s": [0], "qpos_7": [0.04], "ctrl_7": [255]}), tmp_path / "data.csv")
    columns = pd.read_csv(tmp_path / "data_中文.csv").columns.tolist()
    assert columns == ["仿真时间_秒", "夹爪关节1位置_米", "夹爪控制输入_0至255"]


def test_viewer_waits_and_cannot_change_physics(monkeypatch):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    source = mujoco.MjData(model)
    state = {}

    class FakeWindow:
        running = True
        syncs = 0
        def is_running(self):
            return self.running
        def sync(self):
            self.syncs += 1
            if self.syncs == 2:
                state["callback"](32)
            if state.get("ending"):
                self.running = False
        def close(self):
            self.running = False

    window = FakeWindow()
    def launch(model_copy, data_copy, key_callback):
        state["callback"] = key_callback
        model_copy.opt.gravity[:] = 0  # 模拟用户改变显示窗口的物理选项。
        data_copy.qpos[:] = 1
        return window
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr("common.experiment_view.time.sleep", lambda _: None)
    with ExperimentView(model, True) as view:
        view.wait_start(source)
        assert view.started and window.syncs == 2
        assert source.time == 0 and source.qpos[0] == 0
        assert model.opt.gravity[2] == pytest.approx(-9.81)
        view.update(source)
        state["ending"] = True
        view.wait_end(source)
        assert source.time == 0


def test_viewer_rejects_early_close(monkeypatch):
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    data = mujoco.MjData(model)
    class ClosedWindow:
        def is_running(self): return False
        def close(self): pass
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: ClosedWindow())
    with ExperimentView(model, True) as view:
        with pytest.raises(RuntimeError, match="尚未开始"):
            view.wait_start(data)


@pytest.mark.parametrize("speed", [0, -1, float("nan")])
def test_invalid_display_speed(speed):
    with pytest.raises(ValueError, match="播放速度"):
        ExperimentView(None, False, speed)
