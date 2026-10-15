"""M0-05：连续接近、保持、缓慢退出、离面保持；无力反馈调节。"""
import argparse
import json
from pathlib import Path

import numpy as np

from common.kinematics import approach_reference
from experiments.m0_01_model_inspection import ROOT
from experiments.m0_03_contact_force import load_config as load_base, run as run_base


def load_config(path=Path("configs/m0_05.json")):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {**load_base(Path(config["base_config"])), **config}


def exit_reference(time_s, start_position, normal, config):
    start, duration = config["contact_hold_end_s"], config["withdraw_duration_s"]
    distance = config["withdraw_distance_m"]
    if time_s == 0:  # 在真实运行的异常记录范围内检查时间与速度配置。
        values = [start, duration, distance, config["final_clearance_min_m"], config["withdraw_force_bin_s"]]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("退出参数必须是有限正数")
        if start - config["settle_duration_s"] - config["approach_duration_s"] < config["hold_metrics_window_s"]:
            raise ValueError("退出前必须保留完整接触保持统计窗口")
        if config["duration_s"] - start - duration < config["hold_metrics_window_s"]:
            raise ValueError("退出后必须保留完整离面保持统计窗口")
        if 1.875 * distance / duration > config["max_reference_speed_m_s"]:
            raise ValueError("退出参考速度超过上限")
    if time_s < start:
        return approach_reference(time_s, start_position, normal, config)
    u = np.clip((time_s - start) / duration, 0.0, 1.0)
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    # 从接触参考连续向 +法向移动，不重置真实状态、不直接改 qpos。
    position = start_position - normal * config["approach_distance_m"] + normal * distance * blend
    return position, "缓慢退出" if u < 1 else "离面保持"


def check_exit(frame, config):
    start = config["contact_hold_end_s"]
    retreat = frame[frame.time_s >= start]
    released = retreat[retreat.tool_contact_count == 0]
    if released.empty:
        raise ValueError("退出后未解除接触")
    release_time = float(released.time_s.iloc[0])
    # 首次零接触不一定等于稳定解除；明确记录软接触离面瞬态，不能隐藏回碰。
    touching = retreat[retreat.tool_contact_count > 0]
    last_touch = float(touching.time_s.iloc[-1]) if not touching.empty else start
    stable = retreat[(retreat.time_s > last_touch) & (retreat.tool_contact_count == 0)]
    if stable.empty:
        raise ValueError("实验结束前未稳定解除接触")
    stable_time = float(stable.time_s.iloc[0])
    recontact = retreat[(retreat.time_s > release_time) & (retreat.tool_contact_count > 0)]
    recontact_peak = float(recontact.normal_force_n.max()) if not recontact.empty else 0.0
    if stable_time - release_time > config["release_transition_max_s"] or recontact_peak > config["recontact_force_max_n"]:
        raise ValueError("离面回碰时长或接触力超限")
    # 所有无接触时刻必须是三维零力，不仅验证最后一个样本或法向分量。
    no_contact = frame[frame.tool_contact_count == 0]
    zero_error = float(np.abs(no_contact[[f"tool_force_{a}_n" for a in "xyz"]].to_numpy()).max())
    if zero_error > config["zero_force_tolerance_n"]:
        raise ValueError("无接触时的世界接触力未归零")
    final = frame[frame.time_s > config["duration_s"] - config["hold_metrics_window_s"]]
    if final.normal_gap_m.min() < config["final_clearance_min_m"] or (final.tool_contact_count != 0).any():
        raise ValueError("末段离面距离或无接触保持不合格")
    # 分箱均值检验下降趋势，保留原始瞬态，不要求每个物理步严格单调。
    groups = np.floor((retreat.time_s - start) / config["withdraw_force_bin_s"]).astype(int)
    means = retreat.groupby(groups).normal_force_n.mean()
    rise = float(max(0.0, np.diff(means.to_numpy()).max()))
    if rise > config["withdraw_force_rise_tolerance_n"]:
        raise ValueError("退出过程的分段平均力出现超限回升")
    return {"退出开始时间_秒": start, "首次解除接触时间_秒": release_time,
            "稳定解除接触时间_秒": stable_time,
            "离面过渡时长_秒": stable_time - release_time,
            "首次离面后回碰样本数": len(recontact), "离面回碰峰值力_N": recontact_peak,
            "解除后持续无接触时长_秒": float(frame.time_s.iloc[-1] - stable_time),
            "所有无接触样本最大力分量_N": zero_error,
            "末段最小离面间隙_mm": float(final.normal_gap_m.min() * 1000),
            "末段平均法向力_N": float(final.normal_force_n.mean()),
            "退出分箱平均力最大回升_N": rise,
            "接触保持统计窗口结束_秒": start}


def run(config, output_root=None, viewer=False, speed=1.0, snapshot=False):
    return run_base(config, output_root or ROOT / "results" / "M0-05", viewer, speed, snapshot,
                    reference_function=exit_reference, end_checks=check_exit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0_05.json"), help="退出实验配置")
    parser.add_argument("--viewer", action="store_true", help="空格开始，完整播放接近与退出，结束后关窗")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率，不改变物理步长")
    parser.add_argument("--snapshot", action="store_true", help="保存离面末态截图，需要 OpenGL")
    args = parser.parse_args()
    run(load_config(args.config), viewer=args.viewer, speed=args.speed, snapshot=args.snapshot)
