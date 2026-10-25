"""局部平稳滑动的逆/正向接触一致性与速度包络；不执行控制或修改默认任务。"""
import argparse
import hashlib
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pandas as pd

from common.chinese_output import save_csv, configure_chinese_font
from common.contact_scene import build_scene
from common.contact_feedback import contact_force
from common.kinematics import solve_pose
from experiments.c1_wipe import load_config
from experiments.m0_01_model_inspection import ROOT, new_run, save_json


def local_state(model, home, site, arm, target, rotation, speed, config):
    """构造给定局部位姿、切向TCP速度、零关节加速度的诊断快照。"""
    data = mujoco.MjData(model)
    data.qpos[:] = home.qpos
    solve_pose(model, data, site, arm, target, rotation,
               {**config, "ik_position_tolerance_m": 1e-11, "ik_orientation_tolerance_rad": 1e-10})
    mujoco.mj_forward(model, data)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site)
    va = model.jnt_dofadr[arm]
    jac = np.vstack([jp[:, va], jr[:, va]])
    data.qvel[va] = np.linalg.pinv(jac) @ np.array([speed, 0, 0, 0, 0, 0])
    data.qacc[:] = 0.
    mujoco.mj_inverse(model, data)
    return data


def run(output_root=None):
    output = new_run(output_root or ROOT / "results/V")
    config = load_config()
    save_json(output / "config.json", config)
    model, home, report = build_scene(config, output / "contact_scene.xml")
    site, tool, plate = model.site("tool_tip").id, model.geom("tool_sphere").id, model.geom("contact_plane").id
    arm = np.array([model.joint(n).id for n in config["arm_joints"]])
    qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
    acts = [next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == j) for j in arm]
    rotation = home.site_xmat[site].reshape(3, 3).copy()
    target = home.site_xpos[site].copy()
    target[0] += config["wipe_length_m"] / 2
    touch = home.geom_xpos[plate, 2] + model.geom_size[plate, 2] + config["tool_radius_m"]
    rows = []
    for speed in [0., .002, .00234375, .0025, .003125, .00625, -.00625]:
        for depth in [1e-8, 1e-6, 5e-6, 2e-5, 5e-5]:
            target[2] = touch-depth
            data = local_state(model, home, site, arm, target, rotation, speed, config)
            inverse = contact_force(model, data, tool, plate, [0, 0, 1])["normal_force_n"]
            tau = data.qfrc_inverse.copy()
            # 七轴通过原位置执行器回代。仅手指残差用qfrc_applied补齐全模型。
            data.ctrl[acts] = data.qpos[qa] + (tau[va] - model.actuator_biasprm[acts, 2]*data.qvel[va]) / model.actuator_gainprm[acts, 0]
            mujoco.mj_fwdActuation(model, data)
            data.qfrc_applied[:] = tau - data.qfrc_actuator
            data.qfrc_applied[va] = 0.
            mujoco.mj_forward(model, data)
            forward = contact_force(model, data, tool, plate, [0, 0, 1])["normal_force_n"]
            rows.append({"切向速度_m每s": speed, "压入量_m": depth,
                         "逆解法向力_N": inverse, "正解法向力_N": forward,
                         "正逆差_N": abs(inverse-forward), "正向qacc范数": float(np.linalg.norm(data.qacc)),
                         "臂额外广义力范数": float(np.linalg.norm(data.qfrc_applied[va])),
                         "执行器力矩余量_Nm": float(np.min(model.actuator_forcerange[acts, 1] - np.abs(tau[va]))),
                         "位置命令有效": bool(np.all(data.ctrl[acts] >= model.actuator_ctrlrange[acts, 0])
                                               and np.all(data.ctrl[acts] <= model.actuator_ctrlrange[acts, 1]))})
    table = pd.DataFrame(rows)
    save_csv(table, output / "force_envelope.csv")
    edge = table[table["压入量_m"] == 1e-8].set_index("切向速度_m每s").sort_index()
    slope = float((edge.loc[.00625, "逆解法向力_N"]-edge.loc[0., "逆解法向力_N"]) / .00625)
    critical = float((config["force_target_n"]-edge.loc[0., "逆解法向力_N"]) / slope)
    passed = bool(table["正逆差_N"].max() < 1e-7 and table["正向qacc范数"].max() < 1e-6
                  and (table["执行器力矩余量_Nm"] > 0).all() and table["位置命令有效"].all())
    save_json(output / "summary.json", {"passed": passed, "model_report": report,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "局部边界力速度斜率_Ns每m": slope, "局部5N临界速度_m每s": critical,
              "10mm五次曲线最小单程时间局部估计_s": 1.875*config["wipe_length_m"]/critical,
              "限制": "局部零关节加速度快照、有限深度扫描，不是全轨迹可行性或稳定性证明。正逆回代不是实际控制器运行。"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zh = configure_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(edge.index*1000, edge["逆解法向力_N"], "o-")
    ax.axhline(config["force_target_n"], color="black", linestyle="--", label="目标5N" if zh else "5 N target")
    ax.set(xlabel="切向速度（mm/s）" if zh else "Tangential speed (mm/s)",
           ylabel="近零压入逆解法向力（N）" if zh else "Near-zero-depth inverse force (N)")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "force_speed_envelope.png", dpi=180); plt.close(fig)
    if not passed:
        raise ValueError(f"正逆回代或执行器范围检查失败，诊断不能作为有效证据：{output}")
    print(f"局部可行性诊断已记录：{output}；临界速度约{critical*1000:.3f}mm/s", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output)
