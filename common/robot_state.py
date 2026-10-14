"""Read compiled model addresses; never equate arm DOFs with full state size."""
import mujoco
import numpy as np


def inspect_model(model, data, config):
    joints = []
    for j in range(model.njnt):
        joints.append({
            "name": model.joint(j).name, "id": j,
            "type": mujoco.mjtJoint(int(model.jnt_type[j])).name,
            "qpos_address": int(model.jnt_qposadr[j]),
            "qvel_address": int(model.jnt_dofadr[j]),
            "limited": bool(model.jnt_limited[j]),
            "range": model.jnt_range[j].tolist(),
        })
    actuators = []
    for a in range(model.nu):
        trn = int(model.actuator_trntype[a])
        target = int(model.actuator_trnid[a, 0])
        if trn == mujoco.mjtTrn.mjTRN_JOINT:
            target_name = model.joint(target).name
        elif trn == mujoco.mjtTrn.mjTRN_TENDON:
            target_name = model.tendon(target).name
        else:
            raise ValueError(f"未知传动类型： {trn}")
        actuators.append({
            "name": model.actuator(a).name, "ctrl_index": a,
            "transmission": mujoco.mjtTrn(trn).name, "target": target_name,
            "dyntype": mujoco.mjtDyn(int(model.actuator_dyntype[a])).name,
            "gaintype": mujoco.mjtGain(int(model.actuator_gaintype[a])).name,
            "biastype": mujoco.mjtBias(int(model.actuator_biastype[a])).name,
            "gainprm": model.actuator_gainprm[a].tolist(),
            "biasprm": model.actuator_biasprm[a].tolist(),
            "gear": model.actuator_gear[a].tolist(),
            "ctrlrange": model.actuator_ctrlrange[a].tolist(),
            "forcerange": model.actuator_forcerange[a].tolist(),
        })
    arm_ids = np.array([model.joint(n).id for n in config["arm_joints"]])
    finger_ids = [model.joint(n).id for n in config["finger_joints"]]
    if len(arm_ids) != 7 or len(finger_ids) != 2 or model.njnt != 9:
        raise ValueError("模型必须包含七个臂关节和两个夹爪关节")
    if not np.all(model.jnt_type[arm_ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("臂关节类型不匹配")
    if not np.all(model.jnt_type[finger_ids] == mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError("夹爪关节类型不匹配")
    for name in config["arm_joints"]:
        matches = [a for a in actuators if a["transmission"] == "mjTRN_JOINT" and a["target"] == name]
        if len(matches) != 1:
            raise ValueError(f"执行器缺失或映射不唯一： {name}")
    if model.nu != 8 or sum(a["transmission"] == "mjTRN_TENDON" for a in actuators) != 1:
        raise ValueError("夹爪执行器映射异常")
    tendons = []
    for t in range(model.ntendon):
        terms = []
        for w in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t]):
            if model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_JOINT:
                raise ValueError("此模型应使用固定关节腱传动")
            terms.append({"joint": model.joint(int(model.wrap_objid[w])).name,
                          "coefficient": float(model.wrap_prm[w])})
        tendons.append({"name": model.tendon(t).name, "terms": terms})
    return {"njnt": model.njnt, "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "qpos_shape": list(data.qpos.shape), "qvel_shape": list(data.qvel.shape),
            "ctrl_shape": list(data.ctrl.shape), "joints": joints, "actuators": actuators,
            "tendons": tendons, "neq": model.neq,
            "default_qpos0": model.qpos0.tolist(), "initial_qpos": data.qpos.tolist(),
            "initial_qvel": data.qvel.tolist(), "initial_ctrl": data.ctrl.tolist(),
            "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "gravity": model.opt.gravity.tolist()}


def check_state(model, data, tolerance):
    """Raise on numerical failure, solver warning, or out-of-range command/state."""
    for name in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"):
        if not np.isfinite(getattr(data, name)).all():
            raise ValueError(f"检测到非有限数值： {name}")
    if np.any(data.warning.number):
        raise ValueError(f"MuJoCo 引擎警告： {data.warning.number.tolist()}")
    q = data.qpos[model.jnt_qposadr]
    limited = model.jnt_limited.astype(bool)
    if np.any(limited & ((q < model.jnt_range[:, 0] - tolerance) |
                         (q > model.jnt_range[:, 1] + tolerance))):
        raise ValueError("关节超出限位")
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(limited & ((data.ctrl < model.actuator_ctrlrange[:, 0]) |
                         (data.ctrl > model.actuator_ctrlrange[:, 1]))):
        raise ValueError("控制输入超出限位")
