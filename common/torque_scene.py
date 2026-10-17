"""由已验证接触场景派生独立力矩MJCF，绝不修改上游资源。"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.contact_scene import build_scene, numbers, reset_home


def build_torque_scene(config, directory):
    _, _, placement = build_scene(config, directory / "panda_position.xml")
    root = ET.parse(directory / "panda_position.xml").getroot()
    root.set("model", "panda_torque_contact")
    for joint, limit in zip(config["arm_joints"], config["torque_limits_nm"], strict=True):
        element = root.find(f"actuator/*[@joint='{joint}']")
        name = element.get("name")
        element.attrib.clear()  # 删除位置增益和位置偏置，显式指定单位增益、零偏置。
        element.set("name", name)
        element.attrib.update(joint=joint, dyntype="none", gaintype="fixed", biastype="none",
                              gainprm="1", biasprm="0 0 0", gear="1", ctrllimited="true", forcelimited="true",
                              ctrlrange=numbers([-limit, limit]), forcerange=numbers([-limit, limit]))
    key = root.find(f"keyframe/key[@name='{config['home_key']}']")
    controls = np.fromstring(key.get("ctrl"), sep=" ")
    # XML执行器次序不假设夹爪是第八个臂关节，只按目标关节筛选。
    for index, element in enumerate(root.find("actuator")):
        if element.get("joint") in config["arm_joints"]:
            controls[index] = 0
    key.set("ctrl", numbers(controls))
    ET.indent(root)
    with (directory / "panda_torque.xml").open("x", encoding="utf-8") as stream:
        stream.write(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(directory / "panda_torque.xml"))
    data = reset_home(model, config["home_key"])
    arm = np.array([model.joint(name).id for name in config["arm_joints"]])
    qa, va = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
    acts = np.array([next(a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == j) for j in arm])
    if (not np.all(model.actuator_biastype[acts] == mujoco.mjtBias.mjBIAS_NONE) or
            not np.all(model.actuator_gaintype[acts] == mujoco.mjtGain.mjGAIN_FIXED) or
            not np.all(model.actuator_dyntype[acts] == mujoco.mjtDyn.mjDYN_NONE) or
            not np.all(model.actuator_gainprm[acts, 0] == 1) or
            not np.all(model.actuator_gear[acts] == np.array([1, 0, 0, 0, 0, 0]))):
        raise ValueError("编译结果不是预期的单位力矩执行器")
    return model, data, placement, qa, va, acts
