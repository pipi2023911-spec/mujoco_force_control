"""M0-03 的名义位置轨迹映射；没有力误差输入，不是力闭环控制器。"""
import mujoco
import numpy as np


def orientation_error(current, target):
    # 近姿态旋转误差：世界系三个对应轴叉积之和的一半；本实验转角很小。
    return 0.5 * sum(np.cross(current[:, i], target[:, i]) for i in range(3))


def solve_pose(model, reference_data, site_id, arm_ids, target_pos, target_rotation, config):
    """只修改独立的名义运动学数据，返回七个臂关节角，不改真实仿真 qpos。"""
    qa, va = model.jnt_qposadr[arm_ids], model.jnt_dofadr[arm_ids]
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    weight = config["ik_orientation_weight_m"]
    lower = model.jnt_range[arm_ids, 0] + config["joint_limit_margin_rad"]
    upper = model.jnt_range[arm_ids, 1] - config["joint_limit_margin_rad"]
    for _ in range(config["ik_max_iterations"]):
        mujoco.mj_kinematics(model, reference_data)
        mujoco.mj_comPos(model, reference_data)  # Jacobian 所需的质心位置信息。
        position_error = target_pos - reference_data.site_xpos[site_id]
        rotation_error = orientation_error(reference_data.site_xmat[site_id].reshape(3, 3), target_rotation)
        if (np.linalg.norm(position_error) <= config["ik_position_tolerance_m"] and
                np.linalg.norm(rotation_error) <= config["ik_orientation_tolerance_rad"]):
            return reference_data.qpos[qa].copy()
        mujoco.mj_jacSite(model, reference_data, jac_pos, jac_rot, site_id)
        jac = np.vstack((jac_pos[:, va], weight * jac_rot[:, va]))
        error = np.concatenate((position_error, weight * rotation_error))
        # 阻尼最小二乘：避免直接求逆奇异 Jacobian；7 自由度取局部最小范数增量。
        delta = jac.T @ np.linalg.solve(jac @ jac.T + config["ik_damping"] ** 2 * np.eye(6), error)
        delta *= min(1.0, config["ik_max_step_rad"] / max(np.linalg.norm(delta), 1e-15))
        reference_data.qpos[qa] = np.clip(reference_data.qpos[qa] + delta, lower, upper)
    raise ValueError("名义位置轨迹的逆运动学未收敛")


def approach_reference(time_s, start_position, normal, config):
    distance = config["approach_distance_m"]
    u = np.clip((time_s - config["settle_duration_s"]) / config["approach_duration_s"], 0.0, 1.0)
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5  # 五次平滑曲线：起止速度、加速度均为零。
    phase = "初始保持" if time_s < config["settle_duration_s"] else ("缓慢接近" if u < 1 else "固定位置保持")
    return start_position - normal * distance * blend, phase
