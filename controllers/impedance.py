"""笛卡尔阻抗及动态一致零空间力矩；输入为普通NumPy数组。"""
import numpy as np


def impedance_torque(position_error, rotation_error, desired_velocity, jacobian,
                     velocity, mass_matrix, bias, posture_error, config):
    kp = np.r_[config["translation_stiffness_n_m"], np.full(3, config["rotation_stiffness_nm_rad"])]
    kd = np.r_[config["translation_damping_ns_m"], np.full(3, config["rotation_damping_nms_rad"])]
    limits = np.asarray(config["torque_limits_nm"])
    if not np.isfinite(np.r_[kp,kd,limits]).all() or np.any(np.r_[kp,kd,limits]<=0):
        raise ValueError("阻抗增益和力矩限幅必须为有限正数")
    error = np.r_[position_error, rotation_error]
    wrench = kp * error + kd * (np.r_[desired_velocity, np.zeros(3)] - jacobian @ velocity)
    task = jacobian.T @ wrench
    minv_jt = np.linalg.solve(mass_matrix, jacobian.T)
    jbar = minv_jt @ np.linalg.pinv(jacobian @ minv_jt, rcond=config["null_pinv_rcond"], hermitian=True)
    projector = np.eye(len(velocity)) - jacobian.T @ jbar.T  # 力矩投影：J M^-1 N_tau ≈ 0。
    null = projector @ (config["null_stiffness_nm_rad"] * posture_error - config["null_damping_nms_rad"] * velocity)
    raw = task + bias + null
    torque = np.clip(raw, -limits, limits)
    if not np.isfinite(torque).all():
        raise ValueError("阻抗力矩出现非有限数值")
    return torque, {"task": task, "null": null, "bias": bias.copy(), "raw": raw,
                    "wrench": wrench, "limited": bool(np.any(np.abs(raw-torque)>1e-12)),
                    "null_leak": float(np.linalg.norm(jacobian @ np.linalg.solve(mass_matrix, null)))}
