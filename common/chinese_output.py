"""中文展示副本；保留原始字段，兼容已有分析程序与历史日志。"""
import json

LABELS = {
    "display_speed": "图形播放速度倍率",
    "passed": "验收通过", "started_utc": "开始时间_UTC", "python": "Python版本",
    "platform": "运行平台", "versions": "软件版本", "model_dimensions": "模型维度",
    "njnt": "关节数量", "nq": "位置维度", "nv": "速度维度", "nu": "执行器数量",
    "simulated_duration_s": "仿真时长_秒", "physics_steps": "物理步数",
    "state_samples": "状态记录条数", "command_samples": "控制记录条数",
    "max_arm_drift_rad": "机械臂最大位置偏差_rad", "max_arm_speed_rad_s": "机械臂峰值速度_rad每秒",
    "final_arm_speed_rad_s": "机械臂末态最大速度_rad每秒", "max_ncon": "最大接触数",
    "warning_counts": "引擎警告计数", "initial_gap_m": "初始间隙_米",
    "minimum_gap_m": "最小间隙_米", "final_gap_m": "最终间隙_米",
    "max_gap_distance_disagreement_m": "两种间距计算最大差异_米",
    "error": "失败原因", "failed_at_sim_time_s": "失败时仿真时间_秒",
    "viewer_requested": "启用图形窗口", "snapshot_requested": "请求场景截图",
    "snapshot_saved": "截图已保存", "git_head_at_run": "运行时Git提交",
    "git_worktree_status": "运行时工作区状态", "source_sha256": "源码SHA256",
    "asset_sha256": "模型资源SHA256", "scene_sha256": "派生场景SHA256",
    "model_source": "模型来源", "repository": "上游仓库", "commit": "上游提交",
    "upstream_directory": "上游目录", "license": "许可证", "modifications": "修改说明",
    "model": "模型文件", "home_key": "初始关键帧", "physics_dt_s": "物理步长_秒",
    "control_dt_s": "控制周期_秒", "duration_s": "实验时长_秒",
    "joint_limit_tolerance": "关节限位容差", "max_final_arm_speed_rad_s": "末态速度验收上限_rad每秒",
    "arm_joints": "机械臂关节", "finger_joints": "夹爪关节", "joints": "关节映射",
    "actuators": "执行器映射", "tendons": "腱传动映射", "name": "名称", "id": "编号",
    "type": "类型", "qpos_address": "位置数组地址", "qvel_address": "速度数组地址",
    "limited": "启用限位", "range": "范围", "ctrl_index": "控制数组索引",
    "transmission": "传动类型", "target": "作用对象", "dyntype": "动态类型",
    "gaintype": "增益类型", "biastype": "偏置类型", "gainprm": "增益参数",
    "biasprm": "偏置参数", "gear": "传动比", "ctrlrange": "控制输入范围",
    "forcerange": "执行器力范围", "terms": "组成项", "joint": "关节", "coefficient": "系数",
    "neq": "等式约束数量", "qpos_shape": "位置数组形状", "qvel_shape": "速度数组形状",
    "ctrl_shape": "控制数组形状", "default_qpos0": "默认位置",
    "initial_qpos": "初始位置", "initial_qvel": "初始速度", "initial_ctrl": "初始控制输入",
    "integrator": "积分器", "gravity": "重力加速度",
    "tool_parent": "工具安装父刚体", "hand_position_world_m": "手部世界位置_米",
    "hand_rotation_world_from_local": "手部局部到世界旋转矩阵",
    "tool_center_home_world_m": "初始球心世界坐标_米", "tool_tip_definition": "工具参考点定义",
    "tool_radius_m": "工具球半径_米", "added_tool_mass_kg": "新增工具总质量_千克",
    "surface_origin_world_m": "接触面原点世界坐标_米", "plate_center_world_m": "薄板中心世界坐标_米",
    "normal_world": "世界坐标法向", "approach_direction_world": "世界坐标接近方向",
    "surface_definition": "接触面定义", "plate_half_size_m": "薄板半尺寸_米",
    "expected_initial_gap_m": "预期初始间隙_米", "tool_geom_id": "工具几何体编号",
    "plane_geom_id": "接触面几何体编号", "tool_site_id": "工具参考点编号",
    "base_config": "基础配置文件", "robot_xml": "机器人模型文件",
    "tool_center_local_m": "球心局部坐标_米", "tool_mass_kg": "球体质量_千克",
    "shaft_start_local_m": "连接杆起点局部坐标_米", "shaft_radius_m": "连接杆半径_米",
    "shaft_mass_kg": "连接杆质量_千克", "contact_friction": "接触摩擦参数",
    "contact_solref": "接触求解参考参数", "contact_solimp": "接触约束参数",
    "distance_query_max_m": "距离查询上限_米", "geometry_tolerance_m": "几何校验容差_米",
    "render_width": "图片宽度_像素", "render_height": "图片高度_像素",
    "camera_lookat_m": "相机观察中心_米", "camera_distance_m": "相机距离_米",
    "camera_azimuth_deg": "相机方位角_度", "camera_elevation_deg": "相机俯仰角_度",
    "time_s": "仿真时间_秒", "ncon": "接触数量", "normal_gap_m": "球面法向间隙_米",
    "geom_distance_m": "引擎几何距离_米", "inside_plate_footprint": "位于板面范围内",
    "tool_x_m": "球心世界X_米", "tool_y_m": "球心世界Y_米", "tool_z_m": "球心世界Z_米",
    "geom1": "几何体一", "geom2": "几何体二", "body2": "刚体二", "distance_m": "距离_米",
}


LABELS.update({
    "torque_limits_nm": "七轴力矩限幅_Nm", "translation_stiffness_n_m": "笛卡尔平移刚度_N每m",
    "translation_damping_ns_m": "笛卡尔平移阻尼_Ns每m", "rotation_stiffness_nm_rad": "姿态刚度_Nm每rad",
    "rotation_damping_nms_rad": "姿态阻尼_Nms每rad", "null_stiffness_nm_rad": "零空间姿态刚度_Nm每rad",
    "null_damping_nms_rad": "零空间阻尼_Nms每rad", "null_pinv_rcond": "零空间伪逆截断比例",
    "probe_duration_s": "力矩模型基础验证时长_秒", "probe_max_joint_drift_rad": "基础保持关节漂移上限_rad",
    "contact_transient_window_s": "初次接触统计窗口_秒",
    "admittance_start_s": "导纳接管时刻_秒", "admittance_mass_kg": "虚拟质量_kg",
    "admittance_damping_ns_m": "虚拟阻尼_Ns每m", "admittance_stiffness_n_m": "虚拟刚度_N每m",
    "admittance_offset_limit_m": "导纳位移限幅_米", "admittance_speed_limit_m_s": "导纳速度限幅_米每秒",
    "equilibrium_residual_limit_n": "虚拟弹簧平衡残差上限_N", "settling_force_band_n": "自身平衡力稳定带半宽_N",
    "cases": "对照工况", "selected_case": "选择工况", "external_force_n": "实测外部法向力_N",
    "admittance_x_m": "虚拟退让位移_米", "admittance_v_m_s": "虚拟速度_米每秒",
    "admittance_a_m_s2": "未限幅虚拟加速度_米每秒平方", "admittance_limited": "导纳状态受限",
    "admittance_active": "导纳已接管",
    "force_target_n": "目标法向力_N", "force_control_start_s": "力控制接管时刻_秒",
    "contact_confirm_s": "接管前持续接触确认时长_秒", "pi_kp_m_per_n": "PI比例增益_米每牛顿",
    "pi_ki_m_per_ns": "PI积分增益_米每牛顿秒", "pi_integral_limit_m": "PI积分贡献限幅_米",
    "pi_offset_limit_m": "PI位置修正限幅_米", "pi_offset_speed_m_s": "PI位置修正速度限幅_米每秒",
    "tracking_tolerance_n": "稳定带半宽_N", "settling_min_duration_s": "稳定带最短持续时长_秒",
    "steady_rmse_limit_n": "末秒RMSE上限_N", "steady_force_std_limit_n": "末秒力标准差上限_N",
    "comparison_rmse_ratio_limit": "PI与Baseline末秒RMSE比值上限", "control_mode": "控制模式",
    "requested_mode": "请求实验模式", "measured_force_n": "控制采样实际法向力_N",
    "feedback_time_s": "反馈采样时间_秒", "force_error_n": "目标减实际力_N",
    "pi_p_m": "PI比例贡献_米", "pi_i_m": "PI积分贡献_米", "pi_offset_m": "PI实际位置修正_米",
    "pi_unsaturated_m": "PI限幅前输出_米", "pi_integral_blocked": "积分被抗饱和暂停",
    "pi_output_limited": "PI输出受限", "pi_active": "PI已接管",
    "release_transition_max_s": "离面过渡时长上限_秒", "recontact_force_max_n": "离面回碰峰值上限_N",
    "contact_hold_end_s": "接触保持统计窗口结束_秒", "withdraw_duration_s": "退出行程时长_秒",
    "withdraw_distance_m": "退出参考行程_米", "final_clearance_min_m": "离面保持最小间隙_米",
    "withdraw_force_bin_s": "退出力趋势分箱时长_秒",
    "withdraw_force_rise_tolerance_n": "退出分箱平均力允许回升_N",
    "base_config": "基础实验配置", "approach_distances_m": "各组参考接近行程_米",
    "force_increment_min_n": "相邻组平均力增量下限_N",
    "penetration_increment_min_m": "相邻组实际穿透增量下限_米",
    "force_reconstruction_tolerance_n": "坐标重建误差容差_N",
    "minimum_normal_alignment": "受力与法向夹角余弦下限", "experiment_label": "实验编号",
    "scene_config": "接触场景配置文件", "settle_duration_s": "初始保持时长_秒",
    "approach_duration_s": "接近时长_秒", "approach_distance_m": "预设接近行程_米",
    "max_reference_speed_m_s": "位置参考速度上限_米每秒",
    "max_joint_reference_speed_rad_s": "关节参考速度上限_rad每秒",
    "ik_damping": "逆运动学阻尼系数", "ik_orientation_weight_m": "姿态误差加权长度_米",
    "ik_position_tolerance_m": "逆运动学位置容差_米",
    "ik_orientation_tolerance_rad": "逆运动学姿态容差_rad",
    "ik_max_iterations": "逆运动学最大迭代次数", "ik_max_step_rad": "逆运动学增量上限_rad",
    "joint_limit_margin_rad": "关节限位余量_rad", "max_measured_force_n": "接触合力验收上限_N",
    "max_penetration_m": "实际穿透验收上限_米", "max_lateral_drift_m": "最大横向偏移_米",
    "max_orientation_error_rad": "最大姿态偏差_rad", "contact_detection_force_n": "接触检测力阈值_N",
    "minimum_hold_mean_force_n": "末段平均接触力验收下限_N",
    "minimum_hold_contact_fraction": "末段接触时间占比验收下限",
    "hold_metrics_window_s": "末段统计窗口时长_秒", "zero_force_tolerance_n": "零力容差_N",
    "viewer_force_length_m_per_n": "力箭头显示长度比例_米每牛顿", "display_interval_s": "终端指标显示间隔_秒",
    "plot_width_in": "曲线图片宽度_英寸", "plot_height_in": "曲线图片高度_英寸", "plot_dpi": "曲线图片分辨率_DPI",
    "force_definition": "接触力定义", "raw_contact_rows": "逐接触点原始记录条数",
    "peak_force_n": "峰值法向接触力_N", "hold_mean_force_n": "接触保持窗口平均法向力_N",
    "hold_std_force_n": "接触保持窗口法向力标准差_N", "hold_contact_fraction": "接触保持窗口接触时间占比",
    "first_contact_time_s": "首次超过检测力阈值的时间_秒", "maximum_penetration_m": "最大实际穿透_米",
    "other_contact_count": "其他接触数量", "phase": "实验阶段", "normal_force_n": "工具法向接触力_N",
    "tool_contact_count": "工具球与薄板接触点数", "lateral_drift_m": "横向偏移_米",
    "orientation_error_rad": "姿态偏差_rad", "contact_index": "接触索引",
    "geom1_id": "接触几何体一编号", "geom2_id": "接触几何体二编号",
    "constraint_address": "约束地址", "contact_distance_m": "接触几何距离_米",
    "tool_force_sign": "工具受力方向修正系数", "local_force_normal_n": "接触局部法向力_N",
    "local_force_tangent1_n": "接触局部切向力一_N", "local_force_tangent2_n": "接触局部切向力二_N",
})
for axis in "xyz":
    LABELS[f"target_{axis}_m"] = f"位置参考{axis.upper()}_米"
    LABELS[f"tool_force_{axis}_n"] = f"工具受力世界{axis.upper()}_N"
    LABELS[f"contact_pos_{axis}_m"] = f"接触点世界{axis.upper()}_米"
for i in range(3):
    LABELS[f"local_torque_{i}_nm"] = f"接触局部力矩分量{i}_牛米"
    for j in range(3):
        LABELS[f"frame_{i}{j}"] = f"接触坐标矩阵_{i}{j}"


def chinese(value):
    if isinstance(value, dict):
        return {LABELS.get(k, k): chinese(v) for k, v in value.items()}
    if isinstance(value, list):
        return [chinese(v) for v in value]
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, str):
        return {"sphere center TCP": "球心工具参考点",
                "top face of a fixed finite box, not an infinite plane": "固定有限薄板的水平上表面",
                "None; complete directory copied byte-for-byte": "上游文件未修改，原始目录逐字节复制"}.get(value, value)
    return value


def print_summary(summary):
    visible = {k: v for k, v in summary.items() if not k.endswith("sha256")}
    print(json.dumps(chinese(visible), ensure_ascii=False, indent=2))


def save_csv(frame, path):
    frame.to_csv(path, index=False)  # 保留英文数据接口，已有测试和分析脚本仍可读取。
    names = dict(LABELS)
    for column in frame.columns:
        if column.startswith(("qpos_", "qvel_", "ctrl_")):
            kind, index = column.split("_")
            i = int(index)
            if kind == "ctrl":
                names[column] = f"关节{i+1}目标位置_rad" if i < 7 else "夹爪控制输入_0至255"
            else:
                target = f"关节{i+1}" if i < 7 else f"夹爪关节{i-6}"
                unit = "rad" if i < 7 else "米"
                names[column] = target + ("位置_" if kind == "qpos" else "速度_") + unit + ("每秒" if kind == "qvel" else "")
    translated = frame.rename(columns=names).copy()
    for column in translated.select_dtypes(include="bool").columns:
        translated[column] = translated[column].map({True: "是", False: "否"})
    translated.to_csv(path.with_name(path.stem + "_中文.csv"), index=False, encoding="utf-8-sig")


def configure_chinese_font():
    from matplotlib import font_manager, rcParams
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Noto Sans CJK JP"):
        if name in available:
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return True
    print("未发现中文绘图字体，图片暂用英文；中文表格与报告不受影响。Ubuntu 可安装 fonts-noto-cjk。")
    return False
