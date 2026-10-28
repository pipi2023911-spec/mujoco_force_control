# 接口说明

## 力与坐标

`common.contact_feedback.contact_force(model, data, tool_id, surface_id, normal_world)` 只统计工具球与指定薄板的接触，调用 `mj_contactForce` 获取接触局部力，再按接触坐标系和作用/反作用方向转换为工具所受世界系合力。它不是以穿透量乘常数构造的假力。

`normal_force_n` 是工具受力沿接触面单位法向的投影（N）；默认法向为世界 +Z，不取绝对值。位置 PI 的正误差 `目标力 - 测量力` 对应沿负法向压入。工具位置单位 m、关节单位 rad、时间单位 s。

## 测量通道

`ForceSensor(config, control_dt).step(true_force, time_s)` 返回 `(filtered_force, details)`。每个控制周期调用一次，顺序为：控制时刻真值 → 整数周期延迟 → 偏置与独立高斯白噪声 → 后向欧拉一阶低通。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `sensor_noise_std_n` | 0 | 白噪声标准差，有限非负数，N |
| `sensor_bias_n` | 0 | 有限常值偏置，N |
| `sensor_delay_steps` | 0 | 非负整数，单位为控制周期，不是物理步 |
| `sensor_lpf_cutoff_hz` | null | 正截止频率，Hz；null 关闭低通 |
| `sensor_seed` | 0 | 非负整数随机种子 |

低通系数 `alpha = dt / (dt + 1/(2*pi*fc))`；首个输出取首个原始测量值，不预置虚假的零力。无低通时直接赋值以保证默认精确直通。延迟队列未填满时保持最早样本，不假装已有过去数据。默认 5 ms 控制周期下，延迟 2 步对应稳态延迟 10 ms。低通还有额外频率相关相位滞后，不能用队列延迟字段代表总延迟。

| `control.csv` 字段 | 含义 |
|---|---|
| `time_s`、`sensor_sample_time_s` | 本次控制/测量更新时刻 |
| `sensor_source_time_s` | 延迟通道取用的真值源时刻 |
| `sensor_actual_delay_s` | 采样时刻减源时刻，启动阶段可能小于配置延迟 |
| `sensor_delay_ready` | 是否填满延迟队列 |
| `sensor_true_force_n` | 当前控制时刻的真实法向力 |
| `sensor_delayed_force_n` | 延迟后、尚未添加偏置/噪声的真值 |
| `sensor_raw_force_n` | 添加偏置/噪声后、低通前的测量 |
| `sensor_filtered_force_n` | 送给反馈控制器的力 |
| `measured_force_n` | M1 控制器实际使用的力，等于滤波输出 |
| `feedback_time_s` | 反馈字典当前时刻，不代表延迟信号源时刻 |

模型允许含噪测量为负，不通过截零掩盖噪声。源时间戳只描述延迟前样本；低通输出综合多时刻信号。

## 控制与记录

公共位置循环向反馈参考函数传入状态副本，仅将副本的 `normal_force_n` 替换为测量值；几何与接触数量仍来自当前仿真。真实力在每个物理步记录并用于力、速度、穿透与意外接触检查。M3 为独立力矩循环，不使用该传感器入口。

`ForcePI.step(target_force, measured_force, dt)` 输出法向压入修正（m）和比例/积分/限幅状态。`integral` 存储已经乘积分增益的贡献（m）。力不足时增加压入；力过大时退出。接管时积分补偿比例项；幅值或变化率饱和且误差继续推动饱和时暂停积分。

`pi_algorithm=fuzzy` 使用相同输入输出签名的 `FuzzyAdaptivePI`。`pi_kp_m_per_n`、`pi_ki_m_per_ns` 为当前周期实际生效值；默认 fixed 的两个字段保持常值。模糊输入、规则、增益范围及积分补偿见[算法说明](fuzzy_pi.md)。

`contact_force.csv` 的 `normal_force_n` 和 `force_error_n` 始终是真值及目标减真值；`control.csv` 的 `force_error_n` 为目标减测量值。启用扰动后两者不同，不应混用。`joint_state.csv` 保留物理位置/速度，`raw_contacts.csv` 保留坐标变换重构所需的接触局部力和坐标系。

每组子实验保存 `config.json`、`summary.json`、版本与源码/资源 SHA256、`run.log`。异常会保留已有样本和失败时刻，不补造完整轨迹。停止仿真表示结束数值积分，不是经过验证的真机安全停止。

## C1路径与离线接触诊断

`wipe_displacement(t, config)`输出X方向位移（m）和速度（m/s），采用五次曲线；`wipe_start_s`为起点，`wipe_leg_duration_s`为单程时间，`wipe_length_m`为单程长度。主配置为10秒开始、8秒单程、10 mm，26秒回到起点，27秒结束。`c1_wipe.run`输出`c1_metrics.json`与最终`summary.json`；基础循环检查通过后仍须通过连续接触与实际往返判定。原3秒配置失败时抛出`ValueError`，已有数据保留。

`c1_contact_feasibility.run(output_root=None)`返回新建结果目录，默认`results/V/run_XXX/`。`local_state`按局部TCP位姿、切向速度与零关节加速度构造诊断状态；逆动力学结果通过原七轴位置执行器回代，手指残差单独补齐，不向机械臂额外施加广义力。`force_envelope.csv`记录速度、几何压入量、正逆法向力差、正向加速度、力矩余量及位置命令范围；`summary.json`的`passed`仅代表这组诊断回代检查通过。离线诊断不修改主控制器或场景配置，也不代表闭环稳定，详见[C1诊断定义](c1_contact_feasibility.md)。

`c1_sensor_study.run(output_root=None)`返回九工况结果目录，默认`results/CS/run_XXX/`。`analyze(frame, config)`以物理采样计算运动段指标：`完整时长`与`完整运动段`分开记录，`接触丢失累计_s`和`最长连续丢失_s`单位为秒，`首次丢失时刻_s`为仿真绝对时间、无丢失时为null。未完成运动段不生成全段RMSE与往返指标。`任务验收通过`来自原C1最终摘要，不根据汇总重新放宽或增加力误差门槛；见[测量实验接口与判定](c1_sensor_study.md)。

`c1_fuzzy_comparison.run(output_root=None)`返回固定/模糊配对目录，默认`results/CF/run_XXX/`，各工况的`cN/p/run_001/`为固定PI、`cN/f/run_001/`为模糊PI。复用上述 `analyze`；额外核对实际比例项等于 `Kp×测量误差`、增益范围、积分/输出/变化率限制及接管前一致性。`在线增益实际变化`只表示实际Kp、Ki都变化，不代表性能优势；原始测量与真值的字段定义不变。
