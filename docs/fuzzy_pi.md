# 在线模糊自适应 PI

`controllers/fuzzy_pi.py` 的 `FuzzyAdaptivePI` 继承数值 `ForcePI`。每个 5 ms 控制周期使用测量误差和滤波后的误差变化率，在线计算比例、积分增益，然后实际调用同一 PI 输出限制与抗积分饱和逻辑。它是规则型在线增益调度，不是参数辨识、自学习或最优参数搜索。

## 输入与隶属函数

`e = 目标力 - 测量力`，单位 N；`de = (e-e_previous)/dt`，单位 N/s。误差变化率经过时间常数 0.05 s 的一阶低通，接管时以当前误差初始化历史值，避免首次求导突跳。

归一化输入为 `clip(e/1 N, -1, 1)` 和 `clip(filtered_de/5 N/s, -1, 1)`。对每个归一化值 x 使用三个语言值：

- N：`max(-x, 0)`。
- Z：`1-abs(x)`。
- P：`max(x, 0)`。

隶属度非负且和为 1。规则权重为两个输入隶属度的乘积；零阶 Sugeno 输出为 9 个常数后件的加权平均。

## 规则表

行是误差 N/Z/P，列是误差变化率 N/Z/P。数字为基准增益倍率，不是直接累加到上一周期的增益。

| Kp 倍率 | de=N | de=Z | de=P |
|---|---:|---:|---:|
| e=N | 1.3 | 1.1 | 0.8 |
| e=Z | 0.9 | 1.0 | 0.9 |
| e=P | 0.8 | 1.1 | 1.3 |

| Ki 倍率 | de=N | de=Z | de=P |
|---|---:|---:|---:|
| e=N | 1.2 | 1.1 | 0.65 |
| e=Z | 0.7 | 1.1 | 0.7 |
| e=P | 0.65 | 1.1 | 1.2 |

误差继续扩大时增加修正积极性，接近目标时减小积分倍率；误差变化很快且接近零时抑制积分。该解释是规则设计意图，效果以实际闭环数据为准，规则本身不是稳定性证明。

## 更新、状态与限制

加权倍率乘以配置中的基准 Kp/Ki，经过上下界裁剪得到目标增益。实际增益用 `alpha=dt/(0.1 s+dt)` 平滑跟随目标，并在同一周期用于 PI 计算。

默认基准 `Kp=2e-5 m/N`、`Ki=3e-4 m/(N*s)`；范围分别为 `[1e-5,3e-5]`、`[1.5e-4,4.5e-4]`。没有饱和或状态限制时，平滑更新为范围内的凸组合，不累计无限漂移。

PI 的积分状态存储已乘 Ki 的位移贡献，因此改变 Ki 后不整体缩放已有积分。当 Kp 变化时，积分贡献补偿 `(Kp_old-Kp_new)*e`，避免单纯改变 Kp 造成瞬时比例跳变。补偿仍受积分限制；限幅生效时不承诺完全无突跳，并记录 `fuzzy_compensation_limited`。

既有积分贡献上限 1 mm、法向位置修正上限 0.8 mm、修正速度上限 0.5 mm/s 保留。执行器/关节、穿透、真实力和接触检查也保留。接触接管与传感器测量通道复用原 M1。

## 配置和记录

| 字段 | 默认值 | 含义 |
|---|---|---|
| `pi_algorithm` | fixed | fixed/fuzzy；旧配置默认固定 PI |
| `fuzzy_error_scale_n` | 1 | 误差归一化尺度 |
| `fuzzy_rate_scale_n_s` | 5 | 变化率归一化尺度 |
| `fuzzy_derivative_tau_s` | 0.05 | 误差变化率滤波时间常数 |
| `fuzzy_gain_tau_s` | 0.1 | 增益平滑时间常数 |
| `fuzzy_kp_bounds` | 基准 Kp 的 0.5–1.5 倍 | Kp 最小/最大值，必须包含基准值 |
| `fuzzy_ki_bounds` | 基准 Ki 的 0.5–1.5 倍 | Ki 最小/最大值，必须包含基准值 |

`control.csv` 中 `pi_kp_m_per_n`、`pi_ki_m_per_ns` 是实际使用增益；`fuzzy_kp_target`、`fuzzy_ki_target` 是平滑前目标。另有归一化输入、滤波后变化率、补偿限幅标志。`pi_p_m` 必须等于实际 Kp 乘控制器测量误差，不能仅记录一条未参与计算的增益曲线。

## 运行与验收

```powershell
python -m experiments.m1_force_pi --mode pi --algorithm fuzzy
python -m experiments.m1_force_pi --mode pi --algorithm fuzzy --sensor-config configs/sensor_example.json
python -m experiments.m1_fuzzy_comparison
python -m pytest tests/test_fuzzy_pi.py -q --basetemp=t_fuzzy
```

对照入口在 `results/F/run_XXX/` 保存 6 个条件、12 次运行：无扰动、三随机种子组合测量扰动、两组额外接触参数。每个条件两种算法使用相同初态、目标、传感器配置/种子、PI 基准、限幅和原 M1 阈值。接管前的真实力要求逐值一致；改变算法只在接管后生效。

记录真实力全程/末秒 RMSE、稳定时间、峰值与初始偏差、接触、输出限幅、实际增益范围、`online_gains.png` 和 `force_errors.png`。`contact_solref` 的时间常数变化未经有效刚度标定，不等于已经完成具有物理 N/m 值的多刚度实验。

数值测试包含规则分区、实际增益参与计算、增益变化补偿、饱和/反向、非法输入、锁定增益后逐值复现固定 PI，以及实际 Panda 仿真中的在线更新与限幅。有限的对照不能证明普遍优于固定 PI；对照也没有排除“重新选择固定增益即可取得相近效果”的可能，不能把所有改善都归因于在线变化。
