# 固定 PI 与测量通道实验记录

日期：2026-10-24。未修改基线提交：`211056d3bf859ddbce039ec1db3b0b7f16a89b3f`。修改工作树的实际 Python 源码 SHA256 保存在 `pi_sensor_metrics.json` 和原始运行 summary 中，不能把基线提交当作含传感器代码的提交。

环境：Windows 11、Python 3.12.14、MuJoCo 3.3.7、NumPy 2.2.6、Pandas 2.2.3、Matplotlib 3.10.3、pytest 8.3.5。复用既有虚拟环境，没有重新安装依赖。

## 实际运行

从修改工作副本执行：

```powershell
python -m unittest discover -s tests -p test_force_sensor.py -v
python -m pytest tests/test_m1.py -q --basetemp=results/t
python -m pytest tests -q --basetemp=t_all
```

第一项 7 项通过（最初使用已有 NumPy 2.3.5；随后在匹配依赖环境完整 pytest 中再次覆盖）；第二项 5 项通过；第三项 52 项通过、1 项失败。`results/t` 需要事先存在其父目录 `results`。早期较深的外部/内部输出目录曾导致网格加载失败；缩短测试路径后加载错误消失，未修改模型或测试断言。

在干净基线工作副本执行 `python -m pytest tests/test_m1.py tests/test_c1_wipe.py -q --basetemp=results/b`，结果为 7 项通过、1 项失败。该失败同样为 C1 连续接触验收。

## 实测结果

下表 RMSE 为真实法向力在最后 1 秒窗口内相对 5 N 目标的误差；稳定时间定义见力控方案。

| 工况 | 末秒 RMSE（N） | 稳定时间（s） | 结论 |
|---|---:|---:|---|
| 固定位置 Baseline | 0.964711911 | 未进入稳定带 | 对照运行完成 |
| 固定 PI，无测量扰动 | 0.003493340 | 1.221 | 原 M1 阈值通过 |
| 固定 PI，小幅组合测量扰动 | 0.010731823 | 1.131 | 该单组原 M1 阈值通过 |
| 固定 PI，示例配置 CLI | 0.006420872 | 1.246 | 命令入口与该单组原 M1 阈值通过 |

组合扰动参数为噪声标准差 0.01 N、偏置 0.01 N、延迟 2 个控制步（10 ms）、低通 20 Hz、种子 42。它不是 `sensor_example.json` 的完整工况，也不是多种子鲁棒性统计。单组稳定时间缩短不能证明测量扰动改善性能。

独立执行 `python -m experiments.m1_force_pi --mode pi --sensor-config configs/sensor_example.json` 验证 CLI；示例为噪声 0.05 N、偏置 0、延迟 2 步、20 Hz、种子 42。两组统计分别保存，不据此推断更大噪声效果更好。

未修改基线和默认直通版本的固定位置/固定 PI `contact_force.csv`、`control.csv` 原有字段、`joint_state.csv` 和 `raw_contacts.csv` 逐值完全一致。实际增益仍为固定值，不存在已完成的模糊 PI 对照。

## 保留的失败

C1 运动区间 10–16 秒的真实力 RMSE 为 1.333269045 N，接触保持率为 95.3333%，未通过连续接触验收。未修改基线与修改版本的 C1 指标字典完全一致。保留原有失败，不降低阈值、不标记跳过，也不把公共循环检查通过写成 C1 任务通过。

## 原始证据

- [统计与源码摘要](pi_sensor_metrics.json)：原始数值、兼容性结论与归档校验值。
- [原始数据归档](pi_sensor_raw_2026-10-24.zip)：修改版本的 M1 对照、组合扰动和 C1 失败数据；含完整真值/控制/原始接触 CSV、配置、场景、日志、摘要和图，`SHA256.json` 可检查内容完整性。关节状态全量数据及未修改版本的重复数据留在本地完整归档中，避免重复提交。
- 归档内 `test_logs/` 保存完整测试、基线回归、M1 专项和示例 CLI 的实际输出。
- [固定 PI 曲线](fixed_pi.png)、[测量通道曲线](sensor_force.png)。

这些记录是无窗口仿真。交互窗口、截图渲染、真实传感器、在线模糊 PI、多环境标定和真机均未据此验收。
