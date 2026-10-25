# 测试与证据

## 数值测试

```powershell
python -m unittest discover -s tests -p test_force_sensor.py -v
```

该组只需要 NumPy，不需要 MuJoCo。覆盖默认精确直通（含浮点消减与负零）、延迟启动/源时间戳、偏置、低通阶跃响应、噪声重复性与统计、非法参数及非法样本。通过代表数值测量模型符合这些测试，不代表闭环稳定。

## MuJoCo 集成

```powershell
python -m pytest tests/test_force_sensor.py tests/test_m1.py -q --basetemp=t
python -m pytest tests -q --basetemp=t_all
```

`--basetemp` 指定 pytest 专用目录：pytest 会清理该目录，不能指向需要保留的数据。保存证据后再重跑。Windows 深层路径下，较短的 `t`/`t_all` 通常比系统临时目录更可靠；如仍加载网格失败，将工作副本放到更短路径后重跑。

M1 测试检查：

- 固定 PI 抗积分饱和、幅值/速度约束、力方向和无接触拒绝接管。
- 实际 Panda Baseline/PI 仿真，接管前物理轨迹一致，控制反馈来自相同控制时刻的真值。
- 默认测量精确直通；扰动后 PI 使用滤波输出，延迟序列对应过去控制样本，真实物理日志未被测量覆盖。

公共位置循环也服务 M2/C1，因此完成该入口变更后应运行完整测试。M3 不接入该传感器，但现有阻抗测试保留。现有测试会实际推进 MuJoCo，并生成 CSV、场景和图像；不是编译检查或宏桩。

## 实验运行与失败

```powershell
python -m experiments.m1_force_pi --mode both
python -m experiments.m1_force_pi --mode pi --sensor-config configs/sensor_example.json
```

单元测试的组合扰动为噪声 0.01 N、偏置 0.01 N、延迟两控制步、20 Hz 低通、种子 42；示例配置噪声为 0.05 N、偏置 0，两者已分别运行并留证，不能混为同一工况或多种子鲁棒性实验。

异常测试应确认 `passed=false`、失败时刻/原因以及已有原始样本保留。力超限检查继续使用真实力；不以过滤测量值绕过限制。测试通过后仍需单因素和重复鲁棒性实验。

## 证据与限制

每轮记录实际命令、Python/依赖版本、源码状态、测试退出状态、测试数量、失败原因、原始数据和图像路径。运行记录不能用测试文件的存在代替。论文或简历引用数字前，确认对应配置、代码与原始数据可取回。

已运行的[固定 PI 与测量通道记录](evidence/README.md)为完整测试 52 项通过、1 项 C1 连续接触失败；干净原始提交也复现相同失败。不得据此声称整个仓库测试全绿。

[测量扰动实验记录](evidence/sensor-study-2026-10-25/README.md)含 9 组完整仿真，以及 2 项新增统计测试的运行证据。该阶段测试覆盖为 54 PASS、1 个既有 FAIL。批量入口会继续记录不达标工况，不能仅凭批量命令退出码判定控制全部通过。

在线模糊 PI 增加 10 项数值/物理测试，完整测试已实际重跑为 **64 PASS、1 个既有 C1 FAIL**；以上 54/1 为传感器阶段记录。[模糊 PI 验收](evidence/fuzzy-pi-2026-10-25/README.md)保存 12 次配对、CLI 和完整回归输出。固定/模糊对照使用同一验收阈值；种子 42 的模糊末秒误差较差，保留在报告中。

新增M2b、独立柔顺C1和局部拟合测试后的最新完整回归为 **71 PASS、1个原C1 FAIL**。[遗留项记录](evidence/compliance-followup-2026-10-25/README.md)明确新接触模型通过不代表旧模型已修复；原C1配置、阈值和失败断言均保留。

无窗口仿真通过不能代表交互图形窗口已人工验证，更不能代表真机。项目没有真机驱动，未进行真实传感器标定、机器人闭环运动或现场停止验证。
