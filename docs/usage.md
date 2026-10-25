# 操作与环境

## 准备

从仓库根目录运行，建议使用 Python 3.12。已有匹配环境时直接使用其 `python.exe`。独立安装如下：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux 使用 `.venv/bin/python` 替代 Windows 路径。不要直接运行 `experiments/xxx.py`，采用 `python -m experiments.xxx` 以正确解析包导入。

## 基础与 PI

```powershell
python -m experiments.m0_01_model_inspection
python -m experiments.m0_02_contact_scene
python -m experiments.m0_03_contact_force
python -m experiments.m1_force_pi --mode both
```

`both` 从相同 home 状态依次运行固定位置 Baseline 和固定 PI；`--mode pi` 只运行 PI。M1 默认目标 5 N，物理步长 1 ms、控制周期 5 ms，12 秒仿真，8 秒接管；实际值以保存的完整配置为准。

```powershell
python -m experiments.m1_force_pi --mode pi --sensor-config configs/sensor_example.json
```

传感器配置文件作为字典叠加到实验配置；示例只填写 `sensor_*` 字段，不应覆盖目标、限位或验收阈值来制造通过。示例参数只是仿真假设，并不保证满足原先无扰动 PI 的阈值。失败时读取原始数据和 `summary.json`，保留失败记录。

先将示例复制为独立配置，逐项启用噪声、偏置、延迟、低通，再做组合。关闭扰动使用噪声/偏置/延迟为 0、低通为 `null`；省略所有字段也是默认直通。每次运行重建传感器，随机种子相同则噪声序列可重复。

```powershell
python -m experiments.m1_sensor_study
```

该入口在 `results/S/run_XXX/` 运行 9 组固定 PI：无扰动、仅噪声、仅偏置、仅延迟、仅低通、三种子组合和大延迟。每组参数写入 `study_config.json`，子目录保留原实验的通过/失败判定；顶层 `comparison.csv`、`study_summary.json`、`sensor_study.png` 汇总真值指标。批量命令正常结束表示完成记录，不等于每组通过，应检查汇总通过数和失败原因。提前终止的实验不生成虚假的末秒统计。参数不是硬件规格，不是安全延迟上限。

## 其余现有实验

```powershell
python -m experiments.m0_04_depth_comparison
python -m experiments.m0_05_contact_exit
python -m experiments.m2_admittance --case reference
python -m experiments.m3_impedance --case probe
python -m experiments.c1_wipe
```

M2/M3 的 `--case all` 运行全部既有工况，先通过单组与基础检查，再进行批量实验。M2 当前评价受力退让与虚拟弹簧平衡，不应解释为 5 N 力跟踪。

需要交互观察时在实验后增加 `--viewer`：空格开始，结束后关闭窗口；`--speed` 只改变显示播放倍率；`--snapshot` 保存末态图，需要可用的 OpenGL。默认无窗口运行不要求人工点击。

## 结果与排错

M1 结果位于 `results/M1/run_XXX/`，其中 `baseline/run_001/` 和 `pi/run_001/` 分别保存两组原始数据。顶层包含完整配置、`comparison.csv`、`summary.json`、`run.log`、`force_comparison.png`、`position_command.png` 和运行 PI 时的 `sensor_force.png`。

子目录包含真实接触 CSV、控制 CSV、关节状态、原始接触点、派生场景及源码/资源摘要。`*_中文` 是展示副本，原始字段保持英文供分析程序读取。M1 顶层 `summary.json` 的 `passed` 为最终任务判定；子实验成功不代表两组对照已完成。

Windows 输出目录过深可能使 MuJoCo 无法打开网格。优先使用仓库内短路径，如 `results/t` 作为测试临时目录，避免外部深层临时目录。不要通过删除模型网格、改变碰撞几何来绕过加载失败。

本仓库不含 Panda 真机通信或驱动。仿真脚本不能直接当作真机启动命令；真机迁移需另行确认接口、设备状态、控制权限、限位、停止路径与现场授权。
