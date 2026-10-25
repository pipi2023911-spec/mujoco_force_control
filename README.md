# Panda MuJoCo 接触力控制

使用 Franka Panda 模型开展位置接触、固定增益 PI、导纳与笛卡尔阻抗仿真实验。当前开发主线是复核 PI 基线、引入力测量扰动，并实现和验证在线模糊自适应 PI。

代码存在、数值测试通过、物理仿真通过和真机验证属于不同完成等级。本项目没有真机控制接口；已实现规则型在线模糊自适应 PI，并记录实际生效增益与物理对照。它不等于自动寻找最优参数或已证明全工况稳定。

## 控制链与职责

```text
MuJoCo接触约束 → 工具受力真值 → 延迟 → 偏置/白噪声 → 低通 → 控制器
                      ↓                                ↓
               物理检查与性能评价              法向位置修正 → IK → 位置执行器
```

M1 的 PI 使用目标力减测量力，力不足时沿接触面负法向增加压入。M2 为导纳位置参考，M3 使用独立的力矩执行器与阻抗循环。测量模型接入公共位置反馈循环（M1、M2、C1）；M3 不经过该通道。上游模型资源不原地修改，实验场景生成在结果目录。

## 环境与入口

Python 3.12，依赖版本见 [requirements.txt](requirements.txt)。默认无窗口运行也可导出曲线；交互窗口和截图需要图形环境。

- [操作与环境](docs/usage.md)：安装、实验入口、窗口操作与输出位置。
- [接口说明](docs/interfaces.md)：力方向、单位、时间戳、测量模型、控制与日志。
- [测试说明](docs/testing.md)：数值测试、物理仿真、失败记录与验收边界。
- [力控实验方案](docs/force_control.md)：基线、传感器、自适应 PI 和鲁棒性实验。
- [在线模糊 PI](docs/fuzzy_pi.md)：隶属函数、规则、在线更新、状态衔接与验证。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.m1_force_pi --mode both
```

所有命令从仓库根目录执行。已有依赖完整的 Python 解释器可以直接替代上述解释器，无需重复安装。

## 全部文件说明

| 文件 | 职责 |
|---|---|
| `README.md` | 项目范围、文件导航与验证边界 |
| `.gitignore` | 排除环境、缓存及完整实验结果，保留文档可见性 |
| `.gitattributes` | 固定源码文本换行，保留上游模型资源字节与哈希 |
| `requirements.txt` | 直接依赖与固定版本 |
| `requirements-lock-windows.txt` | Windows 环境完整依赖版本记录 |
| `common/chinese_output.py` | 英文字段的中文展示副本、CSV/JSON 与字体处理 |
| `common/contact_feedback.py` | 筛选工具与板面的接触，转换坐标系并合成工具真实受力 |
| `common/force_sensor.py` | 法向力测量延迟、偏置、白噪声、低通与源时间戳 |
| `common/contact_scene.py` | 模型加载、球形工具与薄板场景、home 状态和几何量 |
| `common/torque_scene.py` | 从位置场景派生独立的力矩执行器场景 |
| `common/robot_state.py` | 关节/执行器映射报告、状态有限性与限位检查 |
| `common/kinematics.py` | 平滑接近参考、姿态误差、阻尼 IK |
| `common/experiment_view.py` | 交互窗口、空格开始、播放倍率与显示/物理数据隔离 |
| `controllers/force_pi.py` | 固定 PI、积分贡献、无突跳初始化、抗积分饱和、幅值/速度限制 |
| `controllers/fuzzy_pi.py` | Sugeno 模糊推理、在线 Kp/Ki 更新、比例变化补偿，复用 PI 限制 |
| `controllers/admittance.py` | 数值导纳状态更新、位移/速度限幅；不是目标力跟踪实验 |
| `controllers/impedance.py` | 笛卡尔阻抗、零空间与力矩限幅 |
| `experiments/m0_01_model_inspection.py` | 模型接口审核、home 保持、结果目录与 JSON 工具 |
| `experiments/m0_02_contact_scene.py` | 工具和接触板放置、几何一致性与场景显示 |
| `experiments/m0_03_contact_force.py` | 公共位置控制物理循环、真实力、测量通道与失败数据记录 |
| `experiments/m0_04_depth_comparison.py` | 多压入参考实验、力重构与趋势分析 |
| `experiments/m0_05_contact_exit.py` | 接近、保持和退出接触实验 |
| `experiments/m1_force_pi.py` | 固定位置/固定 PI 对照、真值跟踪指标与测量通道曲线 |
| `experiments/m1_fuzzy_comparison.py` | 固定/模糊 PI 的六条件十二次仿真、实际增益与真实力对照 |
| `experiments/m1_sensor_study.py` | 固定 PI 单因素、三种子组合、大延迟实验和真实力汇总 |
| `experiments/m2_admittance.py` | 多组虚拟质量/阻尼/刚度导纳对照与平衡评价 |
| `experiments/m3_impedance.py` | 独立力矩循环、偏置补偿验证和阻抗对照 |
| `experiments/c1_wipe.py` | 切向往返路径与法向 PI、路径/接触评价 |
| `configs/m0_01.json` | 模型来源、home、关节与基础仿真参数 |
| `configs/m0_02.json` | 工具、薄板、初始间隙、接触与显示配置 |
| `configs/m0_03.json` | 接近轨迹、IK、真实力/速度/穿透检查与统计窗口 |
| `configs/m0_04.json` | 三组压入参考和趋势判定 |
| `configs/m0_05.json` | 接触退出阶段与检查参数 |
| `configs/m1.json` | 5 N 固定 PI、接管时间、限幅与验收阈值 |
| `configs/sensor_example.json` | 假设的测量扰动组合示例，不是硬件标定数据 |
| `configs/m2.json` | 导纳基准及参数对照组 |
| `configs/m3.json` | 阻抗刚度/阻尼、力矩限制与实验工况 |
| `configs/c1_wipe.json` | 往返路径、接触保持与轨迹验收 |
| `scripts/run_m0_01.ps1` | Windows 模型基础实验入口 |
| `scripts/run_m0_01.sh` | Linux 模型基础实验入口 |
| `tests/test_force_sensor.py` | 无 MuJoCo 的测量数值测试 |
| `tests/test_fuzzy_pi.py` | 模糊数值、限幅、固定增益退化与实际在线增益物理验证 |
| `tests/test_sensor_study.py` | 提前终止不生成末秒指标、失败轨迹保留不放宽验收 |
| `tests/test_m0_01.py` | 模型、状态和 home 保持测试 |
| `tests/test_m0_02.py` | 场景几何与接触放置测试 |
| `tests/test_m0_03.py` | 真实接触坐标变换、力重构、基础运行及故障留证 |
| `tests/test_m0_04.py` | 压入量实验及趋势检查 |
| `tests/test_m0_05.py` | 接触退出与力消失检查 |
| `tests/test_m1.py` | PI 限幅、接管、Baseline 对照、测量入口与真值保留 |
| `tests/test_m2.py` | 导纳数值行为与物理对照 |
| `tests/test_m3.py` | 阻抗数值行为、执行器与物理实验 |
| `tests/test_c1_wipe.py` | 往返轨迹、接触和端点检查 |
| `tests/test_presentation.py` | 中文副本和窗口隔离测试 |
| `docs/usage.md` | 完整运行步骤 |
| `docs/interfaces.md` | 输入、输出、时序与单位 |
| `docs/testing.md` | 测试分层、证据位置与限制 |
| `docs/force_control.md` | 实验顺序、评价定义与开发边界 |
| `docs/fuzzy_pi.md` | 在线模糊 PI 算法、参数、运行和证据要求 |
| `docs/evidence/README.md` | 固定 PI、测量接入与既有 C1 失败的实测记录 |
| `docs/evidence/pi_sensor_metrics.json` | 统计、源码摘要和原始归档校验值 |
| `docs/evidence/pi_sensor_raw_2026-10-24.zip` | 选定实验的原始数据、配置、日志和曲线 |
| `docs/evidence/fixed_pi.png`、`docs/evidence/sensor_force.png` | 原始仿真数据导出的对照图 |
| `docs/evidence/sensor-study-2026-10-25/README.md` | 9 组测量扰动实验、偏置和延迟结论边界 |
| `docs/evidence/sensor-study-2026-10-25/comparison.csv`、`study_summary.json`、`study_config.json` | 每组真值指标、三种子统计与全部实验参数 |
| `docs/evidence/sensor-study-2026-10-25/raw_data.zip`、`manifest.json`、`sensor_study.png` | 9 组原始证据、哈希清单与汇总图 |
| `docs/evidence/fuzzy-pi-2026-10-25/README.md` | 在线模糊 PI 闭环验收、配对结果与不利结果 |
| `docs/evidence/fuzzy-pi-2026-10-25/comparison.csv`、`study_config.json`、`study_summary.json` | 十二次固定/模糊对照统计、配置和完成状态 |
| `docs/evidence/fuzzy-pi-2026-10-25/verification.json`、`raw_data.zip` | 实际增益、限幅、源码和噪声配对核对及原始归档 |
| `docs/evidence/fuzzy-pi-2026-10-25/online_gains.png`、`force_errors.png` | 实际在线增益和真实力误差曲线 |
| `assets/MODEL_SOURCE.json` | 上游仓库、提交和许可证记录 |

上游资源位于 `assets/mujoco_menagerie_panda/`：`panda.xml`、`panda_nohand.xml`、`hand.xml` 分别为机械臂/无手臂/手部模型；`scene.xml` 为基础场景；`mjx_panda.xml`、`mjx_panda_nohand.xml`、`mjx_hand.xml`、`mjx_scene.xml`、`mjx_single_cube.xml` 为上游 MJX 资源，不是本项目 Python 力控入口。`assets/*.obj`、`assets/*.stl` 为显示/碰撞网格；`LICENSE` 保留 Apache-2.0 许可证。

## 验证与证据边界

实验数据默认保存在 `results/`，不把大批运行文件自动提交到 Git。每次运行的配置、版本、源码/模型摘要、原始 CSV、失败信息和曲线应成套保留。数值和结论只对应其实际运行条件；无噪声、零延迟的低误差不能代表真实传感器或真机性能。

在线模糊增益与固定/模糊对照已进入实际仿真。多环境有效刚度标定、更多目标与扰动、独立调参消融仍需继续验证。AI 辅助代码的核心逻辑需由使用者理解、修改并通过实验验证。

[固定 PI 与测量通道实测记录](docs/evidence/README.md)包含最初的默认兼容性验证和原始数据。

[9 组固定 PI 测量扰动实验](docs/evidence/sensor-study-2026-10-25/README.md)保留单因素、三种子组合及偏置问题。

[在线模糊 PI 闭环验收](docs/evidence/fuzzy-pi-2026-10-25/README.md)包含十二次配对、实际 Kp/Ki 曲线与不利结果。当前完整回归为 **64 PASS、1 个既有 C1 连续接触 FAIL**，不能称全仓库测试全绿或所有条件都优于固定 PI。
