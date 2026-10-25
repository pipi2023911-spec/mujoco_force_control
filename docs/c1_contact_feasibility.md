# C1接触、速度与失败诊断

## 实验定义

主任务保持5 N目标、原接触模型、10 mm去程加10 mm返回。按用户选择，单程采用经实际闭环验证的8秒，运动区间为10–26秒，27秒结束。固定PI、积分限制、输出限制、执行器限制及连续接触/位置阈值均沿用原设置；默认测量无噪声、无延迟。

原3秒单程保存于`configs/c1_wipe_legacy_3s.json`，运动10–16秒、17秒结束。它仍失败，属于Debug反例；8秒通过代表采用了新的速度要求，不代表已经解决原3秒要求。`c1_compliant.json`另外改变接触阻抗，仍独立保存。

## 排查思路

原工况在滑动速度接近峰值时出现抬离与法向力波动。冻结运动段PI修正、增加2 Hz低通、降低Ki、改变积分方案均未让原任务通过；只改变这些项不能解释为完成修复。每条分支的真实失败数据和脚本见[证据与Debug材料](evidence/c1-speed-matched-2026-10-25/README.md)。

MuJoCo采用软接触和正则化约束，摩擦与法向响应可能耦合；官方说明滑动开始可以诱发法向运动。这为局部诊断提供建模依据，并不直接证明本例的全部失败原因。[MuJoCo 3.3.7计算模型](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html)

## 局部诊断方法

`experiments/c1_contact_feasibility.py`使用原模型，在路径中点附近构造固定姿态、指定切向TCP速度、零关节加速度的局部快照。扫描7个速度、5个压入量，共35组；以`mj_inverse`得到接触力和所需广义力，再把七轴所需力矩换算为原位置执行器命令，调用`mj_forward`交叉核对。

机械臂不使用额外`qfrc_applied`；手指残差使用该字段补齐全模型。检查正逆法向力差、正向加速度、位置命令和力矩范围，避免只把任意施力逆解作为可实现证据。它仍是离线快照，不是实际控制器执行。

近零压入取10纳米，仅是有限扫描中最浅的一点。该点的法向力随速度升高；在6.25 mm/s时约11.52 N，已经高于5 N目标。由零速与6.25 mm/s两点线性估算，5 N对应局部临界速度约2.713 mm/s。该值不是所有姿态/加速度下的全局速度上限。

五次位移参考的峰值速度为`1.875 * L / T`。10 mm路径、3秒单程对应6.25 mm/s；8秒对应2.34375 mm/s。局部估算给出约6.912秒的参考时间尺度，随后必须运行实际闭环验证，不能单靠公式判定通过。6秒单程仍发生接触丢失，8秒单程在本次原模型实验中保持连续接触。

## 运行和输出

从仓库根目录，使用已安装依赖的Python：

```powershell
python -m experiments.c1_wipe
python -m experiments.c1_wipe --config configs/c1_wipe_legacy_3s.json
python -m experiments.c1_contact_feasibility
python -m pytest tests -q --basetemp=t_all
```

第二条预期验收失败并保留数据。重跑前先另存已有实验；pytest会清理指定临时目录。

| 入口/输出 | 含义 |
|---|---|
| `c1_wipe.run(config, output_root, ...)` | 真正推进MuJoCo闭环，默认结果在`results/C1_Wipe/` |
| `c1_metrics.json` | 运动区间真实力RMSE、接触比例、真实往返位移与端点误差 |
| `contact_force.csv`、`control.csv` | 物理真值与已执行参考，保留各自时间戳 |
| `raw_contacts.csv`、`joint_state.csv` | 接触重构与机械臂状态原始数据 |
| `c1_contact_feasibility.run(output_root=None)` | 离线诊断，默认结果在`results/V/`，返回结果目录 |
| `force_envelope.csv` | 速度(m/s)、压入量(m)、法向力(N)、加速度及命令范围检查 |
| 诊断`summary.json` | 回代检查、源码哈希、局部速度估算与适用边界 |

## 验收边界

8秒主基线通过不代表原3秒工况通过，也不代表所有速度、传感器扰动或模型都稳定。同初态测试与CLI轨迹逐值一致只验证确定性重复。没有真机接口，未验证真实传感器、真机闭环或现场停止路径。失败记录与成功记录共同保留，作为定位问题和理解控制代码的材料。
