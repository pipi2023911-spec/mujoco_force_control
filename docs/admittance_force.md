# 导纳目标力跟踪

`controllers/admittance.py` 使用 `M*a+D*v+K*x=F_ext-F_target`。x正方向顺着工具所受正法向力，外力过大时退让。`admittance_force_target_n` 默认0，原M2仍是受力退让；M2b显式设置5N目标。

刚度允许0，质量、阻尼、位移/速度限幅仍必须有限正数。半隐式积分稳定条件和限幅后的速度同步保留。`admittance_force_error_n` 为外力减目标力，与M1的误差符号相反；目标、误差、状态和限幅标志均记录。

默认目标0与锁定旧提交98eaba5的实际类比较10000步随机受力，输出及全部原有状态字段逐值一致，原M2物理测试亦通过。

```powershell
python -m experiments.m2b_admittance_force
python -m pytest tests/test_m2b.py tests/test_m2.py -q --basetemp=t_b
```

`configs/m2b.json` 通过专用加载器复用完整M2基础。结果位于 `results/B/run_XXX/`，包括配置、原始数据、comparison.csv、summary.json和admittance_force.png。

| 工况 | 验收 |
|---|---|
| K=0，D=1500/3000/6000 | 真值末秒RMSE≤0.1N、有稳定时间、力标准差≤0.05N、接触不丢失、无状态限幅、平衡残差≤0.1N |
| K=2000，D=3000 | 接触、未限幅、平衡残差和波动；另列目标力跟踪是否通过 |

平衡残差为 `F_ext-F_target-K*x`。K>0组的passed只代表弹簧平衡检查，不代表5N跟踪；同时记录“目标力跟踪通过”和“平衡与接触检查通过”。实际该组末秒均值5.213165N，目标跟踪FAIL，平衡PASS。

K=0趋于零误差需要存在可达稳定平衡且未饱和。增大阻尼不保证所有闭环条件更稳，也不保证有限时长误差更小。本组D=6000末秒误差反而较大，保留结果。详细指标及原始数据见[遗留项实测](evidence/compliance-followup-2026-10-25/README.md)。
