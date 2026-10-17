import json

import mujoco
import numpy as np
import pandas as pd
import pytest

from common.torque_scene import build_torque_scene
from controllers.impedance import impedance_torque
from experiments.m3_impedance import load_config, run, run_trial


def test_actual_torque_mapping_and_gripper_preserved(tmp_path):
    c=load_config(); m,d,_,qa,va,acts=build_torque_scene(c,tmp_path)
    assert (m.nq,m.nv,m.nu)==(9,9,8)
    d.ctrl[acts]=np.linspace(-1,1,7)
    for offset in [0,.001]:
        d.qpos[qa]+=offset; mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.qfrc_actuator[va],d.ctrl[acts],atol=1e-12)
    remaining=list(set(range(m.nu))-set(acts))
    assert len(remaining)==1 and m.actuator_trntype[remaining[0]]==mujoco.mjtTrn.mjTRN_TENDON
    assert m.actuator_biastype[remaining[0]]==mujoco.mjtBias.mjBIAS_AFFINE
    assert d.ctrl[remaining[0]]==255


def test_dynamic_null_projection_and_torque_clamp():
    rng=np.random.default_rng(3); a=rng.normal(size=(7,7)); mass=a@a.T+np.eye(7)
    jac=rng.normal(size=(6,7)); c=load_config()
    tau,details=impedance_torque(np.zeros(3),np.zeros(3),np.zeros(3),jac,np.zeros(7),mass,np.zeros(7),np.ones(7),c)
    assert details['null_leak']<1e-10
    assert np.linalg.norm(tau)>0
    # 笛卡尔误差为零时，力矩仍可在动态零空间内调节冗余姿态。
    np.testing.assert_allclose(jac@np.linalg.solve(mass,tau),0,atol=1e-10)
    tau,details=impedance_torque(np.ones(3)*100,np.zeros(3),np.zeros(3),jac,np.zeros(7),mass,np.zeros(7),np.zeros(7),c)
    assert details['limited'] and np.all(np.abs(tau)<=c['torque_limits_nm'])


def test_bias_probe_and_real_impedance(tmp_path):
    c=load_config()
    probe=run_trial(c,tmp_path/'probe',probe=True)
    assert json.loads((probe/'summary.json').read_text(encoding='utf-8'))['passed']
    output=run(c,case='reference',output_root=tmp_path/'contact')
    folder=output/'reference/run_001'
    s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert s['passed'] and s['state_samples']==12001 and s['command_samples']==2400
    assert 4<s['hold_mean_force_n']<6 and s['最大力矩映射误差_Nm']<1e-12
    commands=pd.read_csv(folder/'control.csv')
    for j in range(1,8):
        np.testing.assert_allclose(commands[f'指令_{j}_Nm'],commands[f'任务_{j}_Nm']+commands[f'偏置_{j}_Nm']+commands[f'零空间_{j}_Nm'],atol=1e-10)
    assert not commands['力矩受限'].any()


def test_invalid_gain_keeps_failed_run(tmp_path):
    c=load_config(); c['rotation_stiffness_nm_rad']=-1
    with pytest.raises(ValueError,match='有限正数'):
        run_trial(c,tmp_path)
    s=json.loads((tmp_path/'run_001/summary.json').read_text(encoding='utf-8'))
    assert not s['passed']
