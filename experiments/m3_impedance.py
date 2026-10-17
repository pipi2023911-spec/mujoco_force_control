"""M3：真正关节力矩执行器、笛卡尔阻抗与刚度/阻尼对照。"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import mujoco
import numpy as np
import pandas as pd

from common.chinese_output import configure_chinese_font, save_csv
from common.contact_feedback import contact_force
from common.contact_scene import geometry_state
from common.experiment_view import ExperimentView
from common.kinematics import orientation_error
from common.robot_state import inspect_model, check_state
from common.torque_scene import build_torque_scene
from controllers.impedance import impedance_torque
from experiments.m0_01_model_inspection import ROOT, new_run, save_json
from experiments.m0_02_contact_scene import configure_camera
from experiments.m0_03_contact_force import load_config as load_base

NAMES = {"reference":"基准阻抗", "soft":"低刚度", "stiff":"高刚度", "low_damping":"低阻尼", "high_damping":"高阻尼"}


def load_config(path=Path("configs/m3.json")):
    c = json.loads((ROOT/path).read_text(encoding="utf-8"))
    return {**load_base(Path(c["base_config"])), **c}


def trajectory(t, initial, normal, c):
    u = np.clip((t-c["settle_duration_s"])/c["approach_duration_s"],0,1)
    blend = 10*u**3-15*u**4+6*u**5
    rate = (30*u**2-60*u**3+30*u**4)/c["approach_duration_s"]
    return initial-normal*c["approach_distance_m"]*blend, -normal*c["approach_distance_m"]*rate


def run_trial(c, parent, viewer=False, speed=1., snapshot=False, probe=False):
    output = new_run(parent)
    save_json(output/"config.json",c)
    summary = {"passed":False,"started_utc":datetime.now(timezone.utc).isoformat(),
               "python":platform.python_version(),"platform":platform.platform(),"viewer_requested":viewer,
               "versions":{n:importlib.metadata.version(n) for n in ["mujoco","numpy","pandas","matplotlib"]},
               "git_head_at_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "git_worktree_status":subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).splitlines(),
               "asset_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/c["robot_xml"]).parent.rglob("*") if p.is_file()},
               "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ["common","controllers","experiments"] for p in (ROOT/folder).glob("*.py")}}
    samples, commands, states, contacts = [],[],[],[]
    sim_time=0.
    with (output/"run.log").open("x",encoding="utf-8") as log:
        try:
            m,d,placement,qa,va,acts=build_torque_scene(c,output)
            save_json(output/"scene_report.json",placement)
            save_json(output/"model_report.json",inspect_model(m,d,c))
            summary["model_dimensions"]={"nq":m.nq,"nv":m.nv,"nu":m.nu,"njnt":m.njnt}
            summary["model_source"]=json.loads((ROOT/"assets/MODEL_SOURCE.json").read_text(encoding="utf-8"))
            summary["scene_sha256"]=hashlib.sha256((output/"panda_torque.xml").read_bytes()).hexdigest()
            qhome=d.qpos[qa].copy(); site=m.site("tool_tip").id
            p0=d.site_xpos[site].copy(); r0=d.site_xmat[site].reshape(3,3).copy()
            normal=np.array(c["normal_world"]); sphere=m.geom("tool_sphere").id; plate=m.geom("contact_plane").id
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv)); full_mass=np.zeros((m.nv,m.nv))
            dt=c["physics_dt_s"]; period=c["control_dt_s"]; duration=c["probe_duration_s"] if probe else c["duration_s"]
            substeps=round(period/dt); steps=round(duration/dt)
            if not np.isclose(substeps*dt,period) or not np.isclose(steps*dt,duration):
                raise ValueError("周期必须为物理步长的整数倍")
            if 1.875*c["approach_distance_m"]/c["approach_duration_s"]>c["max_reference_speed_m_s"]:
                raise ValueError("参考接近速度超限")
            def record():
                g=geometry_state(m,d,c); force=contact_force(m,d,sphere,plate,normal)
                lateral=np.linalg.norm((np.eye(3)-np.outer(normal,normal))@(d.site_xpos[site]-p0))
                angle=np.linalg.norm(orientation_error(d.site_xmat[site].reshape(3,3),r0))
                actual_error=float(np.max(np.abs(d.qfrc_actuator[va]-d.ctrl[acts])))
                samples.append({**g,"normal_force_n":force["normal_force_n"],"tool_contact_count":force["tool_contact_count"],
                                "other_contact_count":d.ncon-force["tool_contact_count"],"lateral_drift_m":float(lateral),
                                "orientation_error_rad":float(angle),"力矩映射误差_Nm":actual_error,
                                **{f"tool_force_{a}_n":float(force["force_world_n"][i]) for i,a in enumerate("xyz")}})
                states.append({"time_s":float(d.time),**{f"qpos_{i}":float(v) for i,v in enumerate(d.qpos)},
                               **{f"qvel_{i}":float(v) for i,v in enumerate(d.qvel)},
                               **{f"实际臂力矩_{i+1}_Nm":float(d.qfrc_actuator[v]) for i,v in enumerate(va)}})
                contacts.extend(force["points"]); check_state(m,d,c["joint_limit_tolerance"])
                if not np.isfinite(force["force_world_n"]).all() or actual_error>1e-10:
                    raise ValueError("力反馈数值或直接力矩映射异常")
                if d.ncon!=force["tool_contact_count"] or not g["inside_plate_footprint"]:
                    raise ValueError("出现意外接触或工具超出板面")
                if np.linalg.norm(force["force_world_n"])>c["max_measured_force_n"] or g["normal_gap_m"] < -c["max_penetration_m"]:
                    raise ValueError("接触力或穿透量超限")
                if lateral>c["max_lateral_drift_m"] or angle>c["max_orientation_error_rad"] or np.max(np.abs(d.qvel[va]))>c["max_arm_speed_rad_s"]:
                    raise ValueError("横向、姿态或速度超限")
            record()
            with ExperimentView(m,viewer,speed) as view:
                if viewer:
                    with view.handle.lock():
                        configure_camera(view.handle.cam,c)
                        view.handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=True
                    view.model.vis.map.force=c["viewer_force_length_m_per_n"]*m.stat.meanmass
                view.wait_start(d)
                for step in range(steps):
                    if step%substeps==0:
                        target,desired_v=trajectory(float(d.time),p0,normal,c)
                        mujoco.mj_jacSite(m,d,jp,jr,site); jac=np.vstack([jp[:,va],jr[:,va]])
                        mujoco.mj_fullM(m,full_mass,d.qM)
                        if probe:
                            target=p0.copy(); tau=d.qfrc_bias[va].copy()
                            details={"task":np.zeros(7),"null":np.zeros(7),"bias":tau.copy(),"raw":tau.copy(),"limited":False,"null_leak":0.}
                        else:
                            tau,details=impedance_torque(target-d.site_xpos[site],orientation_error(d.site_xmat[site].reshape(3,3),r0),
                                desired_v,jac,d.qvel[va],full_mass[np.ix_(va,va)],d.qfrc_bias[va],qhome-d.qpos[qa],c)
                        d.ctrl[acts]=tau
                        commands.append({"time_s":float(d.time),"力矩受限":details["limited"],"零空间泄漏范数":details["null_leak"],
                                         **{f"target_{a}_m":float(target[i]) for i,a in enumerate("xyz")},
                                         **{f"{name}_{i+1}_Nm":float(v) for name,array in [("指令",tau),("任务",details["task"]),("偏置",details["bias"]),("零空间",details["null"]),("限幅前",details["raw"])] for i,v in enumerate(array)}})
                    mujoco.mj_step(m,d); mujoco.mj_forward(m,d); sim_time=float(d.time); record()
                    if not np.isclose(sim_time,(step+1)*dt,atol=1e-10,rtol=0):
                        raise ValueError("仿真时间异常重置")
                    if (step+1)%substeps==0:
                        view.update(d)
                        log.write(f"仿真时间_秒={sim_time:.6f} 法向力_N={samples[-1]['normal_force_n']:.6f}\n")
                    if viewer and (step+1)%round(c["display_interval_s"]/dt)==0:
                        print(f"仿真{sim_time:.2f}秒｜力矩阻抗｜法向力{samples[-1]['normal_force_n']:.3f}N",flush=True)
                view.wait_end(d)
            frame=pd.DataFrame(samples); cmd=pd.DataFrame(commands)
            hold=frame[frame.time_s>duration-c["hold_metrics_window_s"]]
            detected=frame[frame.normal_force_n>c["contact_detection_force_n"]]
            onset=None if detected.empty else float(detected.time_s.iloc[0])
            transient=frame if onset is None else frame[(frame.time_s>=onset)&(frame.time_s<=onset+c["contact_transient_window_s"])]
            summary.update(simulated_duration_s=sim_time,state_samples=len(frame),command_samples=len(cmd),
                           peak_force_n=float(frame.normal_force_n.max()),hold_mean_force_n=float(hold.normal_force_n.mean()),
                           hold_std_force_n=float(hold.normal_force_n.std(ddof=0)),max_lateral_drift_m=float(frame.lateral_drift_m.max()),
                           max_orientation_error_rad=float(frame.orientation_error_rad.max()),warning_counts=d.warning.number.tolist(),
                           力矩限幅样本数=int(cmd["力矩受限"].sum()),最大力矩映射误差_Nm=float(frame["力矩映射误差_Nm"].max()),
                           最大零空间泄漏范数=float(cmd["零空间泄漏范数"].max()),末态关节漂移_rad=float(np.max(np.abs(d.qpos[qa]-qhome))))
            summary.update(first_contact_time_s=onset,初次接触窗口峰值力_N=float(transient.normal_force_n.max()))
            if probe:
                if summary["末态关节漂移_rad"]>c["probe_max_joint_drift_rad"] or frame.ncon.max()!=0:
                    raise ValueError("偏置补偿基础保持验证失败")
            elif (hold.tool_contact_count==0).any() or summary["hold_mean_force_n"]<c["minimum_hold_mean_force_n"] or summary["hold_std_force_n"]>c["steady_force_std_limit_n"] or summary["力矩限幅样本数"]:
                raise ValueError("末段接触稳定性或无饱和对照验收失败")
            if np.max(np.abs(d.qvel[va]))>c["max_final_arm_speed_rad_s"]:
                raise ValueError("末态速度未收敛")
            if snapshot:
                from PIL import Image
                camera=mujoco.MjvCamera(); configure_camera(camera,c)
                with mujoco.Renderer(m,height=c["render_height"],width=c["render_width"]) as renderer:
                    renderer.update_scene(d,camera=camera)
                    Image.fromarray(renderer.render()).save(output/"contact_scene.png")
            summary["passed"]=True
        except Exception as exc:
            summary.update(error=f"{type(exc).__name__}: {exc}",failed_at_sim_time_s=sim_time)
            log.write(summary["error"]+"\n"); raise
        finally:
            for name,rows in [("contact_force",samples),("control",commands),("joint_state",states),("raw_contacts",contacts)]:
                if rows: save_csv(pd.DataFrame(rows),output/f"{name}.csv")
            save_json(output/"summary.json",summary)
            print(f"结果：{output}；验收：{'通过' if summary['passed'] else '未通过'}",flush=True)
    return output


def run(c, viewer=False, speed=1., snapshot=False, case="all", output_root=None):
    output=new_run(output_root or ROOT/"results"/"M3")
    save_json(output/"config.json",{**c,"selected_case":case})
    report={"passed":False,"已完成工况":[]}; rows=[]; frames={}
    try:
        cases=list(c["cases"]) if case=="all" else [case]
        for name in cases:
            if name!="probe" and name not in c["cases"]: raise ValueError("未知阻抗工况")
            params={**c,**c["cases"].get(name,{})}
            print(f"开始 {NAMES.get(name,'偏置补偿验证')}",flush=True)
            folder=run_trial(params,output/name,viewer,speed,snapshot,probe=name=="probe")
            s=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            frames[name]=pd.read_csv(folder/"contact_force.csv")
            rows.append({"工况":NAMES.get(name,"偏置补偿"),"法向刚度_N每m":params["translation_stiffness_n_m"][2],
                         "法向阻尼_Ns每m":params["translation_damping_ns_m"][2],"末段平均力_N":s["hold_mean_force_n"],
                         "峰值力_N":s["peak_force_n"],"末段力标准差_N":s["hold_std_force_n"],
                         "首次接触时间_秒":s["first_contact_time_s"],"初次接触窗口峰值力_N":s["初次接触窗口峰值力_N"],
                         "最大横向偏移_mm":s["max_lateral_drift_m"]*1000,"最大姿态误差_rad":s["max_orientation_error_rad"],
                         "力矩限幅样本数":s["力矩限幅样本数"],"数据目录":folder.relative_to(output).as_posix()})
            save_csv(pd.DataFrame(rows),output/"comparison.csv"); report["已完成工况"].append(name)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zh=configure_chinese_font(); fig,ax=plt.subplots(figsize=(c["plot_width_in"],c["plot_height_in"]))
        for name,f in frames.items(): ax.plot(f.time_s,f.normal_force_n,label=NAMES.get(name,name) if zh else name)
        ax.set(xlabel="仿真时间（秒）" if zh else "Time (s)",ylabel="真实法向力（N）" if zh else "Normal force (N)",title="M3 笛卡尔阻抗" if zh else "M3 Cartesian impedance")
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/"force_comparison.png",dpi=c["plot_dpi"]); plt.close(fig)
        report.update(passed=True,对照结果=rows)
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"; raise
    finally:
        save_json(output/"summary.json",report)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/m3.json"))
    parser.add_argument("--case",choices=["all","probe",*NAMES],default="all",help="probe仅验证2秒偏置补偿")
    parser.add_argument("--viewer",action="store_true",help="空格开始，结束后关窗进入下一组")
    parser.add_argument("--speed",type=float,default=1.)
    parser.add_argument("--snapshot",action="store_true")
    args=parser.parse_args()
    run(load_config(args.config),args.viewer,args.speed,args.snapshot,args.case)
