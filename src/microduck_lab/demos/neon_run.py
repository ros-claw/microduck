"""A Jev-controlled physical arcade prototype. python -m microduck_lab.demos.neon_run"""
import argparse,dataclasses,json,time,hashlib,os
from pathlib import Path
import mujoco
import numpy as np
from ..game.world import build_world,REPO
from ..game.chunks import generate,HazardDriver
from ..game.state import observe,candidates,lane_clear
from ..game.skills import SkillRuntime
from ..game.scoring import GameAudit
from ..jev.client import JevClient
from ..jev.decision_loop import DecisionLoop

def baseline(state,legal):
    if 'ROULADE' in legal:return 'ROULADE'
    r=state['robot'];front=[h for h in state['hazards'] if .0<h['distance_m']<1.15 and abs(h['lateral_m']-r['y_m'])<h['half_width_m']+.12]
    if front and front[0]['kind']=='high_bar' and abs(r['y_m'])<.08:
        return 'KEEP_RUNNING' if 'KEEP_RUNNING' in legal else 'BRAKE'
    if front:
        for action in ('DODGE_LEFT','DODGE_RIGHT'):
            if action in legal:return action
    return 'KEEP_RUNNING' if 'KEEP_RUNNING' in legal else 'BRAKE'

def run(seed=0,seconds=60,brain='baseline',realtime=False,output=None,capture=False,deadline=1.5,confidence=.25):
    if not (0<=confidence<=1 and deadline>0 and seconds>0):raise ValueError('Invalid confidence, deadline or duration')
    protocol_files=['game/world.py','game/chunks.py','game/state.py','game/skills.py','game/scoring.py','jev/client.py','jev/decision_loop.py','demos/neon_run.py']
    protocol={name:hashlib.sha256((REPO/'src/microduck_lab'/name).read_bytes()).hexdigest() for name in protocol_files}
    chunks=generate(seed);m,d,duck=build_world(chunks);driver=HazardDriver(m,d,chunks);skill=SkillRuntime(duck);audit=GameAudit(m,chunks)
    loop=DecisionLoop(JevClient(timeout=3.),deadline,confidence) if brain=='jev' else None
    if loop and not realtime:raise ValueError('Live Jev requires --realtime so network delay is represented in physical time')
    path=Path(output or REPO/f'artifacts/neon-escape/{brain}-seed{seed}');path.mkdir(parents=True,exist_ok=True)
    wall_start=time.monotonic();last_request=-10.;frames=[];trace=[];fallbacks=0;termination='timeout';max_lag=0.
    try:
        for step in range(round(seconds*50)):
            t=float(d.time);state=observe(m,d,duck,chunks,skill.action,skill.target_lane);state['mission']['time_limit_s']=seconds;legal=candidates(state,skill.locked(t),t>=skill.roll_ready_at)
            if t>=2:
                if not skill.locked(t) and skill.action in ('KEEP_RUNNING','DODGE_LEFT','DODGE_RIGHT') and 'KEEP_RUNNING' not in legal:
                    skill.apply('BRAKE',t);fallbacks+=1
                selected=None
                if loop:
                    selected=loop.poll(legal,t,skill.locked(t))
                    if not skill.locked(t) and t-last_request>=.5:
                        if loop.request(state,legal,t):last_request=t
                elif not skill.locked(t) and t-last_request>=.2:selected=baseline(state,legal);last_request=t
                if selected is not None:
                    if selected=='ROULADE' and skill.apply(selected,t):audit.begin_roll(t)
                    elif selected!='ROULADE':skill.apply(selected,t)
            skill.step(t);driver.step(m,d,float(duck.trunk_pos()[0]))
            for sub in range(100):
                mujoco.mj_step(m,d);audit.physics(m,d,duck)
                if capture and sub%25==24:frames.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
            audit.checkpoints(duck)
            trace.append(dict(t=float(d.time),position=duck.trunk_pos().tolist(),upright=float(d.xmat[duck.trunk_body_id,8]),action=skill.action,lane=skill.target_lane,score=audit.result()['score'],hits=sorted(audit.hit)))
            if step%250==249:print(f'progress t={d.time:.1f}s x={duck.trunk_pos()[0]:.2f} action={skill.action} hits={len(audit.hit)}',flush=True)
            if duck.trunk_pos()[2]<-.1 or abs(duck.trunk_pos()[1])>1.:termination='fell_off_track';break
            if duck.trunk_pos()[0]>=8 and audit.support_s>.05 and duck.is_upright(.9):termination='finish';break
            if realtime:
                target=wall_start+float(d.time);lag=time.monotonic()-target;max_lag=max(max_lag,lag)
                if lag<0:time.sleep(-lag)
    finally:
        if loop:loop.close()
    passed=termination=='finish' and not audit.hit
    report=dict(protocol_sha256=protocol,mujoco_version=mujoco.__version__,policy_sha256={name:hashlib.sha256(Path(value).read_bytes()).hexdigest() for name,value in duck.bank.paths.items()},seed=seed,brain=brain,passed=passed,termination=termination,duration_sim_s=float(d.time),wall_seconds=time.monotonic()-wall_start,max_realtime_lag_s=max_lag,deadline_s=deadline,confidence_threshold=confidence,safety_brakes=fallbacks,physics=dict(timestep_s=float(m.opt.timestep),mocap_bodies=m.nmocap,robot_pose_writes_after_initialization=False,hazard_contact_solref_s=.002),chunks=[dataclasses.asdict(c) for c in chunks],audit=audit.result(),skill_events=skill.events,trace=trace,decisions=loop.records if loop else [])
    if capture:
        mujoco.mj_saveModel(m,str(path/'scene.mjb'),None)
        np.savez_compressed(path/'trajectory.npz',time=np.array([x[0] for x in frames]),qpos=np.array([x[1] for x in frames]),qvel=np.array([x[2] for x in frames]),ctrl=np.array([x[3] for x in frames]))
        report['capture']=dict(hz=200,files={n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ('scene.mjb','trajectory.npz')})
    (path/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('seed','brain','passed','termination','duration_sim_s','audit','safety_brakes')},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=0);p.add_argument('--seconds',type=float,default=60);p.add_argument('--brain',choices=['baseline','jev'],default='baseline');p.add_argument('--realtime',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--deadline',type=float,default=1.5);p.add_argument('--confidence',type=float,default=.25)
    a=p.parse_args();run(a.seed,a.seconds,a.brain,a.realtime,a.output,a.capture,a.deadline,a.confidence)
if __name__=='__main__':main()
