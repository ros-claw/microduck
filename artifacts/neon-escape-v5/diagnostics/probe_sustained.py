import json,math
from pathlib import Path
import numpy as np,mujoco
from microduck_lab.sim.runtime import DuckRuntime,DEFAULT_POSE
from microduck_lab.parkour.policies import ParkourPolicyBank
from microduck_lab.parkour.skills import lane_command
from microduck_lab.parkour.world import foot_support
rdir=Path.cwd();src=rdir/'artifacts/neon-escape-v4/film-402';u=dict(np.load(src/'inputs.npz'));m=mujoco.MjModel.from_binary_path(str(src/'scene.mjb'));d=mujoco.MjData(m);d.qpos[:]=u['initial_qpos'];d.qvel[:]=u['initial_qvel'];mujoco.mj_forward(m,d)
for k,t in enumerate(u['time']):
 if t>=25.24:break
 d.ctrl[:]=u['ctrl'][k];d.eq_active[:]=u['eq_active'][k];d.xfrc_applied[:]=u['xfrc_applied'][k];mujoco.mj_step(m,d,100)
base=mujoco.MjData(m);mujoco.mj_copyData(base,m,d)
paths={'stand':str(rdir.parent/'microduck/policies/alpha_stand.onnx'),'run':str(rdir.parent/'microduck/policies/alpha_walking.onnx')}
for hop in ['ropehop_centered','ropehop_high','ropehop_contact']:
 for gesture in [False,True]:
  mujoco.mj_copyData(d,m,base);bank=ParkourPolicyBank(paths|{'victory_hop':str(rdir/f'policies/{hop}.onnx')});r=DuckRuntime(m,d,bank,prefix='duck/');r.last_action=(d.ctrl[r.act_ids]-DEFAULT_POSE).astype(np.float32);finish=None;start=float(d.time);samples=[];flights=[];air=None;minup=1.;dep={};f=np.zeros(6)
  for i in range(550):
   t=float(d.time);r.set_command();r.head_override=None
   if finish is None:
    r.active_policy='run';r.command[:3]=lane_command(r,.025*math.sin((t-start)*5),.5)
    if gesture:r.head_override=np.array([.349+.1*math.sin(t*8),.349,.18*math.sin(t*6),.10*math.sin(t*6)])
    if r.trunk_pos()[0]>=7.98:finish=t
   else:
    q=t-finish
    r.active_policy='run' if q<.6 else 'stand'
    if 1.2<q<5.2:r.active_policy='victory_hop';r.hop_phase_source=lambda:2*math.pi*2.8*(d.time-finish)-math.pi
    if gesture:r.head_override=np.array([.349+.14*math.sin(q*7),.349+.06*math.sin(q*7),.22*math.sin(q*4),.10*math.sin(q*7)])
   d.ctrl[m.actuator("portal/motor").id]=0. if r.trunk_pos()[0]<=7.85 else -.6
   r.step()
   for j in range(100):
    mujoco.mj_step(m,d);up=float(d.xmat[r.trunk_body_id,8]);minup=min(minup,up)
    for ci,c in enumerate(d.contact):
     names=[m.body(m.geom_bodyid[g]).name for g in [c.geom1,c.geom2]];du=sum(n.startswith('duck/') for n in names)
     if not du:continue
     mujoco.mj_contactForce(m,d,ci,f)
     if f[0]>1e-7:
      key='self' if du==2 else 'floor' if 'world' in names else 'hazard';dep[key]=max(dep.get(key,0),max(0,-float(c.dist)))
   if finish and d.time-finish>1.2:
    support=foot_support(m,d)
    if not support and air is None:air=float(d.time)
    if support and air is not None:
     flights.append(float(d.time)-air);air=None
   samples.append(dict(t=float(d.time),pos=r.trunk_pos().tolist(),up=float(d.xmat[r.trunk_body_id,8])))
  out=dict(hop=hop,gesture=gesture,finish=finish,final=samples[-1],minup=minup,max_z=max(s['pos'][2] for s in samples),flights=flights,depth_mm={k:v*1000 for k,v in dep.items()});print(json.dumps(out),flush=True)
