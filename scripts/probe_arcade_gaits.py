import sys,math,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.game.world import build_world
import mujoco,numpy as np
out=[]
for kind,param in [('run',.4),('run',.6),('dodge',.4),('jump',2.8),('jump',3.1),('jump',3.5)]:
 m,d,r=build_world();samples=[]
 for i in range(400):
  t=d.time;r.command[:]=0;r.active_policy= 'stand' if t<2 else ('run' if kind=='dodge' else kind)
  if t>=2:
   if kind=='run':r.command[0]=param
   if kind=='dodge':r.command[0]=.35;r.command[2]=np.clip(3*(math.atan2(.3-r.trunk_pos()[1],.5)-r.trunk_yaw()),-1.5,1.5)
   if kind=='jump':r.hop_phase_source=lambda:2*math.pi*param*(d.time-2)-math.pi
  r.step();mujoco.mj_step(m,d,100)
  samples.append(dict(t=float(d.time),pos=r.trunk_pos().tolist(),upright=float(d.xmat[r.trunk_body_id,8]),feet=[r.site_pos(f'{side}_foot').tolist() for side in ('left','right')]))
 summary=dict(kind=kind,param=param,final=samples[-1],min_up=min(s['upright'] for s in samples[100:]),max_z=max(s['pos'][2] for s in samples[100:]),max_min_foot_z=max(min(p[2] for p in s['feet']) for s in samples[100:]),samples=samples)
 out.append(summary);print({k:v for k,v in summary.items() if k!='samples'},flush=True)
open(ROOT/'artifacts/neon-escape/skill-sweep.json','w').write(json.dumps(out,indent=2)+'\n')
