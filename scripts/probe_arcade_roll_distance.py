import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.game.world import build_world
from microduck_lab.game.chunks import Chunk
import mujoco,numpy as np
out=[]
for distance in [.31,.35,.37,.39,.42]:
 height=.285
 m,d,r=build_world([Chunk('bar','high_bar',distance,0,height=height,half_width=.85)]);hits=0;depth=0.;samples=[]
 owner=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '') for b in m.geom_bodyid]
 for i in range(300):
  r.active_policy='roulade' if 2<=d.time<4 else 'stand';r.command[:]=0;r.step()
  for j in range(100):
   mujoco.mj_step(m,d)
   for c in d.contact:
    names=[owner[int(g)] for g in (c.geom1,c.geom2)]
    if 'bar' in names and any(n.startswith('duck/') for n in names) and c.efc_address>=0:hits+=1;depth=max(depth,-float(c.dist))
  samples.append(dict(t=float(d.time),x=float(r.trunk_pos()[0]),up=float(d.xmat[r.trunk_body_id,8]),pitch=float(np.arctan2(-d.xmat[r.trunk_body_id,6],d.xmat[r.trunk_body_id,0]))))
 rotation=float(np.ptp(np.unwrap([s['pitch'] for s in samples])))
 res=dict(distance=distance,height=height,hits=hits,max_penetration_m=depth,final=samples[-1],pitch_excursion_rad=rotation);out.append(res);print(res,flush=True)
open(ROOT/'artifacts/neon-escape/roll-distance-probe.json','w').write(json.dumps(out,indent=2)+'\n')
