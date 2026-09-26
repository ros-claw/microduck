"""Measure existing motor skills before making obstacle-clearance claims."""
import os,sys,json,argparse,math
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco,numpy as np
from microduck_lab.game.world import build_world

def run(skill,seconds=7):
    m,d,r=build_world();samples=[]
    for i in range(int(seconds*50)):
        t=float(d.time);r.command[:]=0
        if t>=2 and t<4:
            r.active_policy='run' if skill=='dodge' else skill
            if skill=='run':r.command[0]=.25
            if skill=='dodge':r.command[1]=.15
            if skill=='jump':
                r.hop_phase_source=lambda:2*math.pi*2.8*(d.time-2)-math.pi
                r.hop_target=np.array([.15*(t-2),0,0])
        else:r.active_policy='stand'
        r.step();mujoco.mj_step(m,d,100)
        samples.append(dict(t=float(d.time),pos=r.trunk_pos().tolist(),upright=float(d.xmat[r.trunk_body_id,8]),gyro=r.base_ang_vel().tolist(),policy=r.active_policy))
    out=dict(skill=skill,final=samples[-1],min_upright=min(s['upright'] for s in samples),max_z=max(s['pos'][2] for s in samples),samples=samples)
    (ROOT/f'artifacts/neon-escape/probe-{skill}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(skill,out['final'],'min_up',out['min_upright'],'max_z',out['max_z'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('skill',choices=['run','dodge','roulade','jump']);a=p.parse_args();run(a.skill)
