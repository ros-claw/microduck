"""Physical feasibility probe. Static duo is NOT moving-rope entry or relay."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.demos.honest_skip import run_honest_classic_skip
from microduck_lab.sim.composer import DuckSpec
from microduck_lab.sim.skip_metrics import PhysicalSkipAudit

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seconds',type=float,default=20);p.add_argument('--seed',type=int,default=0)
p.add_argument('--dx',type=float,default=.075);p.add_argument('--dy',type=float,default=0.)
p.add_argument('--sky-y',type=float,default=0.)
p.add_argument('--jitter',type=float,default=.02)
p.add_argument('--span',type=float,default=.448);p.add_argument('--length',type=float,default=.58)
p.add_argument('--mode',choices=['duo','waiting','entry'],default='duo')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
audits={name:PhysicalSkipAudit(name) for name in ('sky','graphite')}
contacts=[];positions=[];ctx={};steps=0

def setup(m,d,info):
 m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
 for audit in audits.values():audit.setup(m,d,info)
 ctx['owners']=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '').split('/')[0] for b in m.geom_bodyid]

def physics(m,d,info,hopping):
 global steps
 for audit in audits.values():audit(m,d,info,hopping)
 for c in d.contact:
  if c.efc_address>=0 and {ctx['owners'][int(c.geom1)],ctx['owners'][int(c.geom2)]}=={'sky','graphite'}:
   if not contacts or d.time-contacts[-1]>.02:contacts.append(float(d.time))
 if steps%100==0:
  positions.append([float(d.time),*[float(x) for name in ('sky','graphite') for x in d.body(name+'/trunk_base').xpos]])
 steps+=1

def policy(m,d,info,rt):
 g=rt['graphite']
 if a.mode=='duo':g.active_policy='jump'
 elif a.mode=='entry' and d.time>11:
  if g.trunk_pos()[1]<-.02:
   g.active_policy='walk';g.set_command(twist=(.30,0.,0.))
  else:g.active_policy='jump';g.set_command()

kw=dict(spawn_jitter=a.jitter,seconds=a.seconds,seed=a.seed,render=False,hop_onnx=ROOT/'policies/ropehop_contact.onnx',turner_onnx=ROOT/'policies/turner_rope.onnx',
 rope_contacts='full',legacy_rope_offset=False,connect_timeconst=.004,rope_floor_timeconst=.0004,physics_dt=.0002,hop_start_delay=0,
 rope_joint_type='ball',rope_radius=.0015,rope_length=a.length,jumper_y=0,settle_seconds=0,rope_initial_phase=math.pi/2,rope_velocity_limit=0,
 rope_pass_time_fn=lambda:audits['sky'].last_underfoot_crossing,max_turn_hz=3.1,model_setup=setup,physics_observer=physics,policy_observer=policy,
 ducks=[DuckSpec('lavender',(-a.span/2,0),0),DuckSpec('cream',(a.span/2,0),math.pi),DuckSpec('sky',(-a.dx,a.sky_y),math.pi/2),
 DuckSpec('graphite',(a.dx,a.dy if a.mode=='duo' else -.4),math.pi/2)])
start=time.monotonic();error=None
try:run_honest_classic_skip(None,**kw)
except (RuntimeError,ValueError) as exc:error=str(exc)
result=dict(kind='static_duo_feasibility' if a.mode=='duo' else a.mode,relay_passed=False,config=vars(a)|{'output':str(a.output)},
 elapsed_s=time.monotonic()-start,error=error,inter_jumper_contact_times=contacts,positions=positions,jumpers={n:au.scorer.result() for n,au in audits.items()})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({n:(r['clean_skips'],r['full_revolutions']) for n,r in result['jumpers'].items()}), 'inter-contact samples',len(contacts),'error',error,flush=True)
