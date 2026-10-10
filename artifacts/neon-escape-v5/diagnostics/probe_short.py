from pathlib import Path
exec(open(Path(__file__).with_name('probe_sustained.py')).read().split("for hop in ['ropehop_centered'")[0])
for dur in [.25,.40,.60,.8,1.0]:
 for hop in ['ropehop_centered','ropehop_contact']:
  mujoco.mj_copyData(d,m,base);bank=ParkourPolicyBank(paths|{'victory_hop':str(rdir/f'policies/{hop}.onnx')});r=DuckRuntime(m,d,bank,prefix='duck/');r.last_action=(d.ctrl[r.act_ids]-DEFAULT_POSE).astype(np.float32);finish=None;start=float(d.time);minup=1.;dep={};f=np.zeros(6);flight=[];air=None;hz=0.;samples=[]
  for i in range(500):
   t=float(d.time);r.set_command();r.head_override=None
   if finish is None:
    r.active_policy='run';r.command[:3]=lane_command(r,0,.5)
    if r.trunk_pos()[0]>=7.95:finish=t
   else:
    q=t-finish;r.active_policy='run' if q<.8 else 'stand'
    if (1.8<q<1.8+dur) or (4.<q<4.+dur):r.active_policy='victory_hop';r.hop_phase_source=lambda:2*math.pi*3.1*(d.time-finish)-math.pi
   d.ctrl[m.actuator('portal/motor').id]=0. if r.trunk_pos()[0]<=7.85 else -.6
   r.step()
   for j in range(100):
    mujoco.mj_step(m,d);up=float(d.xmat[r.trunk_body_id,8]);minup=min(minup,up);hz=max(hz,float(r.trunk_pos()[2]));support=foot_support(m,d)
    if finish and d.time-finish>1.8:
     if not support and air is None:air=float(d.time)
     if support and air is not None:flight.append(float(d.time)-air);air=None
    for ci,c in enumerate(d.contact):
     names=[m.body(m.geom_bodyid[g]).name for g in [c.geom1,c.geom2]];du=sum(n.startswith('duck/') for n in names)
     if not du:continue
     mujoco.mj_contactForce(m,d,ci,f)
     if f[0]>1e-7:dep['self' if du==2 else 'floor']=max(dep.get('self' if du==2 else 'floor',0),max(0,-float(c.dist)))
  print(json.dumps(dict(hop=hop,duration=dur,minup=minup,up=float(d.xmat[r.trunk_body_id,8]),z=float(r.trunk_pos()[2]),xy=r.trunk_pos()[:2].tolist(),depth_mm={k:v*1000 for k,v in dep.items()},height=hz,flights=[v for v in flight if v>.025])),flush=True)
