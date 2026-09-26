"""Read-only physics-step contact audit; never award an obstacle on a timer."""
import math
import mujoco
import numpy as np

class GameAudit:
    def __init__(self,m,chunks):
        self.names={c.name:c for c in chunks};self.passed=set();self.hit=set();self.contacts=[];self.max_penetration=0.;self.steps=0
        self.owners=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '') for b in m.geom_bodyid]
        self.floor=int(m.geom('floor').id);self.support_s=0.;self.rolls=[];self.roll=None;self.previous_pitch=None
    def begin_roll(self,t):self.roll=dict(start=t,rotation_rad=0.,hit=False);self.previous_pitch=None;self.roll_min_upright=1.
    def physics(self,m,d,duck):
        self.steps+=1;support=False
        for i,c in enumerate(d.contact):
            if c.efc_address<0:continue
            a,b=int(c.geom1),int(c.geom2);oa,ob=self.owners[a],self.owners[b]
            robot=oa.startswith('duck/') or ob.startswith('duck/')
            if not robot:continue
            if self.floor in (a,b) and ('ankle_' in oa or 'ankle_' in ob):support=True
            hazard=oa if oa in self.names else (ob if ob in self.names else None)
            if hazard:
                depth=max(0.,-float(c.dist));self.max_penetration=max(self.max_penetration,depth)
                if self.roll:self.roll['hit']=True
                if hazard not in self.hit:
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                    self.contacts.append(dict(t=float(d.time),hazard=hazard,body=ob if oa==hazard else oa,penetration_m=depth,normal_force_n=float(force[0])))
                self.hit.add(hazard)
        self.support_s=self.support_s+m.opt.timestep if support else 0.
        if self.roll is not None:
            mat=d.xmat[duck.trunk_body_id];self.roll_min_upright=min(self.roll_min_upright,float(mat[8]));angle=math.atan2(-mat[6],mat[0])
            if self.previous_pitch is not None:
                delta=math.atan2(math.sin(angle-self.previous_pitch),math.cos(angle-self.previous_pitch));self.roll['rotation_rad']+=delta
            self.previous_pitch=angle
            if d.time-self.roll['start']>=2.8:
                self.roll.update(end=float(d.time),upright=float(mat[8]),supported_s=self.support_s)
                self.roll['clean']=abs(self.roll['rotation_rad'])>5.7 and self.roll_min_upright<-.8 and mat[8]>.9 and self.support_s>.05 and not self.roll['hit']
                self.rolls.append(self.roll);self.roll=None
    def checkpoints(self,duck):
        x=duck.trunk_pos()[0]
        for name,c in self.names.items():
            if c.kind!='boulder' and x>c.x+.38:self.passed.add(name)
    def result(self):
        return dict(physics_steps=self.steps,cleared=sorted(self.passed-self.hit),hit=sorted(self.hit),contacts=self.contacts,max_hazard_penetration_m=self.max_penetration,rolls=self.rolls,score=100*len(self.passed-self.hit)+200*sum(r['clean'] for r in self.rolls))
