"""Seeded mechanical hazards. Their motors act through contacts, never pose writes."""
from dataclasses import dataclass
import math
import mujoco
import numpy as np

@dataclass(frozen=True)
class Chunk:
    name:str
    kind:str
    x:float
    y:float
    phase:float=0.
    height:float=.26
    half_width:float=.22
    def add_to(self,spec):
        geom=dict(contype=1,conaffinity=1,priority=1,solref=[.002,1.],solimp=[.95,.99,.001,.5,2.],friction=[.8,.005,.0001])
        color=[1.,.08,.22,1.]
        body=spec.worldbody.add_body(name=self.name,pos=[self.x,self.y,0])
        if self.kind=='high_bar':
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,-self.half_width,self.height,0,self.half_width,self.height],size=[.009],rgba=color,**geom);return
        if self.kind=='boulder':
            body.pos=[self.x,self.y,.25];body.add_freejoint(name=self.name+'_free')
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.25],mass=.8,rgba=[.65,.12,1,1],**geom)
            # Physical visual stripes attached to the same rotating body, no contacts.
            for axis in range(3):
                q=[1,0,0,0] if axis==2 else ([.7071,.7071,0,0] if axis==0 else [.7071,0,.7071,0])
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.251,.006],quat=q,rgba=[.9,.4,1,1],contype=0,conaffinity=0,mass=0)
            return
        hinge=self.kind in ('sweeper','pendulum')
        axis=[0,0,1] if self.kind!='pendulum' else [1,0,0]
        if self.kind=='moving_wall':axis=[0,1,0]
        body.add_joint(name=self.name+'_joint',type=mujoco.mjtJoint.mjJNT_HINGE if hinge else mujoco.mjtJoint.mjJNT_SLIDE,axis=axis,damping=.03)
        if self.kind=='gate':
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.035,.19,.19],pos=[0,0,.19],mass=.15,rgba=color,**geom)
        elif self.kind=='drop_crate':
            body.pos=[self.x,self.y,.7]
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.095,.095,.095],mass=.08,rgba=[1,.5,.06,1],**geom)
        elif self.kind=='moving_wall':
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.05,.12,.16],pos=[0,0,.16],mass=.12,rgba=[.1,.9,1,1],**geom)
        elif self.kind=='sweeper':
            body.pos=[self.x,self.y,.065]
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,-.27,0,0,.27,0],size=[.014],mass=.05,rgba=[1,.1,.3,1],**geom)
        elif self.kind=='pendulum':
            body.pos=[self.x,self.y,.58]
            body.add_geom(name=self.name+'_g',type=mujoco.mjtGeom.mjGEOM_CAPSULE,fromto=[0,0,0,0,0,-.42],size=[.045],mass=.10,rgba=[.1,.9,1,1],**geom)
        else:raise ValueError(self.kind)
        a=spec.add_actuator(name=self.name+'_motor',target=self.name+'_joint',trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gainprm[0]=15.;a.biasprm[1]=-15.;a.biasprm[2]=-.8;a.biastype=mujoco.mjtBias.mjBIAS_AFFINE
        a.forcelimited=True;a.forcerange=[-4,4]

def generate(seed):
    rng=np.random.default_rng(seed)
    kinds=['gate','sweeper','drop_crate','moving_wall','pendulum'];rng.shuffle(kinds)
    chunks=[Chunk('bar_00','high_bar',1.,0.,height=float(rng.uniform(.285,.295)),half_width=.85)]
    for i,kind in enumerate(kinds):
        chunks.append(Chunk(f'{kind}_{i+1:02d}',kind,2.2+1.15*i,float(rng.choice([-.48,0,.48])),float(rng.uniform(0,2*math.pi))))
    chunks.append(Chunk('boss_ball','boulder',-1.1,0))
    return chunks

class HazardDriver:
    def __init__(self,m,d,chunks):
        self.chunks=chunks;self.released=set()
        self.motors={c.name:int(m.actuator(c.name+'_motor').id) for c in chunks if c.kind not in ('high_bar','boulder')}
    def step(self,m,d,duck_x):
        for c in self.chunks:
            if c.kind=='high_bar':continue
            if c.kind=='boulder':
                # Rolling force at center of mass. Contact friction produces rotation.
                bid=int(m.body(c.name).id);qadr=int(m.jnt_dofadr[m.body_jntadr[bid]])
                d.xfrc_applied[bid,:]=0
                if d.time>15:
                    d.xfrc_applied[bid,0]=float(np.clip(.8*(.28-d.qvel[qadr]),-.2,.35))
                continue
            aid=self.motors[c.name]
            if c.kind=='gate':target=.28*(1+math.sin(1.3*d.time+c.phase))
            elif c.kind=='moving_wall':target=.16*math.sin(1.7*d.time+c.phase)
            elif c.kind=='sweeper':target=2.2*d.time+c.phase
            elif c.kind=='pendulum':target=.6*math.sin(2*d.time+c.phase)
            else:
                target=0.
                if duck_x>c.x-.9:
                    self.released.add(c.name);m.actuator_gainprm[aid,0]=0;m.actuator_biasprm[aid,:]=0
            d.ctrl[aid]=target
