"""Single physical Microduck and actuator-driven arcade obstacles. SI units."""
from pathlib import Path
import os
import mujoco
import numpy as np
from ..sim.runtime import DuckRuntime,PolicyBank,apply_current_limit

REPO=Path(__file__).resolve().parents[3]
ASSETS=Path(os.environ.get('MICRODUCK_ROOT',REPO.parent))

def build_world(chunks=()):
    spec=mujoco.MjSpec()
    spec.option.timestep=.0002
    spec.option.solver=mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations=40
    spec.visual.global_.offwidth=1920;spec.visual.global_.offheight=1080
    spec.visual.headlight.ambient=[.35,.35,.4];spec.visual.headlight.diffuse=[.7,.7,.7]
    spec.worldbody.add_light(pos=[0,0,4],dir=[0,0,-1],type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_BOX,size=[12,1.0,.04],pos=[10,0,-.04],rgba=[.045,.065,.10,1],friction=[1.0,.005,.0001])
    # Emissive lane markings are paint, deliberately non-colliding.
    for side,color in [(-1,[1,.06,.4,1]),(1,[.02,.8,1,1])]:
        mat=spec.add_material(name=f'neon{side}');mat.rgba=color;mat.emission=.8
        spec.worldbody.add_geom(name=f'lane_edge{side}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[3.8,side*.86,.002],size=[4.8,.008,.001],material=mat.name,contype=0,conaffinity=0)
    for x in np.arange(-.5,8.5,.4):
        for y in (-.24,.24):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[float(x),y,.001],size=[.075,.003,.0005],rgba=[.18,.25,.35,1],contype=0,conaffinity=0)
    for x in range(-1,10):
        for y in (-1.5,1.5):
            height=.3+.15*((x+3)%4)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,height/2],size=[.18,.16,height/2],rgba=[.06,.08,.16,1],contype=0,conaffinity=0)
    spec.worldbody.add_geom(name='finish_paint',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[8,0,.002],size=[.035,.85,.001],rgba=[.1,1,.55,1],contype=0,conaffinity=0)
    robot=ASSETS/'microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml'
    spec.attach(mujoco.MjSpec.from_file(str(robot)),prefix='duck/',frame=spec.worldbody.add_frame())
    for chunk in chunks:chunk.add_to(spec)
    m=spec.compile();d=mujoco.MjData(m)
    paths={n:str(ASSETS/'microduck/policies'/p) for n,p in {'stand':'alpha_stand.onnx','run':'alpha_walking.onnx','roulade':'roulade.onnx'}.items()}
    paths['jump']=str(REPO/'policies/ropehop_contact.onnx')
    bank=PolicyBank(paths)
    duck=DuckRuntime(m,d,bank,prefix='duck/')
    apply_current_limit(m,'duck/')
    d.qpos[duck.joint_qpos_idx]=duck.default_pose
    d.ctrl[duck.act_ids]=duck.default_pose
    mujoco.mj_forward(m,d)
    return m,d,duck
