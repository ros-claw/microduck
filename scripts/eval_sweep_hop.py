"""CPU contact rehearsal on the training apparatus, NOT flexible-rope acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from microduck_lab.sim.composer import compose_world, DuckSpec
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime


def evaluate(policy, seed=0, seconds=12., frequency=3.1, feedback=True, video=None):
    robot_dir = ROOT.parent/'microduck_rl/src/mjlab_microduck/robot'
    sweep = mujoco.MjSpec.from_file(str(robot_dir/'hop_sweep.xml'))
    world = compose_world(robot_dir/'microduck/robot_allcollisions.xml',
        [DuckSpec('sky', (0, 0), math.pi/2, (.55,.78,.98,1))], playground=False,
        extra_specs=(('apparatus/', sweep),))
    m, d = world.model, world.data
    m.opt.timestep = .001
    m.opt.iterations = 100
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    bank = PolicyBank({'stand':str(ROOT.parent/'microduck/policies/alpha_stand.onnx'), 'hop':str(policy)})
    duck = DuckRuntime(m, d, bank, prefix='sky/')
    rng = np.random.default_rng(seed)
    d.qpos[duck.joint_qpos_idx] = duck.default_pose+rng.uniform(-.02,.02,14)
    joint = m.joint('apparatus/passive_sweep_hinge')
    qadr, vadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    actuator = m.actuator('apparatus/sweep_drive').id
    obstacle = m.geom('apparatus/sweep_collision').id
    floor = m.geom('floor').id
    feet = {m.geom('sky/'+s+'_foot_collision').id for s in ('left', 'right')}
    # Most non-foot collision meshes are unnamed in the source XML. Identify
    # ownership by their named bodies, otherwise head/torso hits disappear.
    robot_geoms = {g for g in range(m.ngeom) if (mujoco.mj_id2name(
        m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').startswith('sky/')}
    d.qpos[qadr] = math.pi
    mujoco.mj_forward(m, d)
    # Stand before the experiment; rotor is stationary overhead during settling.
    for _ in range(100):
        duck.step()
        for _ in range(20):
            mujoco.mj_step(m,d)
    d.qpos[qadr] = math.pi + rng.uniform(-.3,.3)
    d.qvel[vadr] = 2*math.pi*frequency
    d.ctrl[actuator] = 2*math.pi*frequency
    duck.active_policy = 'hop'
    duck.hop_phase_source = (lambda: d.qpos[qadr]) if feedback else (lambda: math.pi)
    origin = duck.trunk_pos()[:2]
    writer = renderer = None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        writer = imageio.get_writer(str(video),fps=25)
        renderer = mujoco.Renderer(m, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat = [0, 0, .22]
    camera.distance = .85
    camera.azimuth, camera.elevation = 125, -12
    cycles, traces = [], []
    current = dict(contact_steps=0, body_ground_steps=0, min_cos=1., max_drift=0., crossing=None, crossings={}, landed=False)
    support_time = 0.
    previous_q = float(d.qpos[qadr])
    previous_dy = {g:float(d.geom_xpos[obstacle,1]-d.geom_xpos[g,1]) for g in feet}
    max_drift = 0.
    try:
        for p in range(round(seconds*50)):
            duck.step()
            for _ in range(20):
                mujoco.mj_step(m,d)
                if np.any(d.warning.number) or not np.isfinite(d.qacc).all():
                    raise RuntimeError(f'invalid physics at {d.time}')
                grounded = set()
                touching_obstacle = False
                for c in d.contact:
                    if c.efc_address < 0:
                        continue
                    pair = {int(c.geom1), int(c.geom2)}
                    if obstacle in pair and pair & robot_geoms:
                        current['contact_steps'] += 1
                        touching_obstacle = True
                    if floor in pair:
                        grounded |= pair & robot_geoms
                current['body_ground_steps'] += bool(grounded-feet)
                current['min_cos'] = min(current['min_cos'], float(-duck.projected_gravity()[2]))
                drift = float(np.linalg.norm(duck.trunk_pos()[:2]-origin))
                max_drift = max(max_drift, drift)
                current['max_drift'] = max(current['max_drift'], drift)
                q = float(d.qpos[qadr])
                for foot in feet:
                    dy = float(d.geom_xpos[obstacle,1]-d.geom_xpos[foot,1])
                    if previous_dy[foot] < 0 <= dy and d.geom_xpos[obstacle,2] < .24:
                        sample = dict(t=d.time-2.,
                            sole_z=float(mujoco.mj_geomDistance(m,d,foot,floor,1.,None)),
                            obstacle_top=float(d.geom_xpos[obstacle,2]+m.geom_size[obstacle,0]),
                            within_span=bool(abs(d.geom_xpos[foot,0]-d.geom_xpos[obstacle,0]) < m.geom_size[obstacle,1]))
                        current['crossings'][mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,foot)] = sample
                    previous_dy[foot] = dy
                if math.floor(q/(2*math.pi)) > math.floor(previous_q/(2*math.pi)):
                    clearance = min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in feet)
                    current['crossing'] = dict(t=d.time-2., sole_z=clearance,
                        obstacle_top=float(d.geom_xpos[obstacle,2]+m.geom_size[obstacle,0]))
                # Match the flexible-rope audit: a foot-supported upright
                # landing, without body-ground or obstacle contact. Both feet
                # must clear the obstacle, but need not touch down simultaneously.
                supported = bool(grounded & feet) and not (grounded-feet)
                supported &= not touching_obstacle and duck.is_upright(.8)
                support_time = support_time+m.opt.timestep if supported else 0.
                if len(current['crossings']) == 2 and support_time >= .05:
                    current['landed'] = True
                if math.floor((q-math.pi)/(2*math.pi)) > math.floor((previous_q-math.pi)/(2*math.pi)):
                    crossings = current['crossings'].values()
                    cleared = len(current['crossings']) == 2 and all(
                        c['within_span'] and c['sole_z']>c['obstacle_top']+.001 for c in crossings)
                    current['clean'] = bool(cleared
                        and current['contact_steps']==0 and current['body_ground_steps']==0
                        and current['min_cos']>.8 and current['max_drift']<.08 and current['landed'])
                    cycles.append(current)
                    current = dict(contact_steps=0, body_ground_steps=0, min_cos=1., max_drift=0., crossing=None, crossings={}, landed=False)
                previous_q = q
            traces.append(dict(t=(p+1)*.02, phase=float(d.qpos[qadr]),
                target_z=float(duck.command[9]), trunk_z=float(duck.trunk_pos()[2])))
            if renderer and p%2 == 0:
                from PIL import Image, ImageDraw
                renderer.update_scene(d,camera)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0,0,960,48), fill='black')
                draw.text((10,7),'TRAINING APPARATUS - physical contact; NOT full rope acceptance',fill='white')
                draw.text((10,27),f'{Path(policy).name} | clean passes {sum(c["clean"] for c in cycles)}/{len(cycles)}',fill='white')
                writer.append_data(np.asarray(im))
    finally:
        if renderer: renderer.close()
        if writer: writer.close()
    return dict(kind='training_apparatus_only', scorer_version='sweep-v3', policy=str(policy),
        sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),seed=seed,seconds=seconds,
        frequency=frequency,feedback=feedback,clean_passes=sum(c['clean'] for c in cycles),
        cycles=cycles,max_drift=max_drift,trace=traces)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('policy',type=Path)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--frequency',type=float,default=3.1)
    p.add_argument('--no-feedback',action='store_true')
    p.add_argument('--video',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.policy,a.seed,a.seconds,a.frequency,not a.no_feedback,a.video)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cycles','trace')},indent=2))
