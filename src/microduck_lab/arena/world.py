"""Independent free-body tiles and native-material robot in one MuJoCo world."""
from pathlib import Path
import os
import mujoco
from ..sim.composer import DuckSpec, _yaw_quat
from ..sim.runtime import DuckRuntime, PolicyBank, apply_current_limit

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path(os.environ.get('MICRODUCK_ROOT', REPO.parent))


def build_world(dt=.0005, tile_size=.48, gap=.012):
    if dt not in (.005, .002, .001, .0005):
        raise ValueError('dt must divide the 50 Hz motor period')
    s = mujoco.MjSpec()
    s.option.timestep = dt
    s.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    s.option.iterations = 40
    s.visual.global_.offwidth = 1920
    s.visual.global_.offheight = 1080
    s.visual.headlight.ambient = [.4, .4, .4]
    s.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
    # Only the visible receiver is fixed. No support plane at arena height.
    s.worldbody.add_geom(name='receiver', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -.85], size=[1.1, 1.1, .04], rgba=[.14, .17, .2, 1],
        friction=[1, .005, .0001])
    pitch = tile_size + gap
    for i in range(9):
        x, y = ((i // 3 - 1) * pitch, (i % 3 - 1) * pitch)
        b = s.worldbody.add_body(name=f'tile_{i}', pos=[x, y, -.025])
        b.add_freejoint(name=f'tile_{i}_free')
        b.add_geom(name=f'tile_{i}_geom', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tile_size/2, tile_size/2, .025], mass=.8,
            rgba=[.23, .48, .56, 1], friction=[1, .005, .0001],
            solref=[.006, 1])
        e = s.add_equality(name=f'tile_{i}_support')
        e.type = mujoco.mjtEq.mjEQ_WELD
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = b.name
        e.solref = [.002, 1]
        e.solimp = [.999, .999, .001, .5, 2]
        # MuJoCo computes initial relative pose for a zero relpose quaternion.
        e.data[3:10] = [0]*7
    duck_spec = DuckSpec('cream', (-pitch, 0))
    robot = ASSETS / 'microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml'
    s.attach(mujoco.MjSpec.from_file(str(robot)), prefix=duck_spec.name+'/',
        frame=s.worldbody.add_frame(pos=[*duck_spec.pos, 0], quat=_yaw_quat(duck_spec.yaw)))
    for g in s.geoms:
        if g.contype or g.conaffinity:
            g.solref = [.001, 1]
            g.solimp = [.99, .99, .001, .5, 2]
    m = s.compile()
    d = mujoco.MjData(m)
    paths = {n: str(ASSETS/'microduck/policies'/p) for n, p in
             {'stand':'alpha_stand.onnx', 'walk':'alpha_walking.onnx'}.items()}
    duck = DuckRuntime(m, d, PolicyBank(paths), prefix='cream/', name='cream')
    apply_current_limit(m, 'cream/')
    # Explicit initialization only; the episode never writes root state.
    d.qpos[duck.joint_qpos_idx] = duck.default_pose
    d.ctrl[duck.act_ids] = duck.default_pose
    mujoco.mj_forward(m, d)
    return m, d, duck
