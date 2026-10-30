"""One shared fixed arena; all enemy AND friendly bodies physically collide."""

import math
import mujoco
import numpy as np

from .world import ASSETS, ROBOT, POLICIES, CombatBank, external_parts, model_inventory
from ..sim.runtime import DuckRuntime, apply_current_limit
from ..sim.composer import _yaw_quat

IDENTITIES = ("lavender", "cream", "coral", "cyan")
DISPLAY_NAMES = ("ROOK", "BOLT", "SHADE", "ANCHOR")
CHINESE_NAMES = ("张三", "莽哥", "老六", "稳哥")


def build_match(config, physics, seed):
    s = mujoco.MjSpec()
    s.option.timestep, s.option.iterations = physics.dt, physics.iterations
    s.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    s.visual.global_.offwidth, s.visual.global_.offheight = 1920, 1920
    s.visual.headlight.ambient = [.4, .4, .4]
    s.worldbody.add_light(pos=[-1, -1, 3], dir=[.1, .1, -1], diffuse=[.8, .8, .8])
    s.worldbody.add_light(pos=[1, 1, 2], dir=[-.3, -.3, -1], diffuse=[.45, .45, .5])
    stage_affinity = sum(1 << (2+2*i) for i in range(config.players))
    s.worldbody.add_geom(name="receiver", type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=[0, 0, -.55], size=[3, 3, .04], rgba=[.025, .035, .055, 1],
                         contype=1, conaffinity=stage_affinity)
    s.worldbody.add_geom(name="stage", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[0, 0, -.04], size=[config.radius_m, .04, 0],
                         rgba=[.38, .42, .46, 1], contype=1, conaffinity=stage_affinity)
    # Flush visual markings: no added collision, mass, barriers or footholds.
    for i, color in enumerate(((.92,.19,.22,1), (.12,.52,.92,1))):
        s.worldbody.add_geom(name=f"team_mark_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=[(-1 if i == 0 else 1)*config.radius_m*.55, 0, .0002],
                             size=[.003, config.radius_m*.42, .0001], rgba=color,
                             contype=0, conaffinity=0, mass=0, group=2)
    rng = np.random.default_rng(seed)
    c, sn = math.cos(config.heading_rad), math.sin(config.heading_rad)
    rotation = np.array([[c, -sn], [sn, c]])
    spawns, teams = {}, {}
    for i, name in enumerate(IDENTITIES[:config.players]):
        side = (-1 if i % 2 == 0 else 1) * (-1 if config.swap else 1)
        y = (0 if config.players == 2 else (-.20 if i < 2 else .20))
        xy = rotation @ [side*config.separation_m/2, y]
        yaw = config.heading_rad + (0 if side < 0 else math.pi)
        spec = mujoco.MjSpec.from_file(str(ASSETS / ROBOT))
        external_parts(spec, i, config.players)
        s.attach(spec, prefix=name+"/", frame=s.worldbody.add_frame(pos=[*xy, 0], quat=_yaw_quat(yaw)))
        team = "RED" if i % 2 == 0 else "BLUE"
        teams[name] = team
        spawns[name] = dict(xy=xy.tolist(), yaw=yaw, team=team, display_name=DISPLAY_NAMES[i],
                            chinese_name=CHINESE_NAMES[i], role="competitor")
    for g in s.geoms:
        if g.contype or g.conaffinity:
            g.solref, g.solimp = [physics.contact_ref_s, 1], [.99, .99, .001, .5, 2]
            g.friction = [1, .005, .0001]
    m = s.compile()
    d = mujoco.MjData(m)
    bank = CombatBank({n:str(ASSETS/"microduck/policies"/p) for n,p in POLICIES.items()})
    if config.candidate_motor_path:
        bank.paths["dash_candidate"] = config.candidate_motor_path
    ducks = {n:DuckRuntime(m, d, bank, prefix=n+"/", name=n) for n in spawns}
    for duck in ducks.values():
        apply_current_limit(m, duck.prefix)
        d.qpos[duck.joint_qpos_idx] = duck.default_pose + rng.uniform(-.005, .005, 14)
        d.ctrl[duck.act_ids] = duck.default_pose
    mujoco.mj_forward(m, d)
    return m, d, ducks, spawns, teams


def match_inventory(m, names):
    inventory = model_inventory(m)
    inventory["mass_by_duck"] = {n:float(sum(m.body_mass[i] for i in range(m.nbody)
                                             if m.body(i).name.startswith(n+"/"))) for n in names}
    return inventory
