"""Relay-only scene builder; archived multiplayer scenes remain unchanged.

The lower receiver is wider than the arena so falling feet land on its top,
rather than hitting a narrow box side/corner. It is not arena-height support.
"""

import math
import mujoco
import numpy as np
from .world import ASSETS
from .tiles import Tile
from .multiplayer import NAMES, add_external_envelopes
from ..sim.composer import _yaw_quat
from ..sim.runtime import DuckRuntime, PolicyBank, apply_current_limit


def build_relay_world(config, seed):
    """The only root/joint initialization boundary; no later state assistance."""
    s = mujoco.MjSpec()
    s.option.timestep = config.dt
    s.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    s.option.iterations = config.solver_iterations
    s.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    if config.ccd == "libccd":
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    s.visual.global_.offwidth = 1920
    s.visual.global_.offheight = 1080
    s.visual.headlight.ambient = [0.4, 0.4, 0.4]
    s.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
    half = config.receiver_half_extent_m
    s.worldbody.add_geom(
        name="receiver",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -0.85],
        size=[half, half, 0.04],
        rgba=[0.06, 0.08, 0.12, 1],
        friction=[1, 0.005, 0.0001],
    )
    for i in config.ids:
        b = s.worldbody.add_body(name=f"tile_{i}", pos=[*config.centre(i), -0.025])
        b.add_freejoint(name=f"tile_{i}_free")
        b.add_geom(
            name=f"tile_{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[config.tile_size / 2, config.tile_size / 2, 0.025],
            mass=0.8,
            rgba=[0.13, 0.42, 0.5, 1],
            friction=[1, 0.005, 0.0001],
        )
        e = s.add_equality(name=f"tile_{i}_support")
        e.type, e.objtype, e.name1 = (
            mujoco.mjtEq.mjEQ_WELD,
            mujoco.mjtObj.mjOBJ_BODY,
            b.name,
        )
        e.solref, e.solimp = [0.002, 1], [0.999, 0.999, 0.001, 0.5, 2]
        e.data[3:10] = [0] * 7
    # Widely separated cardinal edge spawns, with a seed-based identity rotation.
    n = config.grid
    mid = n // 2
    spawn = [mid, mid * n + n - 1, (n - 1) * n + mid, mid * n]
    spawn = spawn[::2] if config.players == 2 else spawn
    names = list(NAMES[: config.players])
    shift = seed % config.players
    names = names[shift:] + names[:shift]
    robot = (
        ASSETS
        / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
    )
    native = mujoco.MjModel.from_xml_path(str(robot))
    spawn_map = {}
    for name, tile in zip(names, spawn, strict=True):
        if tile not in config.ids:
            raise ValueError("Layout removes a spawn")
        pos = config.centre(tile)
        yaw = math.atan2(-pos[1], -pos[0])
        spec = mujoco.MjSpec.from_file(str(robot))
        if config.external_envelopes:
            add_external_envelopes(spec, native)
        s.attach(
            spec,
            prefix=name + "/",
            frame=s.worldbody.add_frame(pos=[*pos, 0], quat=_yaw_quat(yaw)),
        )
        spawn_map[name] = dict(tile=tile, yaw=yaw)
    for g in s.geoms:
        if g.contype or g.conaffinity:
            g.solref, g.solimp = (
                (
                    [0.0005, 1]
                    if g.name.startswith("tile_")
                    else [config.contact_ref_s, 1]
                ),
                [0.99, 0.99, 0.001, 0.5, 2],
            )
    m = s.compile()
    d = mujoco.MjData(m)
    bank = PolicyBank(
        {
            n: str(ASSETS / "microduck/policies" / p)
            for n, p in {
                "stand": "alpha_stand.onnx",
                "walk": "alpha_walking.onnx",
            }.items()
        }
    )
    ducks = {
        name: DuckRuntime(m, d, bank, prefix=name + "/", name=name) for name in names
    }
    rng = np.random.default_rng(seed)
    for name, duck in ducks.items():
        apply_current_limit(m, name + "/")
        d.qpos[duck.joint_qpos_idx] = duck.default_pose + rng.uniform(-0.005, 0.005, 14)
        d.ctrl[duck.act_ids] = duck.default_pose
    tiles = {
        i: Tile(
            i,
            m.equality(f"tile_{i}_support").id,
            int(m.jnt_qposadr[m.joint(f"tile_{i}_free").id]),
            int(m.jnt_dofadr[m.joint(f"tile_{i}_free").id]),
        )
        for i in config.ids
    }
    mujoco.mj_forward(m, d)
    return m, d, ducks, tiles, spawn_map
