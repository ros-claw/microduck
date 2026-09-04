"""Classic (overhead) rope skipping world — the elastic-cable rope forms a real
rotating loop that goes OVER the jumper's head and grazes the floor, with three
ducks in the scene. The rope ends ride on mocap carriers (the declared "firm
grip" idealization — the turner ducks' whole-body drive is the RL program, see
the 0904 doc); the rope's rotation physics, the jumper's contacts, and the
trips are all real.

The elastic cable (bend stiffness) is load-bearing: the floppy serial chain
never inflates into a loop (measured — CR-02), the cable with mild bend
stiffness does. CPU MuJoCo only (the Warp GPU backend rejects body plugins).
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

from .composer import DuckSpec, DUCK_COLORS, _yaw_quat, _tint_duck
from ..sim.runtime import PolicyBank, DuckRuntime

_ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = _ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"


def build_classic_world(
    robot_xml=str(ROBOT),
    ducks: list[DuckSpec] | None = None,
    rope_length: float = 0.86,
    rope_density: float = 400.0,
    rope_bend: float = 1e3,
    rope_radius: float = 0.003,
    turner_sep: float = 0.50,
    carrier_height: float = 0.18,
    playground: bool = False,
):
    """3 ducks + an elastic-cable rope on mocap carriers, one MuJoCo world.

    Returns (model, data, info) with info = {rope_body_ids, mocap_a, mocap_b,
    carrier_centers}.
    """
    if ducks is None:
        ducks = [
            DuckSpec("lavender", (-turner_sep / 2, 0.0), yaw=0.0, color=DUCK_COLORS["lavender"]),
            DuckSpec("cream", (turner_sep / 2, 0.0), yaw=math.pi, color=DUCK_COLORS["cream"]),
            DuckSpec("sky", (0.0, -0.10), yaw=math.pi / 2, color=DUCK_COLORS["sky"]),
        ]
    cA = np.array([-turner_sep / 2, 0.0, carrier_height])
    cB = np.array([turner_sep / 2, 0.0, carrier_height])

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001          # the elastic cable needs fine steps
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 20
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1088
    # lighting + a visible floor (the rope and ducks must read on video)
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    light = spec.worldbody.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec.add_texture(name="gp", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.36, 0.55, 0.36], rgb2=[0.30, 0.47, 0.30],
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.85, 0.85, 0.85],
                     width=300, height=300)
    gmat = spec.add_material(name="gp")
    gmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "gp"
    gmat.texuniform = True
    gmat.texrepeat = [4, 4]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    floor.material = "gp"

    # ducks
    for d in ducks:
        child = mujoco.MjSpec.from_file(robot_xml)
        if d.color:
            _tint_duck(child, d.color)
        frame = spec.worldbody.add_frame(pos=[d.pos[0], d.pos[1], 0.0], quat=_yaw_quat(d.yaw))
        spec.attach(child, prefix=f"{d.name}/", frame=frame)

    # rope carriers (mocap, non-colliding)
    for cname, p in (("ropeA", cA), ("ropeB", cB)):
        cb = spec.worldbody.add_body(name=cname, pos=[float(p[0]), float(p[1]), float(p[2])])
        cb.mocap = True
        cb.add_geom(name=cname + "_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004],
                    rgba=[1, 0, 0, 0.0], contype=0, conaffinity=0)

    # elastic cable: first body is world-welded, so ride it on a massed
    # freejoint wrapper (see oracle_rope.py note)
    cable_xml = f"""
    <mujoco model="rope">
      <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
      <worldbody>
        <body name="ropewrap" pos="{cA[0]} {cA[1]} {cA[2]}">
          <freejoint/>
          <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 1 0 0"/>
          <composite type="cable" curve="s" count="41 1 1" size="{rope_length}"
                     offset="0 0 0" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <config key="twist" value="1e4"/><config key="bend" value="{rope_bend}"/>
            </plugin>
            <joint kind="main" damping="0.0001"/>
            <geom type="capsule" size="{rope_radius}" density="{rope_density}"
                  rgba="0.9 0.55 0.1 1" friction="0.05 0.005 0.0001" contype="1" conaffinity="1"/>
            <skin rgba="0.9 0.55 0.1 1" inflate="0.002"/>
          </composite>
        </body>
      </worldbody>
    </mujoco>
    """
    rope_spec = mujoco.MjSpec.from_string(cable_xml)
    rope_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
    spec.attach(rope_spec, prefix="rope/", frame=rope_frame)

    # connect: wrapper→carrierA, cable last body→carrierB
    for n1, n2 in (("rope/ropewrap", "ropeA"), ("rope/B_last", "ropeB")):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_CONNECT
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = n1
        eq.name2 = n2
        for k in range(3):
            eq.data[k] = 0.0

    # The turner ducks brace the rope they're holding — exclude rope↔turner
    # contact pairs (declared: they hold it, it doesn't knock them over). Only
    # the JUMPER's rope contact is physically scored (real trips).
    duck_names = [d.name for d in ducks]
    turner_names = sorted({duck_names[0], duck_names[1]})   # first two = turners
    # collect body names from the spec (rope segments + turner duck bodies)
    rope_body_names = sorted({(b.name or "") for b in spec.bodies
                              if (b.name or "").startswith("rope/")})
    turner_body_names = sorted({(b.name or "") for b in spec.bodies
                                if any((b.name or "").startswith(t + "/") for t in turner_names)})
    for rb in rope_body_names:
        for tb in turner_body_names:
            ex = spec.add_exclude()
            ex.bodyname1 = rb
            ex.bodyname2 = tb

    model = spec.compile()
    data = mujoco.MjData(model)

    rope_body_ids = [b for b in range(model.nbody)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("rope/B_")]
    mocap_a = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ropeA")])
    mocap_b = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ropeB")])
    info = dict(rope_body_ids=rope_body_ids, mocap_a=mocap_a, mocap_b=mocap_b,
                carrier_centers=(cA, cB), turner_names=tuple(turner_names))
    return model, data, info
