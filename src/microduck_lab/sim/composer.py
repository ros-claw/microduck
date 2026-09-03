"""Multi-duck MuJoCo world composer.

Loads the official robot_allcollisions.xml once per duck and attaches it into
a shared world with a namespace prefix (``lavender/``, ``cream/``, ...), so
several ducks share ONE physics world and interact with ONE rope.

The rope is a MuJoCo ``composite type="cable"`` chain (real bodies, real
collisions — no ghost rope), latched to the turners' ``jaw_soft`` bodies at
the ``mouth_tip`` anchor via connect equalities. The latch can be toggled at
runtime so ducks can pick up / drop the rope mid-episode.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

DUCK_COLORS = {
    "lavender": (0.72, 0.62, 0.95, 1.0),
    "cream": (0.96, 0.87, 0.66, 1.0),
    "sky": (0.55, 0.78, 0.98, 1.0),
    "graphite": (0.35, 0.36, 0.40, 1.0),
}

# mouth_tip site position in the jaw_soft body frame (from robot MJCF)
MOUTH_TIP_LOCAL = (-0.00809334, 0.0, -0.0777383)


@dataclass
class DuckSpec:
    name: str
    pos: tuple[float, float]
    yaw: float = 0.0
    color: tuple | None = None


@dataclass
class RopeSpec:
    turner_a: str                       # duck name holding rope start
    turner_b: str                       # duck name holding rope end
    length: float = 1.0                 # m
    count: int = 34                     # composite vertex count
    radius: float = 0.0035              # m
    density: float = 300.0              # kg/m^3 (light rope)
    color: tuple = (1.0, 0.55, 0.10, 1.0)


@dataclass
class ComposedWorld:
    model: mujoco.MjModel
    data: mujoco.MjData
    ducks: list[DuckSpec]
    rope_body_ids: list[int] = field(default_factory=list)
    rope_eq_ids: list[int] = field(default_factory=list)

    def set_rope_latched(self, latched: bool):
        for eq in self.rope_eq_ids:
            self.data.eq_active[eq] = int(latched)

    @property
    def has_rope(self) -> bool:
        return bool(self.rope_body_ids)

    def rope_kinematics(self) -> dict:
        """Sensor-world rope observation: positions of rope bodies.

        (In a real deployment this would come from vision; in sim v0 it is a
        deterministic 'rope detector' — see docs: sensor world, not oracle
        task state.)
        """
        pos = np.array([self.data.xpos[b] for b in self.rope_body_ids])
        return {
            "points": pos,
            "middle": pos[len(pos) // 2],
            "min_z": float(pos[:, 2].min()),
            "max_z": float(pos[:, 2].max()),
        }


def _yaw_quat(yaw: float):
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def _tint_duck(spec: mujoco.MjSpec, color):
    """Recolor a child duck spec's visual geoms before attach."""
    def walk(body):
        for g in body.geoms:
            if g.group == 2:  # visual class
                g.rgba = list(color)
        for b in body.bodies:
            walk(b)
    for b in spec.worldbody.bodies:
        walk(b)


def _rope_spec_xml(rope: RopeSpec, pA, pB) -> str:
    """Serial ball-joint chain (a real soft rope) spawned as a hanging arc.

    Both ends are FREE — ``seg_0`` carries a free joint (a ball joint would
    pin it to the world). The chain follows a parabola from pA down through a
    sag and up to pB, so spawning near the mouths produces no latch yank.
    Bodies: ``seg_0 .. seg_{n-1}`` plus tip body ``end`` (origin = rope end).
    """
    r, g, b, a = rope.color
    n = rope.count
    pA = np.asarray(pA, float); pB = np.asarray(pB, float)
    D = float(np.linalg.norm(pB[:2] - pA[:2]))
    z0 = float(pA[2])

    # solve sag s so that parabola arc length ≈ rope.length (guarded Newton)
    def arc_len(s):
        k = 4 * s / D
        if k < 1e-9:
            return D
        f = lambda t: np.sqrt(1 + (k * t) ** 2)
        return D / 2 * (f(1.0) + np.arcsinh(k) / k)
    if rope.length <= D * 1.01:
        # rope barely longer than the gap: shallow V
        s = max(0.01, 0.25 * np.sqrt(max(rope.length ** 2 - D ** 2, 1e-6)))
    else:
        lo, hi = 0.01, rope.length          # arc_len increasing in s
        for _ in range(60):                  # bisection, no derivatives
            mid_s = 0.5 * (lo + hi)
            if arc_len(mid_s) < rope.length:
                lo = mid_s
            else:
                hi = mid_s
        s = 0.5 * (lo + hi)

    # arc points + tangents (in the vertical plane containing pA→pB)
    pts, alphas = [], []
    for i in range(n + 1):
        u = i / n
        x = -D / 2 + u * D
        z = z0 - s * (1 - (2 * u - 1) ** 2)
        pts.append(np.array([x, 0.0, z]))
        dzdx = (4 * s / D) * (2 * u - 1)
        alphas.append(np.arctan2(-dzdx, 1.0))      # R_y(α) maps +X along tangent
    # rotate into world frame (arc plane may be yawed if pA,pB differ in y)
    yaw = np.arctan2(pB[1] - pA[1], pB[0] - pA[0])
    c, sn = np.cos(yaw), np.sin(yaw)
    mid = (pA + pB) / 2
    pts_w = []
    for p in pts:
        px = c * p[0] - sn * p[1] + mid[0]
        py = sn * p[0] + c * p[1] + mid[1]
        pz = p[2] - z0 + (pA[2] + pB[2]) / 2
        pts_w.append(np.array([px, py, pz]))

    seg = rope.length / n
    seg_mass = rope.density * np.pi * rope.radius ** 2 * seg

    def quat_rotY(alpha):
        return (np.cos(alpha / 2), 0.0, -np.sin(alpha / 2), 0.0)

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    def quat_inv(q):
        return (q[0], -q[1], -q[2], -q[3])

    yaw_q = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    abs_q = [quat_mul(yaw_q, quat_rotY(alphas[min(i, n - 1)])) for i in range(n)]

    xml = ['<mujoco model="rope">', '  <worldbody>']
    indent = "    "
    for i in range(n):
        if i == 0:
            rel_q = abs_q[0]
            pos_attr = f'pos="{pts_w[0][0]:.4f} {pts_w[0][1]:.4f} {pts_w[0][2]:.4f}"'
        else:
            rel_q = quat_mul(quat_inv(abs_q[i - 1]), abs_q[i])
            pos_attr = f'pos="{seg:.5f} 0 0"'
        joint = (f'{indent}  <freejoint name="rope_free"/>\n' if i == 0 else
                 f'{indent}  <joint name="rope_j{i}" type="ball" damping="0.0001"/>\n')
        xml.append(
            f'{indent}<body name="seg_{i}" {pos_attr} '
            f'quat="{rel_q[0]:.5f} {rel_q[1]:.5f} {rel_q[2]:.5f} {rel_q[3]:.5f}">\n'
            + joint +
            f'{indent}  <inertial mass="{seg_mass:.6f}" pos="{seg/2:.5f} 0 0"\n'
            f'{indent}           diaginertia="1e-8 1e-8 1e-8"/>\n'
            f'{indent}  <geom name="rope_g{i}" type="capsule" fromto="0 0 0 {seg:.5f} 0 0"\n'
            f'{indent}        size="{rope.radius}" rgba="{r} {g} {b} {a}"\n'
            f'{indent}        friction="0.4 0.02 0.001" contype="1" conaffinity="1"/>\n'
        )
        indent += "  "
    # tip body: latch point for turner B, extends the final tangent
    xml.append(
        f'{indent}<body name="end" pos="{seg:.5f} 0 0">\n'
        f'{indent}  <joint name="rope_jend" type="ball" damping="0.0001"/>\n'
        f'{indent}  <inertial mass="{seg_mass:.6f}" pos="0 0 0" diaginertia="1e-8 1e-8 1e-8"/>\n'
    )
    xml.append("</body>\n" * (n + 1))
    xml.append("  </worldbody>\n</mujoco>\n")
    xml_str = "".join(xml)

    # initial free-joint pose for post-compile qpos fixup
    q0 = np.array([*pts_w[0], *abs_q[0]])
    return xml_str, q0


def compose_world(
    robot_xml: str | pathlib.Path,
    ducks: list[DuckSpec],
    rope: RopeSpec | None = None,
    rope_height: float = 0.21,
    playground: bool = True,
    grippy_ducks: list[str] | None = None,
) -> ComposedWorld:
    """Compose ducks (+ optional rope) into one compiled MuJoCo world."""
    robot_xml = str(robot_xml)
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100 if rope is not None else 8
    spec.option.ls_iterations = 20
    # big offscreen framebuffer for HD rendering
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1088

    # --- arena visuals ---
    if playground:
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.add_texture(
            name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.55, 0.70, 0.90], rgb2=[0.10, 0.14, 0.22],
            width=512, height=3072,
        )
        spec.add_texture(
            name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=[0.36, 0.55, 0.36], rgb2=[0.30, 0.47, 0.30],
            markrgb=[0.85, 0.85, 0.85], width=300, height=300,
        )
        mat = spec.add_material(name="groundplane")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
        mat.texuniform = True
        mat.texrepeat = [4, 4]
        floor = spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
        floor.material = "groundplane"
    else:
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    light = spec.worldbody.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    if playground:
        # circus ring: a flat disc (visual only) + corner cones
        ring = spec.worldbody.add_geom(name="ring", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                       size=[0.85, 0.002], pos=[0, 0, 0.001])
        ring.contype = 0; ring.conaffinity = 0
        ring.rgba = [0.92, 0.88, 0.80, 1.0]
        for cx, cy in ((-0.55, 0.55), (0.55, 0.55), (-0.55, -0.55), (0.55, -0.55)):
            cone = spec.worldbody.add_geom(name=f"cone{cx}{cy}",
                                           type=mujoco.mjtGeom.mjGEOM_CONE if hasattr(mujoco.mjtGeom, 'mjGEOM_CONE') else mujoco.mjtGeom.mjGEOM_CYLINDER,
                                           size=[0.035, 0.06, 0], pos=[cx, cy, 0.03])
            cone.contype = 0; cone.conaffinity = 0
            cone.rgba = [1.0, 0.45, 0.15, 1.0]

    # --- ducks ---
    for d in ducks:
        child = mujoco.MjSpec.from_file(robot_xml)
        if d.color:
            _tint_duck(child, d.color)
        frame = spec.worldbody.add_frame(pos=[d.pos[0], d.pos[1], 0.0],
                                         quat=_yaw_quat(d.yaw))
        spec.attach(child, prefix=f"{d.name}/", frame=frame)

    # --- rope ---
    rope_eq_specs: list = []
    rope_q0 = None
    if rope is not None:
        da = next(d for d in ducks if d.name == rope.turner_a)
        db = next(d for d in ducks if d.name == rope.turner_b)
        # estimated mouth positions at the settled standing pose
        pA = np.array([da.pos[0] + 0.086 * np.cos(da.yaw),
                       da.pos[1] + 0.086 * np.sin(da.yaw), rope_height])
        pB = np.array([db.pos[0] + 0.086 * np.cos(db.yaw),
                       db.pos[1] + 0.086 * np.sin(db.yaw), rope_height])
        rope_xml, rope_q0 = _rope_spec_xml(rope, pA, pB)
        rope_spec = mujoco.MjSpec.from_string(rope_xml)
        rope_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
        spec.attach(rope_spec, prefix="rope/", frame=rope_frame)

        for turner, rbody in ((rope.turner_a, "rope/seg_0"), (rope.turner_b, "rope/end")):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = f"{turner}/jaw_soft"
            eq.name2 = rbody
            for i in range(3):
                eq.data[i] = MOUTH_TIP_LOCAL[i]
            eq.solref = [0.02, 1.0]
            rope_eq_specs.append(eq)

    model = spec.compile()
    data = mujoco.MjData(model)

    # Real PU soles (grippy + compliant) — but ONLY for ducks that jump.
    # Explosive policies (jump) transfer only with the soft grippy sole
    # (mu~2, solref 0.04 — infer_policy.py's sole emulation). Turners keep
    # the stiffer default sole: grippy-soft feet change the sitting contact
    # and kill the rope swing. `grippy_ducks` selects which ducks get it.
    grippy = set(grippy_ducks or [])
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm and re.search(r"(left|right)_foot_collision$", nm):
            duck = nm.split("/")[0]
            if duck in grippy:
                model.geom_friction[g, 0] = 2.0
                model.geom_solref[g] = [0.04, 1.0]

    if rope is not None:
        # MuJoCo auto-computes the body2 anchor of a connect equality at qpos0
        # — where our attached rope sits at the world origin, which silently
        # breaks the latch. Override: anchor on the rope body = its origin
        # (seg_0 / end ARE the rope endpoints).
        for eq_i in range(model.neq):
            model.eq_data[eq_i][3:6] = [0.0, 0.0, 0.0]
        # Free-joint bodies ignore their frame offset at attach time; set the
        # rope's spawn pose explicitly (hanging arc between the mouths).
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rope/rope_free")
        adr = int(model.jnt_qposadr[jid])
        data.qpos[adr:adr + 7] = rope_q0

    world = ComposedWorld(model=model, data=data, ducks=ducks)
    if rope is not None:
        ids = []
        for b in range(model.nbody):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if n and re.match(r"rope/(seg_\d+|end)$", n):
                ids.append(b)
        # order along the rope: seg_0 .. seg_{n-1} .. end
        def _key(b):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if n.endswith("/end"):
                return 10**6
            return int(n.rsplit("_", 1)[1])
        world.rope_body_ids = sorted(ids, key=_key)
        world.rope_eq_ids = [int(e.id) for e in rope_eq_specs]
    return world
