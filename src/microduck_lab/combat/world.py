"""Fixed stage; native Pollen materials, inertia and convex collision meshes."""

from pathlib import Path
import os
import math
import hashlib
import mujoco
import numpy as np
from ..sim.runtime import DuckRuntime, PolicyBank, apply_current_limit
from ..sim.composer import _yaw_quat

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(os.environ.get("MICRODUCK_ROOT", ROOT.parent))
ROBOT = "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POLICIES = {
    "stand": "alpha_stand.onnx",
    "walk": "alpha_walking.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "roulade": "roulade.onnx",
}
NAMES = ("lavender", "cream")


def external_parts(spec, index, players=2):
    """Separate collision layers avoid overlapping native+proxy responses.

    Native type1/type2 become per-robot self-contact layers; type1 retains
    stage interaction. Massless copies of visual mesh pieces contact ONLY the
    other duck's visual hull pieces. No broad AABB and no duplicate pair shell.
    """
    own1, own2 = 1 << (2 + 2 * index), 1 << (3 + 2 * index)
    if players not in (2, 4) or not 0 <= index < players:
        raise ValueError("Collision layers support exactly two or four ducks")
    external = 1 << (2 + 2 * players + index)
    opponent = sum(1 << (2 + 2 * players + j) for j in range(players) if j != index)
    geoms = list(spec.geoms)
    children = {}
    for i, g in enumerate(geoms):
        if g.contype or g.conaffinity:
            old_type, old_affinity = g.contype, g.conaffinity
            g.contype = (own1 if old_type & 1 else 0) | (own2 if old_type & 2 else 0)
            g.conaffinity = ((own1 | 1) if old_affinity & 1 else 0) | (
                own2 if old_affinity & 2 else 0
            )
        if g.group != 2 or not g.meshname:
            continue
        parent = g.parent
        if parent.name not in children:
            children[parent.name] = parent.add_body(name=parent.name + "_combat_shell")
        children[parent.name].add_geom(
            name=f"combat_piece_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=g.meshname,
            pos=g.pos.copy(),
            quat=g.quat.copy(),
            mass=0.0,
            density=0.0,
            group=3,
            contype=external,
            conaffinity=opponent,
            rgba=[0.85, 0.25, 0.2, 0.4],
            friction=[1, 0.005, 0.0001],
        )


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CombatBank(PolicyBank):
    """Bound CPU threads; validate each legacy motor input/output before use."""

    def get(self, name):
        if name not in self._sessions:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            s = ort.InferenceSession(
                self.paths[name], sess_options=opts, providers=["CPUExecutionProvider"]
            )
            if (
                len(s.get_inputs()) != 1
                or s.get_inputs()[0].shape != [1, 61]
                or s.get_inputs()[0].type != "tensor(float)"
                or s.get_outputs()[0].shape != [1, 14]
            ):
                raise ValueError(f"Invalid legacy 61D/14D policy: {name}")
            if name == "dash_candidate":
                meta = s.get_modelmeta().custom_metadata_map
                required = {
                    "duckfu_motor_schema": "legacy61.v1",
                    "duckfu_training_actuator": "position_servo_simulation_only",
                    "duckfu_motor_hz": "50",
                    "duckfu_normalization_baked": "true",
                }
                if any(meta.get(k) != v for k, v in required.items()):
                    raise ValueError(
                        "Candidate lacks the audited motor/export contract"
                    )
            self._sessions[name] = s
        return self._sessions[name]


def build(config, physics, seed):
    s = mujoco.MjSpec()
    s.option.timestep = physics.dt
    s.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    s.option.iterations = physics.iterations
    s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    s.visual.global_.offwidth, s.visual.global_.offheight = 1920, 1080
    s.visual.headlight.ambient = [0.45, 0.45, 0.45]
    s.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    s.worldbody.add_geom(
        name="receiver",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -0.55],
        size=[3, 3, 0.04],
        rgba=[0.16, 0.18, 0.21, 1],
        friction=[1, 0.005, 0.0001],
    )
    if config.arena == "lab":
        kind, size = mujoco.mjtGeom.mjGEOM_BOX, [config.radius_m, config.radius_m, 0.04]
    else:
        kind, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [config.radius_m, 0.04, 0]
    s.worldbody.add_geom(
        name="stage",
        type=kind,
        pos=[0, 0, -0.04],
        size=size,
        rgba=[0.62, 0.64, 0.65, 1],
        friction=[1, 0.005, 0.0001],
    )
    rng = np.random.default_rng(seed)
    h = config.heading_rad
    c, sn = math.cos(h), math.sin(h)
    rotation = np.array([[c, -sn], [sn, c]])
    order = NAMES[::-1] if config.swap else NAMES
    spawns = {}
    for i, name in enumerate(order):
        sign = -1 if i == 0 else 1
        xy = rotation @ np.array(
            [sign * config.separation_m / 2, (0 if i == 0 else config.lateral_offset_m)]
        )
        # The absent-target negative remains physical, but starts far away.
        if i == 1 and config.defense == "absent":
            xy = rotation @ np.array([0.9, 1.0])
        yaw = h + (0 if i == 0 else math.pi)
        spec = mujoco.MjSpec.from_file(str(ASSETS / ROBOT))
        if physics.collision == "visual_convex_parts":
            # Layers follow identity, independent of spawn/coach role.
            external_parts(spec, NAMES.index(name))
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                g.solref = [physics.contact_ref_s, 1]
                g.solimp = [0.99, 0.99, 0.001, 0.5, 2]
        s.attach(
            spec,
            prefix=name + "/",
            frame=s.worldbody.add_frame(pos=[*xy, 0], quat=_yaw_quat(yaw)),
        )
        spawns[name] = dict(
            xy=xy.tolist(), yaw=yaw, role="coach_attacker" if i == 0 else config.defense
        )
    if physics.collision == "visual_convex_parts":
        for g in s.geoms:
            if g.name in ("stage", "receiver"):
                g.contype, g.conaffinity = 1, (1 << 2) | (1 << 4)
    # Same contact profile on stage/receiver. No added collision exclusions.
    for g in s.geoms:
        if g.contype or g.conaffinity:
            g.solref, g.solimp = [physics.contact_ref_s, 1], [0.99, 0.99, 0.001, 0.5, 2]
    m = s.compile()
    d = mujoco.MjData(m)
    bank = CombatBank(
        {n: str(ASSETS / "microduck/policies" / p) for n, p in POLICIES.items()}
    )
    ducks = {
        name: DuckRuntime(m, d, bank, prefix=name + "/", name=name) for name in NAMES
    }
    # The only initialization boundary. Runtime never writes root state.
    for duck in ducks.values():
        apply_current_limit(m, duck.prefix)
        d.qpos[duck.joint_qpos_idx] = duck.default_pose + rng.uniform(-0.005, 0.005, 14)
        d.ctrl[duck.act_ids] = duck.default_pose
    mujoco.mj_forward(m, d)
    return m, d, ducks, spawns


def model_inventory(m):
    rows = []
    for i in range(m.ngeom):
        if m.geom_contype[i] or m.geom_conaffinity[i]:
            rows.append(
                dict(
                    id=i,
                    name=m.geom(i).name,
                    body=m.body(int(m.geom_bodyid[i])).name,
                    type=int(m.geom_type[i]),
                    mesh_id=int(m.geom_dataid[i]),
                    contype=int(m.geom_contype[i]),
                    conaffinity=int(m.geom_conaffinity[i]),
                    solref=m.geom_solref[i].tolist(),
                    friction=m.geom_friction[i].tolist(),
                )
            )
    return dict(
        collision_geoms=rows,
        exclusions=m.exclude_signature.tolist(),
        mass_by_duck={
            n: float(
                sum(
                    m.body_mass[i]
                    for i in range(m.nbody)
                    if m.body(i).name.startswith(n + "/")
                )
            )
            for n in NAMES
        },
        actuator_force_range=m.actuator_forcerange.tolist(),
        actuator_gain=m.actuator_gainprm.tolist(),
        actuator_bias=m.actuator_biasprm.tolist(),
        joint_damping=m.dof_damping.tolist(),
    )
