"""Robot-scaled parkour course with real voids, using native asset materials."""

from dataclasses import dataclass
from itertools import product
import mujoco
import numpy as np
from ..game.world import ASSETS, REPO
from ..sim.runtime import DuckRuntime, apply_current_limit
from .policies import ParkourPolicyBank


def geom_vertices(m, d, g):
    """World-space mesh vertices or conservative primitive bounding corners."""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        start = m.mesh_vertadr[mesh]
        points = m.mesh_vert[start : start + m.mesh_vertnum[mesh]]
    else:
        size = m.geom_size[g].copy()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            size[:] = size[0]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            size = np.array([size[0], size[0], size[1] + size[0]])
        points = np.array(list(product((-1, 1), repeat=3))) * size
    return points @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def duck_bounds(m, d, feet_only=False, include_visual=False):
    points = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if not body.startswith("duck/") or not (
            (m.geom_contype[g] & 1) or (include_visual and m.geom_group[g] == 2)
        ):
            continue
        if feet_only and "ankle" not in body:
            continue
        points.append(geom_vertices(m, d, g))
    points = np.concatenate(points)
    return points.min(axis=0), points.max(axis=0)


@dataclass(frozen=True)
class Track:
    duck_width: float
    duck_length: float
    lane_width: float
    width: float
    start: float = -1.5
    end: float = 10.0
    gaps: tuple = ()

    @property
    def lanes(self):
        return (-self.lane_width, 0.0, self.lane_width)


def robot_spec():
    spec = mujoco.MjSpec.from_file(
        str(
            ASSETS
            / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
        )
    )
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"part_{i}"
    return spec


def build_world(
    gaps=(),
    end=10.0,
    grip=False,
    hazards=(),
    rails=False,
    contact_pairs=True,
    hazard_proxies=True,
    recovery_bay=None,
    start=-1.5,
    duck_floor_ref=None,
    hard_contacts=False,
):
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.0002
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 40
    spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    robot = robot_spec()
    if hard_contacts:
        if duck_floor_ref not in (None, 0.002):
            raise ValueError("Hard-contact profile fixes ground reference at .002 s")
        duck_floor_ref = 0.002
        for geom in robot.geoms:
            if geom.contype & 1:
                geom.solref = [0.0012, 1.0]
                geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
                geom.margin = 0.0
    spec.attach(robot, prefix="duck/", frame=spec.worldbody.add_frame())
    # Measure the actual standing collision shell, not the head height.
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    bank = ParkourPolicyBank(
        {
            **{
                k: str(ASSETS / "microduck/policies" / v)
                for k, v in {
                    "stand": "alpha_stand.onnx",
                    "run": "alpha_walking.onnx",
                    "roulade": "roulade.onnx",
                }.items()
            },
            "jump": str(REPO / "policies/jump.onnx"),
        }
    )
    r0 = DuckRuntime(m0, d0, bank, prefix="duck/")
    d0.qpos[r0.joint_qpos_idx] = r0.default_pose
    mujoco.mj_forward(m0, d0)
    lo, hi = duck_bounds(m0, d0, include_visual=True)
    lane = float(np.ceil(1.4 * (hi[1] - lo[1]) * 100) / 100)
    track = Track(
        float(hi[1] - lo[1]),
        float(hi[0] - lo[0]),
        lane,
        3 * lane,
        start=start,
        end=end,
        gaps=tuple(gaps),
    )
    cursor = track.start
    segments = []
    for left, right in sorted(gaps):
        if not cursor < left < right < end:
            raise ValueError("Gaps must be disjoint, positive and inside the track")
        segments.append((cursor, left))
        cursor = right
    segments.append((cursor, end))
    widths = [track.width] * len(segments)
    if recovery_bay is not None:
        bay_left, bay_right, bay_width = recovery_bay
        rebuilt = []
        widths = []
        for left, right in segments:
            cuts = sorted(
                set(
                    [left, right]
                    + [x for x in (bay_left, bay_right) if left < x < right]
                )
            )
            for a, b in zip(cuts, cuts[1:]):
                rebuilt.append((a, b))
                widths.append(
                    bay_width if bay_left <= (a + b) / 2 <= bay_right else track.width
                )
        segments = rebuilt
    for i, (left, right) in enumerate(segments):
        width = widths[i]
        spec.worldbody.add_geom(
            name=f"floor/{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[(left + right) / 2, 0, -0.04],
            size=[(right - left) / 2, width / 2, 0.04],
            rgba=[0.045, 0.065, 0.1, 1],
            friction=[1.0, 0.005, 0.0001],
            conaffinity=5,
        )
        for side in (-1, 1):
            spec.worldbody.add_geom(
                name=f"edge/{i}/{side}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(left + right) / 2, side * width / 2, 0.002],
                size=[(right - left) / 2, 0.004, 0.001],
                rgba=[0.03, 0.8, 1, 1],
                contype=0,
                conaffinity=0,
            )
    # Low physical side rails keep the rolling boss on the narrow course.
    # They end at every gap: neither duck nor props can use a hidden bridge.
    for i, (left, right) in enumerate(segments if rails else []):
        width = widths[i]
        for side in (-1, 1):
            spec.worldbody.add_geom(
                name=f"rail/{i}/{side}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(left + right) / 2, side * (width / 2 + 0.018), 0.0125],
                size=[(right - left) / 2, 0.018, 0.0125],
                rgba=[0.07, 0.16, 0.23, 1],
                solref=[0.002, 1.0],
                solimp=[0.99, 0.999, 0.0005, 0.5, 2.0],
                conaffinity=5,
            )
    # Close scenery is explicitly visual; no invisible support below the gaps.
    for i, x in enumerate(np.arange(-1, end, 0.4)):
        for side in (-1, 1):
            spec.worldbody.add_geom(
                name=f"post/{i}/{side}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, side * max(0.55, track.width / 2 + 0.15), 0.25],
                size=[0.025, 0.025, 0.25],
                rgba=[0.1, 0.18, 0.24, 1],
                contype=0,
                conaffinity=0,
            )
    proxies = []
    if hazard_proxies and hazards:
        # Body-local boxes conservatively enclose each rigid visual part. They
        # affect hazards only (8<->4), not floor support or robot self-contact.
        # Explicit mass=0 preserves every original robot inertia.
        for bid in range(1, m0.nbody):
            body_name = m0.body(bid).name
            if not body_name.startswith("duck/"):
                continue
            ids = [
                g
                for g in range(m0.ngeom)
                if m0.geom_bodyid[g] == bid
                and (m0.geom_group[g] == 2 or m0.geom_contype[g] & 1)
            ]
            if not ids:
                continue
            points = np.concatenate([geom_vertices(m0, d0, g) for g in ids])
            local = (points - d0.xpos[bid]) @ d0.xmat[bid].reshape(3, 3)
            low, high = local.min(0), local.max(0)
            name = body_name + "/hazard_proxy"
            spec.body(body_name).add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(low + high) / 2,
                size=np.maximum((high - low) / 2 + 0.0002, 0.001),
                mass=0.0,
                group=4,
                rgba=[0.2, 1.0, 0.3, 0.25],
                contype=8,
                conaffinity=4,
            )
            proxies.append(name)
    for hazard in hazards:
        hazard.add_to(spec)
        if getattr(hazard, "kind", None) is not None and hazard_proxies:
            geom = next(g for g in spec.geoms if g.name == hazard.name + "/geom")
            geom.contype = 4
            geom.conaffinity = 12

    if contact_pairs:
        # Explicit pairs prevent the native duck's soft floor-contact reference
        # from being averaged into a fast obstacle impact. Floor physics stays native.
        profiles = {
            "sweeper": (0.0008, 0.4),
            "push_bar": (0.0012, 0.5),
            "crate": (0.0012, 0.35),
            "boulder": (0.001, 0.4),
            "finish_gate": (0.0012, 0.4),
            "bowling_ball": (0.0008, 0.25),
            "pin": (0.0008, 0.35),
            "guide": (0.0008, 0.3),
        }
        for hazard in hazards:
            kind = getattr(hazard, "kind", None)
            if kind not in profiles:
                continue
            timeconst, friction = profiles[kind]
            names = (
                proxies
                if hazard_proxies
                else [
                    m0.geom(g).name for g in range(m0.ngeom) if m0.geom_contype[g] & 1
                ]
            )
            for name in names:
                spec.add_pair(
                    geomname1=name,
                    geomname2=hazard.name + "/geom",
                    condim=3,
                    margin=0.0002,
                    solref=[timeconst, 1.0],
                    solimp=[0.99, 0.999, 0.0005, 0.5, 2.0],
                    friction=[friction, friction, 0.003, 0.0001, 0.0001],
                )
    # Props need their own rigid floor/rail contacts as well: the native floor
    # default is appropriate for duck feet, but too soft for a falling 0.8 kg ball.
    for h in hazards:
        for geom in list(spec.geoms):
            if geom.name.startswith(("floor/", "rail/")):
                spec.add_pair(
                    geomname1=h.name + "/geom",
                    geomname2=geom.name,
                    condim=3,
                    margin=0.0002,
                    solref=[0.0008, 1.0],
                    solimp=[0.99, 0.999, 0.0005, 0.5, 2.0],
                    friction=[1.2, 1.2, 0.003, 0.0001, 0.0001]
                    if h.kind == "pin"
                    else [0.25, 0.25, 0.003, 0.0001, 0.0001],
                )
    if duck_floor_ref is not None:
        native = [m0.geom(g).name for g in range(m0.ngeom) if m0.geom_contype[g] & 1]
        for ground in list(spec.geoms):
            if not ground.name.startswith(("floor/", "rail/")):
                continue
            for name in native:
                spec.add_pair(
                    geomname1=name,
                    geomname2=ground.name,
                    condim=3,
                    margin=0.0002,
                    solref=[duck_floor_ref, 1.0],
                    solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
                    friction=[1.0, 1.0, 0.005, 0.0001, 0.0001],
                )
    m = spec.compile()
    d = mujoco.MjData(m)
    r = DuckRuntime(m, d, bank, prefix="duck/")
    apply_current_limit(m, "duck/")
    if grip:
        for g in range(m.ngeom):
            if m.geom(g).name.endswith("_foot_collision"):
                m.geom_friction[g, 0] = 2.0
                m.geom_solref[g] = [0.04, 1.0]
    d.qpos[r.joint_qpos_idx] = r.default_pose
    d.ctrl[r.act_ids] = r.default_pose
    for h in hazards:
        if h.kind == "sweeper":
            d.qpos[m.jnt_qposadr[m.joint(h.name + "/joint").id]] = h.phase
    mujoco.mj_forward(m, d)
    return m, d, r, track


def foot_support(m, d, floor_name=None):
    """Only real sole/platform contacts count as landing support."""
    for ci, c in enumerate(d.contact):
        for foot, floor in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if "ankle" in m.body(m.geom_bodyid[foot]).name and m.geom(
                floor
            ).name.startswith("floor/"):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, force)
                if (
                    force[0] > 1e-5
                    and c.dist <= 0.0002
                    and (floor_name is None or m.geom(floor).name == floor_name)
                ):
                    return True
    return False
