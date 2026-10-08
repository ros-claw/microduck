"""Shared-world Duckverse. The actor API contains only current public state.

Old DG-01/02 calibration modules are deliberately preserved for archived replays.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import gzip
import json
import math
import time
import hashlib

import mujoco
import numpy as np

from .world import ASSETS
from .tiles import Tile
from .referee import Referee
from .episode import STATE, sha
from ..sim.composer import _yaw_quat
from ..sim.runtime import DuckRuntime, PolicyBank, apply_current_limit

NAMES = ("lavender", "cream", "sky", "graphite")


@dataclass(frozen=True)
class GameConfig:
    players: int = 4
    grid: int = 5
    tile_size: float = 0.48
    gap: float = 0.012
    dt: float = 0.00025
    layout: str = "square"
    cadence: str = "steady"
    duration: float = 45.0
    contact_ref_s: float = 0.0005
    solver_iterations: int = 80
    external_envelopes: bool = True
    ccd: str = "libccd"

    def __post_init__(self):
        if self.players not in (2, 4) or self.grid not in (4, 5):
            raise ValueError("Use two/four robots and a four/five tile grid")
        if self.layout not in ("square", "ring", "cross", "terraces"):
            raise ValueError("Unknown layout")
        if self.cadence not in ("steady", "rapid") or self.dt not in (
            0.001,
            0.0005,
            0.00025,
        ):
            raise ValueError("Unknown cadence or unsupported integration rate")
        if not 10 <= self.duration <= 90 or not 0.42 <= self.tile_size <= 0.52:
            raise ValueError("Invalid physical dimensions/duration")
        if not 0.005 <= self.gap <= 0.02:
            raise ValueError("Require a physically calibrated seam")
        if self.ccd not in ("native", "libccd"):
            raise ValueError("Unsupported convex contact backend")
        if self.contact_ref_s not in (0.001, 0.0005) or self.solver_iterations not in (
            40,
            80,
            160,
        ):
            raise ValueError("Unsupported contact solver calibration")

    @property
    def pitch(self):
        return self.tile_size + self.gap

    def centre(self, i):
        return np.array(
            [
                (i // self.grid - (self.grid - 1) / 2) * self.pitch,
                (i % self.grid - (self.grid - 1) / 2) * self.pitch,
            ]
        )

    @property
    def ids(self):
        n = self.grid
        return [
            i
            for i in range(n * n)
            if self.layout == "square"
            or (self.layout == "ring" and (i // n in (0, n - 1) or i % n in (0, n - 1)))
            or (
                self.layout == "cross"
                and (abs(i // n - (n - 1) / 2) <= 1 or abs(i % n - (n - 1) / 2) <= 1)
            )
            or (
                self.layout == "terraces"
                and not (i // n in (0, n - 1) and i % n in (0, n - 1))
            )
        ]

    @property
    def warning_s(self):
        return 5.0 if self.cadence == "steady" else 3.5


def add_external_envelopes(spec, native):
    """Massless conservative head/torso AABBs derived from native visual vertices.

    Original native self contacts remain active; only the added child geoms
    are excluded from their own robot. These are external collision proxies.
    """
    original_bodies = [native.body(i).name for i in range(1, native.nbody)]
    bounds = {}
    for name in ("jaw_soft", "trunk_base"):
        points = []
        bid = native.body(name).id
        for g in range(native.ngeom):
            if native.geom_bodyid[g] != bid or native.geom_group[g] != 2:
                continue
            mesh = int(native.geom_dataid[g])
            start = int(native.mesh_vertadr[mesh])
            end = start + int(native.mesh_vertnum[mesh])
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, native.geom_quat[g])
            points.extend(
                native.mesh_vert[start:end] @ mat.reshape(3, 3).T + native.geom_pos[g]
            )
        xyz = np.array(points)
        low, high = xyz.min(0) - 0.0005, xyz.max(0) + 0.0005
        child = spec.body(name).add_body(name=name + "_external_contact")
        child.add_geom(
            name=name + "_external_envelope",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(low + high) / 2,
            size=(high - low) / 2,
            mass=0.0,
            density=0.0,
            group=3,
            rgba=[0.9, 0.3, 0.1, 0.3],
            friction=[1, 0.005, 0.0001],
        )
        for own in original_bodies:
            spec.add_exclude(bodyname1=child.name, bodyname2=own)
        bounds[name] = dict(low=low.tolist(), high=high.tolist(), padding_m=0.0005)
    spec.add_exclude(
        bodyname1="jaw_soft_external_contact", bodyname2="trunk_base_external_contact"
    )
    return bounds


def build_game(config, seed):
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
    half = config.grid * config.pitch / 2 + 0.1
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


class GameSchedule:
    """Immutable evaluator plan: seeded perimeter-to-centre tie order.

    The cadence/countdown is public; the permutation and future times aren't.
    """

    def __init__(self, config, seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(config.ids).tolist()
        order.sort(key=lambda i: -max(abs(config.centre(i))))
        interval = 1.1 if config.cadence == "steady" else 0.85
        self.plan = [
            dict(
                tile=i,
                warning_at=2.0 + j * interval,
                release_at=2.0 + j * interval + config.warning_s,
            )
            for j, i in enumerate(order)
        ]
        self.emitted = set()

    def advance(self, data, tiles, now):
        events = []
        for item in self.plan:
            for state, key in (("WARNING", "warning_at"), ("RELEASED", "release_at")):
                token = item["tile"], state
                if now + 1e-9 >= item[key] and token not in self.emitted:
                    tile = tiles[item["tile"]]
                    tile.warn(now) if state == "WARNING" else tile.release(data, now)
                    self.emitted.add(token)
                    events.append(dict(time=now, tile=tile.id, state=state))
        return events


class ContactGraph:
    def __init__(self, model, ducks, tiles):
        self.tiles = {model.geom(f"tile_{i}_geom").id: i for i in tiles}
        self.owners = {}
        self.feet = {}
        for g in range(model.ngeom):
            body = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])
                )
                or ""
            )
            name = body.split("/")[0]
            if name in ducks:
                self.owners[g] = name
        for name in ducks:
            for side in ("left", "right"):
                self.feet[model.geom(name + "/" + side + "_foot_collision").id] = name

    def read(self, m, d, tiles):
        support = {name: set() for name in set(self.owners.values())}
        pairs, encounters = [], []
        force = np.zeros(6)
        for j in range(d.ncon):
            c = d.contact[j]
            a, b = int(c.geom1), int(c.geom2)
            mujoco.mj_contactForce(m, d, j, force)
            pairs.append(
                [
                    a,
                    b,
                    float(c.dist),
                    *force.tolist(),
                    *c.pos.tolist(),
                    *c.frame[:3].tolist(),
                ]
            )
            i = self.tiles.get(a, self.tiles.get(b))
            foot = self.feet.get(a, self.feet.get(b))
            if i is not None and foot is not None:
                up = c.frame[2] * (1 if b in self.feet else -1)
                if force[0] > 0.05 and up > 0.5 and d.eq_active[tiles[i].eq_id]:
                    support[foot].add(i)
            oa, ob = self.owners.get(a), self.owners.get(b)
            if oa and ob and oa != ob and force[0] > 0.05:
                encounters.append(
                    dict(
                        a=oa,
                        b=ob,
                        force_N=float(force[0]),
                        depth_m=max(0.0, -float(c.dist)),
                    )
                )
        return pairs, {n: sorted(v) for n, v in support.items()}, encounters


def observe(name, ducks, tiles, support, config, now, eliminated):
    duck = ducks[name]
    pos = duck.trunk_pos()
    current = (
        min(support[name], key=lambda i: np.linalg.norm(pos[:2] - config.centre(i)))
        if support[name]
        else None
    )
    return dict(
        schema="microduck.arena.observation.v2",
        sim_time_s=now,
        robot=dict(
            body_id=name,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            yaw=duck.trunk_yaw(),
            current_tile_id=current,
            speed_m_s=float(np.linalg.norm(duck.trunk_linvel()[:2])),
        ),
        visible_tiles=[
            dict(
                **tile.public(now),
                centre_xy=config.centre(i).tolist(),
                warning_remaining_s=None
                if tile.warning_at is None
                else max(0.0, config.warning_s - (now - tile.warning_at)),
            )
            for i, tile in tiles.items()
        ],
        nearby_ducks=[
            dict(
                body_id=n,
                x=float(o.trunk_pos()[0]),
                y=float(o.trunk_pos()[1]),
                vx=float(o.trunk_linvel()[0]),
                vy=float(o.trunk_linvel()[1]),
            )
            for n, o in ducks.items()
            if n != name and n not in eliminated
        ],
        grid=config.grid,
        objective="survive",
    )


class Survivor:
    """Finite adjacent legal choices, body-relative steering and public crowd cost."""

    def __init__(self):
        self.target = None
        self.last_tile = None

    def choose(self, obs):
        r = obs["robot"]
        pos = np.array([r["x"], r["y"]])
        tiles = {t["id"]: t for t in obs["visible_tiles"]}
        current = r["current_tile_id"]
        n = obs["grid"]

        def usable(i):
            t = tiles[i]
            return t["state"] == "LOCKED" or (
                t["state"] == "WARNING" and t["warning_remaining_s"] > 2.5
            )

        if self.target is not None and not usable(self.target):
            self.target = None
        if (
            self.target is None
            and current is not None
            and tiles[current]["state"] == "WARNING"
        ):
            candidates = [
                i
                for i, t in tiles.items()
                if usable(i)
                and abs(i // n - current // n) + abs(i % n - current % n) == 1
            ]

            def cost(i):
                centre = np.array(tiles[i]["centre_xy"])
                delta = centre - pos
                angle = math.atan2(delta[1], delta[0]) - r["yaw"]
                angle = abs(math.atan2(math.sin(angle), math.cos(angle)))
                crowd = sum(
                    max(0.0, 0.45 - np.linalg.norm(centre - np.array([o["x"], o["y"]])))
                    * 3
                    for o in obs["nearby_ducks"]
                )
                connectivity = sum(
                    t["state"] == "LOCKED"
                    and abs(j // n - i // n) + abs(j % n - i % n) == 1
                    for j, t in tiles.items()
                )
                warning_cost = 0.5 if tiles[i]["state"] == "WARNING" else 0.0
                return (
                    float(np.linalg.norm(delta))
                    + 0.08 * angle
                    + crowd
                    - 0.08 * connectivity
                    + 0.12 * np.linalg.norm(centre)
                    + warning_cost
                )

            if candidates:
                self.target = min(candidates, key=lambda i: (cost(i), i))
        if self.target is None:
            return (
                "stand",
                (0.0, 0.0, 0.0),
                "NO_SAFE_OPTION"
                if current is not None and tiles[current]["state"] == "WARNING"
                else "HOLD",
            )
        delta = np.array(tiles[self.target]["centre_xy"]) - pos
        dist = float(np.linalg.norm(delta))
        if dist < 0.045 and current == self.target:
            self.target = None
            return "stand", (0.0, 0.0, 0.0), "ARRIVED"
        angle = math.atan2(delta[1], delta[0]) - r["yaw"]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        vx = float(np.clip(2.5 * dist, 0.08, 0.45)) if abs(angle) < 0.45 else 0.0
        # Yield for a nearby robot ahead only when the current floor isn't urgent.
        ahead = any(
            np.linalg.norm(np.array([o["x"], o["y"]]) - pos) < 0.20
            and np.dot(np.array([o["x"], o["y"]]) - pos, delta) > 0
            for o in obs["nearby_ducks"]
        )
        urgent = current is None or tiles[current]["state"] == "WARNING"
        if ahead and not urgent:
            return "stand", (0.0, 0.0, 0.0), "YIELD"
        return "walk", (vx, 0.0, float(np.clip(3 * angle, -1.6, 1.6))), "GO_TO_TILE"


class GameReferee(Referee):
    """Winning requires a standing, supported survivor; a toppled body can't win."""

    def update(self, now, observations):
        events = super().update(now, observations)
        for name, r in observations.items():
            if not r["upright"]:
                self.support_since.pop(name, None)
        return events


def run_game(out, seed=101, config=None, capture=False):
    config = config or GameConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, ducks, tiles, spawns = build_game(config, seed)
    source_snapshot = {
        str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
        for p in [
            *Path(__file__).parent.glob("*.py"),
            Path(__file__).parent.parent / "sim/runtime.py",
            Path(__file__).parent.parent / "sim/composer.py",
        ]
    }
    policy_snapshot = {
        n: sha(p) for n, p in next(iter(ducks.values())).bank.paths.items()
    }
    graph = ContactGraph(m, ducks, tiles)
    schedule = GameSchedule(config, seed)
    controllers = {name: Survivor() for name in ducks}
    referee = GameReferee(tuple(ducks))
    events, decisions = [], []
    support = {name: [] for name in ducks}
    maxsteps = round(config.duration / config.dt)
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    states = controls = None
    if capture:
        mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)
        states = np.lib.format.open_memmap(
            out / "states.npy", mode="w+", dtype="float64", shape=(maxsteps, len(state))
        )
        controls = np.lib.format.open_memmap(
            out / "ctrl.npy", mode="w+", dtype="float64", shape=(maxsteps, m.nu)
        )
    depth_hist = np.zeros(100001, dtype=np.int64)
    worst_contacts = {}
    maxdepth = motor_max = locked_drift = 0.0
    resets = False
    finite = True
    body_contact_steps = 0
    peak_body_force = 0.0
    terminal_at = None
    terminal = None
    inference_s = physics_s = 0.0
    inference_batches = []
    start = time.monotonic()
    contact_file = gzip.open(out / "contacts.jsonl.gz", "wt") if capture else None
    try:
        for step in range(maxsteps):
            t = step * config.dt
            # Match-over stops future releases. Already falling bodies keep falling;
            # no equality is restored and no body state is changed.
            if terminal is None:
                events.extend(schedule.advance(d, tiles, t))
            if step % round(0.02 / config.dt) == 0:
                begin = time.monotonic()
                for name, duck in ducks.items():
                    obs = observe(
                        name, ducks, tiles, support, config, t, referee.eliminated
                    )
                    policy, command, intent = controllers[name].choose(obs)
                    if name in referee.eliminated:
                        # After a physical loss, hold measured joint positions:
                        # don't keep asking a standing policy to fight the receiver.
                        policy, command, intent = (
                            "joint_hold",
                            (0.0, 0.0, 0.0),
                            "ELIMINATED",
                        )
                    elif terminal is not None:
                        policy, command, intent = (
                            "stand",
                            (0.0, 0.0, 0.0),
                            "MATCH_COMPLETE",
                        )
                    duck.set_command(twist=command)
                    if policy == "joint_hold":
                        d.ctrl[duck.act_ids] = d.qpos[duck.joint_qpos_idx]
                    else:
                        duck.active_policy = policy
                        duck.step()
                    decisions.append(
                        dict(
                            time=t,
                            body_id=name,
                            input=obs,
                            policy=policy,
                            command=command,
                            intent=intent,
                            target=controllers[name].target,
                        )
                    )
                elapsed = time.monotonic() - begin
                inference_s += elapsed
                inference_batches.append(elapsed)
            if capture:
                mujoco.mj_getState(m, d, state, STATE)
                states[step] = state
                controls[step] = d.ctrl
            begin = time.monotonic()
            mujoco.mj_step(m, d)
            physics_s += time.monotonic() - begin
            pairs, support, encounters = graph.read(m, d, tiles)
            if contact_file:
                contact_file.write(
                    json.dumps(dict(step=step, time=t, contacts=pairs, support=support))
                    + "\n"
                )
            if encounters:
                body_contact_steps += 1
                peak_body_force = max(
                    peak_body_force, max(e["force_N"] for e in encounters)
                )
                if body_contact_steps == 1:
                    events.append(
                        dict(
                            time=t + config.dt,
                            state="BODY_CONTACT",
                            contacts=encounters,
                        )
                    )
            for p in pairs:
                depth = max(0.0, -p[2])
                maxdepth = max(maxdepth, depth)
                depth_hist[min(100000, math.ceil(depth * 1e6))] += 1
                oa, ob = graph.owners.get(p[0]), graph.owners.get(p[1])
                kind = (
                    "duck_duck"
                    if oa and ob and oa != ob
                    else "duck_self"
                    if oa and ob
                    else "duck_tile_or_receiver"
                    if oa or ob
                    else "tile_tile_or_receiver"
                )
                if depth > worst_contacts.get(kind, {}).get("depth_m", -1):
                    worst_contacts[kind] = dict(
                        time=t,
                        depth_m=depth,
                        geom_ids=p[:2],
                        body_names=[m.body(int(m.geom_bodyid[g])).name for g in p[:2]],
                    )
            events.extend(
                e for tile in tiles.values() for e in tile.update(d, t + config.dt)
            )
            observations = {
                name: dict(
                    z=float(duck.trunk_pos()[2]),
                    vz=float(duck.trunk_linvel()[2]),
                    supporting_tiles=support[name],
                    contact_step=step,
                    upright=bool(d.xmat[duck.trunk_body_id].reshape(3, 3)[2, 2] > 0.65),
                )
                for name, duck in ducks.items()
            }
            events.extend(referee.update(t + config.dt, observations))
            result = referee.outcome()
            if terminal is None and result["status"] in ("DRAW", "WINNER"):
                terminal, terminal_at = result, t + config.dt
                events.append(dict(time=terminal_at, state="TERMINAL", **terminal))
            motor_max = max(motor_max, float(np.max(np.abs(d.actuator_force))))
            locked_drift = max(
                locked_drift,
                max(
                    (
                        abs(d.qpos[tile.qpos_adr + 2] + 0.025)
                        for tile in tiles.values()
                        if d.eq_active[tile.eq_id]
                    ),
                    default=0.0,
                ),
            )
            resets |= abs(d.time - (t + config.dt)) > 1e-7
            finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            if terminal_at is not None and t + config.dt >= terminal_at + 2.0:
                break
    finally:
        if contact_file:
            contact_file.close()
    steps = step + 1
    if capture:
        states.flush()
        controls.flush()
        # Crop unused capacity on disk; no all-episode in-memory copy.
        np.savez_compressed(
            out / "trajectory.npz", states=states[:steps], ctrl=controls[:steps]
        )
        del states, controls
        (out / "states.npy").unlink()
        (out / "ctrl.npy").unlink()
    with gzip.open(out / "decisions.jsonl.gz", "wt") as f:
        for decision in decisions:
            f.write(json.dumps(decision) + "\n")
    total = depth_hist.sum()
    p99 = (
        int(np.searchsorted(np.cumsum(depth_hist), math.ceil(total * 0.99))) * 1e-6
        if total
        else 0.0
    )
    penetrating = depth_hist[1:].sum()
    penetrating_p99 = (
        (
            int(
                np.searchsorted(
                    np.cumsum(depth_hist[1:]), math.ceil(penetrating * 0.99)
                )
            )
            + 1
        )
        * 1e-6
        if penetrating
        else 0.0
    )
    mujoco.mj_getState(m, d, state, STATE)
    final_state_sha256 = hashlib.sha256(state.tobytes()).hexdigest()
    audit = dict(
        schema="microduck.arena.multiplayer.v1",
        seed=seed,
        config=asdict(config),
        steps=steps,
        duration=steps * config.dt,
        mujoco_version=mujoco.__version__,
        capture=capture,
        final_state_sha256=final_state_sha256,
        spawns=spawns,
        outcome=terminal or referee.outcome(timeout=True),
        events=events,
        oracle_schedule=schedule.plan,
        penetration_max_m=maxdepth,
        penetration_p99_m=p99,
        penetrating_contacts_p99_m=penetrating_p99,
        motor_force_max_Nm=motor_max,
        locked_tile_max_drift_m=locked_drift,
        finite=finite,
        time_reset=resets,
        solver_warnings=[int(w.number) for w in d.warning],
        external_forces_zero=bool(
            not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        ),
        body_contact_steps=body_contact_steps,
        peak_body_force_N=peak_body_force,
        worst_contacts=worst_contacts,
        collision_geometry=dict(
            native_meshes="convex hulls",
            native_self_contacts_preserved=True,
            external_envelopes={
                name: {
                    part: dict(
                        centre=m.geom_pos[
                            m.geom(name + "/" + part + "_external_envelope").id
                        ].tolist(),
                        half_size=m.geom_size[
                            m.geom(name + "/" + part + "_external_envelope").id
                        ].tolist(),
                    )
                    for part in ("jaw_soft", "trunk_base")
                }
                for name in ducks
            }
            if config.external_envelopes
            else {},
            added_mass_kg=0.0,
        ),
        performance=dict(
            real_time_factor=steps * config.dt / (time.monotonic() - start),
            physics_s=physics_s,
            inference_s=inference_s,
            physics_step_mean_ms=physics_s / steps * 1000,
            motor_batch_p50_ms=float(np.quantile(inference_batches, 0.5) * 1000),
            motor_batch_p95_ms=float(np.quantile(inference_batches, 0.95) * 1000),
        ),
        final_roots={
            name: d.qpos[duck.trunk_qpos_adr : duck.trunk_qpos_adr + 7].tolist()
            for name, duck in ducks.items()
        },
        policy_sha256=policy_snapshot,
        source_sha256=source_snapshot,
        hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
