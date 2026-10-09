"""Island Rumble prototype: real loaded foot contacts consume the floor.

Separate ruleset from the immutable Last Duck Standing release. No pose helpers.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import time
import mujoco
import numpy as np
from .multiplayer import GameConfig, build_game, ContactGraph, observe, GameReferee
from .episode import STATE, sha


@dataclass(frozen=True)
class RumbleConfig(GameConfig):
    players: int = 2
    grid: int = 3
    tile_size: float = 0.36
    gap: float = 0.008
    duration: float = 20.0
    lifetime_s: float = 3.2
    filter_s: float = 0.12
    final_at_s: float = 6.0
    claim_s: float = 1.5
    passive: bool = False
    claim_radius_m: float = 0.0

    def __post_init__(self):
        if self.players not in (2, 4) or self.grid not in (3, 4):
            raise ValueError("Prototype requires two/four ducks on 3x3/4x4")
        if not (0.30 <= self.tile_size <= 0.48 and 0.005 <= self.gap <= 0.012):
            raise ValueError("Uncalibrated compact dimensions")
        if self.layout != "square" or self.dt != 0.00025:
            raise ValueError("Use square arena at the frozen 4 kHz contact settings")
        if (
            not 10 <= self.duration <= 45
            or not 2 <= self.lifetime_s <= 10
            or not 0.02 <= self.filter_s <= 0.5
            or not 0 <= self.final_at_s < self.duration
            or not 0.5 <= self.claim_s <= 5
            or self.contact_ref_s != 0.0005
            or self.solver_iterations != 80
            or not self.external_envelopes
            or self.ccd != "libccd"
            or not 0 <= self.claim_radius_m <= 0.08
        ):
            raise ValueError("Invalid rules or changed physical calibration")

    @property
    def island(self):
        # 3x3 centre is equally accessible from all cardinal spawns. 4x4 is
        # calibration-only, not qualified fair IslandClaim (no unique centre).
        return self.grid * (self.grid // 2) + self.grid // 2


class ContactTriggeredCollapse:
    """Low-pass vertical *actual foot loads*, integrated in body-weight seconds.

    No occupancy oracle. Total peak clipped at 3 body weights before filtering.
    Two equally loaded ducks accumulate damage twice as fast. Safe island exempt.
    """

    def __init__(self, config, graph, body_weight):
        self.config, self.graph, self.weight = config, graph, body_weight
        self.filtered = {i: 0.0 for i in config.ids}
        self.damage = {i: 0.0 for i in config.ids}
        self.loaded_once = set()
        self.last_emit = {i: 0 for i in config.ids}

    def advance(self, d, tiles, pairs, now, enabled=True):
        if not enabled:
            return []
        loads = {i: 0.0 for i in tiles}
        owners = {i: set() for i in tiles}
        for p in pairs:
            a, b = p[:2]
            i = self.graph.tiles.get(a, self.graph.tiles.get(b))
            owner = self.graph.feet.get(a, self.graph.feet.get(b))
            if i is None or owner is None or not d.eq_active[tiles[i].eq_id]:
                continue
            up = p[12 + 2] * (1 if b in self.graph.feet else -1)
            if p[3] > 0.05 and up > 0.5:
                loads[i] += p[3] * up
                owners[i].add(owner)
        return self.integrate(d, tiles, loads, owners, now)

    def integrate(self, d, tiles, loads, owners, now):
        events = []
        alpha = -math.expm1(-self.config.dt / self.config.filter_s)
        for i, tile in tiles.items():
            if i == self.config.island or not d.eq_active[tile.eq_id]:
                continue
            load = min(3 * self.weight, max(0, loads[i]))
            self.filtered[i] += alpha * (load - self.filtered[i])
            if load > 0.05 and i not in self.loaded_once:
                self.loaded_once.add(i)
                events.append(
                    dict(time=now, tile=i, state="LOADED", owners=sorted(owners[i]))
                )
            self.damage[i] += (
                self.filtered[i] / self.weight * self.config.dt / self.config.lifetime_s
            )
            if self.damage[i] >= 0.02 and tile.state == "LOCKED":
                tile.warn(now)
                events.append(
                    dict(time=now, tile=i, state="CRACKING", damage=self.damage[i])
                )
            level = min(10, int(self.damage[i] * 10))
            if level > self.last_emit[i]:
                self.last_emit[i] = level
                events.append(
                    dict(
                        time=now,
                        tile=i,
                        state="DAMAGE",
                        damage=self.damage[i],
                        load_N=self.filtered[i],
                        owners=sorted(owners[i]),
                    )
                )
            if self.damage[i] >= 1 and tile.state == "WARNING":
                tile.release(d, now)
                events.append(
                    dict(time=now, tile=i, state="RELEASED", damage=self.damage[i])
                )
        return events


class Tactician:
    """Current public state only; walk turns/forward via the frozen ONNX policy."""

    def __init__(self, role="rusher"):
        self.role, self.target = role, None
        self.last_arrived = None
        self.hold_until = 0.0

    def choose(self, obs):
        r, now = obs["robot"], obs["sim_time_s"]
        pos = np.array([r["x"], r["y"]])
        ts = {t["id"]: t for t in obs["visible_tiles"]}
        current, n, goal = r["current_tile_id"], obs["grid"], obs["island"]
        if obs["passive"] or now < 0.8:
            return "stand", (0, 0, 0), "PASSIVE" if obs["passive"] else "SETTLE"
        safe = {
            i
            for i, t in ts.items()
            if t["state"] in ("LOCKED", "WARNING") and t["damage"] < 0.80
        }
        safe.add(goal)
        anchor = (
            current
            if current is not None
            else min(ts, key=lambda i: np.linalg.norm(pos - ts[i]["centre_xy"]))
        )

        # Public BFS: next cardinal step, never a straight-line shortcut over gaps.
        def route(dest):
            queue = [(anchor, [])]
            seen = {anchor}
            for node, path in queue:
                if node == dest:
                    return path
                for j in sorted(safe):
                    if (
                        j not in seen
                        and abs(j // n - node // n) + abs(j % n - node % n) == 1
                    ):
                        seen.add(j)
                        queue.append((j, path + [j]))
            return None

        if obs["final_active"]:
            destination = goal
        else:
            # Before Final Island: explore a loaded neighbour rather than waiting.
            candidates = [
                i
                for i in safe
                if (i != goal or (obs.get("crown_mode") and self.role != "survivor"))
                and i != anchor
                and route(i)
            ]
            if not candidates:
                destination = goal
            else:

                def cost(i):
                    c = np.array(ts[i]["centre_xy"])
                    crowd = sum(
                        max(0, 0.34 - np.linalg.norm(c - [o["x"], o["y"]]))
                        for o in obs["nearby_ducks"]
                    )
                    predicted = [
                        np.array([o["x"] + 0.65 * o["vx"], o["y"] + 0.65 * o["vy"]])
                        for o in obs["nearby_ducks"]
                    ]
                    intercept = min(
                        (np.linalg.norm(c - p) for p in predicted), default=0
                    )
                    radial = np.linalg.norm(c)
                    return (
                        len(route(i))
                        - (
                            3.0
                            if obs.get("crown_mode")
                            and i == goal
                            and self.role == "rusher"
                            else 0
                        )
                        + ts[i]["damage"] * 2
                        + (3 * crowd if self.role == "survivor" else -crowd)
                        + (radial if self.role == "rusher" else 0)
                        + (intercept if self.role == "blocker" else 0)
                        + (0.6 if i == self.last_arrived else 0)
                    )

                destination = min(candidates, key=lambda i: (cost(i), i))
        path = route(destination)
        if self.target not in safe or obs["final_active"] or self.target is None:
            self.target = (
                (path[0] if path else destination) if path is not None else None
            )
        if self.target is None:
            return "stand", (0, 0, 0), "NO_SAFE_OPTION"
        delta = np.array(ts[self.target]["centre_xy"]) - pos
        dist = float(np.linalg.norm(delta))
        if (
            dist < (obs["claim_radius_m"] - 0.005 if obs.get("crown_mode") else 0.035)
            and current == self.target
        ):
            arrived = self.target
            if arrived == goal and obs["final_active"]:
                return "stand", (0, 0, 0), "CLAIM_ISLAND"
            self.last_arrived = arrived
            self.target = None
            return "stand", (0, 0, 0), "ARRIVED"
        angle = math.atan2(delta[1], delta[0]) - r["yaw"]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        vx = float(np.clip(2.5 * dist, 0.08, 0.45)) if abs(angle) < 0.45 else 0.0
        return (
            "walk",
            (vx, 0, float(np.clip(3 * angle, -1.6, 1.6))),
            ("RUSH_ISLAND" if obs["final_active"] else "EXPLORE"),
        )


class IslandClaim:
    """Exclusive real loaded support + upright debounced claim; never identity ties."""

    def __init__(self, config):
        self.config, self.candidate, self.since = config, None, None

    def update(self, now, observations, eliminated):
        occupants = [
            n
            for n, o in observations.items()
            if n not in eliminated
            and self.config.island in o["supporting_tiles"]
            and (
                self.config.claim_radius_m == 0
                or math.hypot(
                    o["x"] - self.config.centre(self.config.island)[0],
                    o["y"] - self.config.centre(self.config.island)[1],
                )
                <= self.config.claim_radius_m
            )
        ]
        if (
            now < self.config.final_at_s
            or len(occupants) != 1
            or not observations[occupants[0]]["upright"]
        ):
            self.candidate = self.since = None
            return None
        eligible = occupants
        n = eligible[0]
        if n != self.candidate:
            self.candidate, self.since = n, now
        if now - self.since + 1e-9 >= self.config.claim_s:
            return dict(
                status="WINNER",
                winner=n,
                alive=[k for k in observations if k not in eliminated],
                rule="CrownClaim" if self.config.claim_radius_m else "IslandClaim",
                exclusive_supported_s=now - self.since,
            )
        return None


class CrownClaim:
    """Public centre zone; stable loaded occupancy tolerates <=10 ms micro-gaps.

    At least 90% of the continuous upright 1.5s hold must have actual loaded feet.
    Walking micro-contact interruptions are not equivalent to a visible jump.
    A rival inside the zone contests it even when momentarily airborne.
    """

    max_gap_s = 0.01
    min_support_fraction = 0.90

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = self.last_time = self.last_loaded = None
        self.loaded_s = self.max_gap = 0.0

    def update(self, now, observations, eliminated):
        centre = self.config.centre(self.config.island)
        occupants = [
            n
            for n, o in observations.items()
            if n not in eliminated
            and o.get("z", 0.12) >= -0.12
            and math.hypot(o["x"] - centre[0], o["y"] - centre[1])
            <= self.config.claim_radius_m
        ]
        if (
            now < self.config.final_at_s
            or len(occupants) != 1
            or not observations[occupants[0]]["upright"]
        ):
            self.reset()
            return None
        n = occupants[0]
        loaded = self.config.island in observations[n]["supporting_tiles"]
        if n != self.candidate:
            self.reset()
            if not loaded:
                return None
            self.candidate = n
            self.since = self.last_time = self.last_loaded = now
        elapsed = now - self.last_time
        self.last_time = now
        if loaded:
            self.loaded_s += elapsed
            self.last_loaded = now
        gap = now - self.last_loaded
        self.max_gap = max(self.max_gap, gap)
        if gap > self.max_gap_s + 1e-9:
            self.reset()
            return None
        duration = now - self.since
        fraction = self.loaded_s / duration if duration else 1.0
        if (
            loaded
            and duration + 1e-9 >= self.config.claim_s
            and fraction >= self.min_support_fraction
        ):
            return dict(
                status="WINNER",
                winner=n,
                alive=[k for k in observations if k not in eliminated],
                rule="CrownClaim",
                exclusive_zone_hold_s=duration,
                measured_loaded_fraction=fraction,
                max_unsupported_gap_s=self.max_gap,
                required_loaded_fraction=self.min_support_fraction,
                allowed_unsupported_gap_s=self.max_gap_s,
            )
        return None


def run_rumble(out, seed=31001, config=None, capture=False):
    config = config or RumbleConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, ducks, tiles, spawns = build_game(config, seed)
    source_snapshot = {
        str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("multiplayer.py"),
            Path(__file__).with_name("tiles.py"),
            Path(__file__).with_name("referee.py"),
            Path(__file__).with_name("world.py"),
            Path(__file__).parent.parent / "sim/runtime.py",
            Path(__file__).parent.parent / "sim/composer.py",
        ]
    }
    policy_snapshot = {
        n: sha(p) for n, p in next(iter(ducks.values())).bank.paths.items()
    }
    graph = ContactGraph(m, ducks, tiles)
    weights = {
        n: float(
            sum(
                m.body_mass[i]
                for i in range(m.nbody)
                if m.body(i).name.startswith(n + "/")
            )
            * 9.81
        )
        for n in ducks
    }
    damage = ContactTriggeredCollapse(config, graph, next(iter(weights.values())))
    referee = GameReferee(tuple(ducks))
    claim = CrownClaim(config) if config.claim_radius_m else IslandClaim(config)
    controllers = {
        n: Tactician(("rusher", "survivor", "blocker", "rusher")[j])
        for j, n in enumerate(ducks)
    }
    events, decisions = [], []
    support = {n: [] for n in ducks}
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
    contact_file = gzip.open(out / "contacts.jsonl.gz", "wt") if capture else None
    maxdepth = maxmotor = 0.0
    finite, reset, external_zero = True, False, True
    contact_steps = 0
    contact_active = False
    last_contact_event = -1.0
    terminal = terminal_at = None
    start = time.monotonic()
    try:
        for step in range(maxsteps):
            t = step * config.dt
            if step % 80 == 0:
                for name, duck in ducks.items():
                    obs = observe(
                        name, ducks, tiles, support, config, t, referee.eliminated
                    )
                    for tile in obs["visible_tiles"]:
                        tile["damage"] = damage.damage[tile["id"]]
                        # No fixed countdown exists in this mode.
                        tile["warning_remaining_s"] = None
                    obs.update(
                        objective="CrownClaim"
                        if config.claim_radius_m
                        else "IslandClaim",
                        island=config.island,
                        final_active=t >= config.final_at_s,
                        passive=config.passive,
                        crown_mode=config.claim_radius_m > 0,
                        claim_radius_m=config.claim_radius_m,
                    )
                    policy, command, intent = controllers[name].choose(obs)
                    if name in referee.eliminated:
                        policy, command, intent = "joint_hold", (0, 0, 0), "ELIMINATED"
                    elif terminal:
                        policy, command, intent = "stand", (0, 0, 0), "MATCH_COMPLETE"
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
            if capture:
                mujoco.mj_getState(m, d, state, STATE)
                states[step], controls[step] = state, d.ctrl
            mujoco.mj_step(m, d)
            pairs, support, encounters = graph.read(m, d, tiles)
            if contact_file:
                contact_file.write(
                    json.dumps(dict(step=step, time=t, contacts=pairs, support=support))
                    + "\n"
                )
            if encounters:
                contact_steps += 1
                if not contact_active and t - last_contact_event >= 0.25:
                    last_contact_event = t
                    events.append(
                        dict(
                            time=t + config.dt,
                            state="BODY_CONTACT",
                            contacts=encounters,
                        )
                    )
            contact_active = bool(encounters)
            events.extend(
                damage.advance(d, tiles, pairs, t + config.dt, terminal is None)
            )
            events.extend(
                e for tile in tiles.values() for e in tile.update(d, t + config.dt)
            )
            observations = {
                n: dict(
                    x=float(o.trunk_pos()[0]),
                    y=float(o.trunk_pos()[1]),
                    z=float(o.trunk_pos()[2]),
                    vz=float(o.trunk_linvel()[2]),
                    supporting_tiles=support[n],
                    contact_step=step,
                    upright=bool(d.xmat[o.trunk_body_id].reshape(3, 3)[2, 2] > 0.65),
                )
                for n, o in ducks.items()
            }
            events.extend(referee.update(t + config.dt, observations))
            outcome = claim.update(t + config.dt, observations, referee.eliminated)
            survival = referee.outcome()
            if not outcome and survival["status"] == "DRAW":
                outcome = dict(**survival, rule="LastSupportedSurvivor")
            if terminal is None and outcome:
                terminal, terminal_at = outcome, t + config.dt
                events.append(dict(time=terminal_at, state="TERMINAL", **terminal))
            maxdepth = max(maxdepth, max((max(0, -p[2]) for p in pairs), default=0))
            maxmotor = max(maxmotor, float(np.max(np.abs(d.actuator_force))))
            finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            reset |= abs(d.time - (t + config.dt)) > 1e-7
            external_zero &= not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            if terminal_at is not None and t + config.dt >= terminal_at + 1.0:
                break
    finally:
        if contact_file:
            contact_file.close()
    steps = step + 1
    if capture:
        states.flush()
        controls.flush()
        np.savez_compressed(
            out / "trajectory.npz", states=states[:steps], ctrl=controls[:steps]
        )
        del states, controls
        (out / "states.npy").unlink()
        (out / "ctrl.npy").unlink()
    with gzip.open(out / "decisions.jsonl.gz", "wt") as f:
        for row in decisions:
            f.write(json.dumps(row) + "\n")
    mujoco.mj_getState(m, d, state, STATE)
    audit = dict(
        schema="microduck.island-rumble.prototype.v1",
        seed=seed,
        config=asdict(config),
        capture=capture,
        steps=steps,
        duration=steps * config.dt,
        mujoco_version=mujoco.__version__,
        spawns=spawns,
        outcome=terminal
        or dict(
            **referee.outcome(timeout=True),
            rule="CrownClaim" if config.claim_radius_m else "IslandClaim",
        ),
        events=events,
        damage=damage.damage,
        body_weight_N=weights,
        body_contact_steps=contact_steps,
        penetration_max_m=maxdepth,
        motor_force_max_Nm=maxmotor,
        finite=finite,
        time_reset=reset,
        external_forces_zero=bool(external_zero),
        solver_warnings=[int(w.number) for w in d.warning],
        final_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        stationary_fraction=sum(
            r["input"]["robot"]["speed_m_s"] < 0.03 for r in decisions
        )
        / len(decisions),
        no_safe_option_fraction=sum(r["intent"] == "NO_SAFE_OPTION" for r in decisions)
        / len(decisions),
        roles={n: c.role for n, c in controllers.items()},
        policy_sha256=policy_snapshot,
        source_sha256=source_snapshot,
        performance=dict(
            real_time_factor=steps * config.dt / (time.monotonic() - start)
        ),
        hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
