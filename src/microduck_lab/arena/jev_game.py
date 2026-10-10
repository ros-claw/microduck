"""Live Jev survival game, separate from archived deterministic survival actors."""
from pathlib import Path
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import time
import mujoco
import numpy as np
from .episode import STATE, sha
from .world import ASSETS
from .multiplayer import ContactGraph, observe, GameReferee
from .relay_rumble import CHARACTERS
from .survival_world import build_survival_world
from .jev_config import JevGameConfig
from .progressive_collapse import ConnectedBeacons
from .jev_fairness import FairCollapse
from .jev_recovery import RecoveryMotor
from .grounded_victory import GroundedVictory
from .jev_tactics import LiveTactics, RecordedTactics


class GentlePlatform:
    """Only after the connected arena reaches three tiles, ramp real motors.

    Unlike the archived finale, no floors are batch-released. The sole survivor
    holds the currently measured platform angles, without a pose reset.
    """
    def __init__(self, config):
        self.config = config
        self.started = None
        self.hold = None

    def targets(self, now, angles, alive, active):
        if active > 3:
            return (0., 0.)
        if self.started is None:
            self.started = now
        if alive <= 1:
            if self.hold is None:
                self.hold = tuple(float(v) for v in angles)
            return self.hold
        elapsed = now-self.started
        ramp = min(1., elapsed/15.)
        ramp = ramp*ramp*(3-2*ramp)
        return (self.config.tilt_max_rad*ramp*math.sin(.8*elapsed),
                self.config.tilt_max_rad*ramp*.7*math.sin(.63*elapsed+.7))


def _actor_source_hashes():
    return {
        str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("jev_rumble.py"),
            Path(__file__).with_name("jev_fairness.py"),
            Path(__file__).with_name("jev_recovery.py"),
            Path(__file__).with_name("jev_config.py"),
            Path(__file__).with_name("grounded_victory.py"),
            Path(__file__).with_name("jev_tactics.py"),
            Path(__file__).with_name("progressive_collapse.py"),
            Path(__file__).with_name("survival_rumble.py"),
            Path(__file__).parent.parent / "jev/client.py",
            Path(__file__).with_name("multiplayer.py"),
            Path(__file__).with_name("relay_world.py"),
            Path(__file__).with_name("relay_rumble.py"),
            Path(__file__).with_name("survival_world.py"),
            Path(__file__).with_name("tiles.py"),
            Path(__file__).with_name("referee.py"),
            Path(__file__).with_name("world.py"),
            Path(__file__).with_name("episode.py"),
            Path(__file__).parent.parent / "sim/runtime.py",
            Path(__file__).parent.parent / "sim/composer.py",
        ]
    }


_LOADED_SOURCE_SNAPSHOT = _actor_source_hashes()


def run_jev_game(out, seed=61204, config=None, capture=False, tape=None):
    if _actor_source_hashes() != _LOADED_SOURCE_SNAPSHOT:
        raise RuntimeError("Actor sources changed after import; restart the worker")
    config = config or JevGameConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, ducks, tiles, spawns = build_survival_world(config, seed)
    next(iter(ducks.values())).bank.paths["sitstand"] = str(ASSETS / "microduck/policies/alpha_sitstand.onnx")
    source_snapshot = _LOADED_SOURCE_SNAPSHOT.copy()
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
    damage = FairCollapse(config, graph, next(iter(weights.values())))
    referee = GameReferee(tuple(ducks))
    claim = ConnectedBeacons(config, ducks)
    finale = GentlePlatform(config)
    live = (RecordedTactics(Path(tape), out / "jev-requests.json") if tape else LiveTactics(out / "jev-requests.json"))
    victory = GroundedVictory(config)
    arena_act = np.array([m.actuator("arena_" + a).id for a in ("roll", "pitch")])
    arena_qpos = np.array(
        [m.jnt_qposadr[m.joint("arena_" + a + "_joint").id] for a in ("roll", "pitch")]
    )
    robot_act = np.concatenate([duck.act_ids for duck in ducks.values()])
    controllers = {n: RecoveryMotor(CHARACTERS[n]["role"]) for n in ducks}
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
    contact_file = gzip.open(out / "contacts.jsonl.gz", "wt", compresslevel=1) if capture else None
    maxdepth = maxmotor = maxarena = 0.0
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
                arena_targets = finale.targets(
                    t, d.qpos[arena_qpos], len(ducks) - len(referee.eliminated),
                    sum(bool(d.eq_active[tile.eq_id]) for tile in tiles.values())
                )
                d.ctrl[arena_act] = arena_targets
                events.append(
                    dict(
                        time=t,
                        state="PLATFORM_MOTOR",
                        targets=list(arena_targets),
                        angles=d.qpos[arena_qpos].tolist(),
                    )
                )
                events.append(
                    dict(
                        time=t,
                        state="SCOREBOARD",
                        scores=claim.scores.copy(),
                        progress=claim.progress.copy(),
                        tile=12 if finale.started is not None else claim.goal,
                        finale_active=finale.started is not None,
                    )
                )
                public = {}
                for name, duck in ducks.items():
                    if name in referee.eliminated:
                        continue
                    obs = observe(name, ducks, tiles, support, config, t, referee.eliminated)
                    obs["robot"]["up_cos"] = float(d.xmat[duck.trunk_body_id, 8])
                    obs["robot"]["recovering"] = controllers[name].recovering
                    obs["robot"]["evacuating"] = controllers[name].evacuating
                    for tile in obs["visible_tiles"]:
                        tile["damage"] = damage.damage[tile["id"]]
                        tile["warning_remaining_s"] = damage.remaining(tile["id"], t)
                    obs.update(objective="LastDuckAlive", island=claim.goal,
                               beacon_scores=claim.scores.copy(), beacon_progress=claim.progress.copy(),
                               last_plan=live.plans.get(name))
                    public[name] = obs
                plans = live.update(t, public if terminal is None else {})
                for name, duck in ducks.items():
                    obs = public.get(name) or observe(name, ducks, tiles, support, config, t, referee.eliminated)
                    plan = plans.get(name)
                    if name in referee.eliminated:
                        policy, command, intent = "joint_hold", (0, 0, 0), "ELIMINATED"
                    elif terminal:
                        policy, command, intent = "stand", (0, 0, 0), "MATCH_COMPLETE"
                    else:
                        policy, command, intent = controllers[name].choose(obs, plan)
                        events.extend(controllers[name].events)
                        controllers[name].events.clear()
                    duck.set_command(twist=command)
                    if policy == "joint_hold":
                        d.ctrl[duck.act_ids] = d.qpos[duck.joint_qpos_idx]
                    else:
                        duck.active_policy = policy
                        duck.step()
                    obs["beacon_scores"] = claim.scores.copy()
                    decisions.append(
                        dict(
                            time=t,
                            body_id=name,
                            input=obs,
                            policy=policy,
                            command=command,
                            intent=intent,
                            target=controllers[name].target,
                            jev_request_id=plan["request_id"] if plan and intent.startswith("JEV_") and intent != "JEV_PLAN_INVALID_BRAKE" else None,
                            plan=plan,
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
            events.extend(damage.advance(d, tiles, pairs, t+config.dt, observations, referee.eliminated, terminal is None))
            events.extend(referee.update(t + config.dt, observations))
            if terminal is None:
                claim.update(t + config.dt, observations, referee.eliminated, tiles)
            events.extend(claim.events)
            claim.events.clear()
            outcome = (
                victory.update(t + config.dt, observations, referee.eliminated)
                if terminal is None
                else None
            )
            if terminal is None and outcome:
                terminal, terminal_at = outcome, t + config.dt
                events.append(dict(time=terminal_at, state="TERMINAL", **terminal))
            maxdepth = max(maxdepth, max((max(0, -p[2]) for p in pairs), default=0))
            maxmotor = max(maxmotor, float(np.max(np.abs(d.actuator_force[robot_act]))))
            maxarena = max(maxarena, float(np.max(np.abs(d.actuator_force[arena_act]))))
            finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            reset |= abs(d.time - (t + config.dt)) > 1e-7
            external_zero &= not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            if terminal_at is not None and t + config.dt >= terminal_at + 1.0:
                break
            # Never freeze physics for HTTP; throttle only when simulation is faster than wall time.
            if tape is None and step % 80 == 0:
                delay = t + config.dt - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        live.close()
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
    if _actor_source_hashes() != _LOADED_SOURCE_SNAPSHOT:
        raise RuntimeError(
            "Actor sources changed during the run; evidence is not sealed"
        )
    audit = dict(
        schema="microduck.jev-rumble.development.v4",
        tactical_backend="recorded_jev_delivery_tape" if tape else "live_jev_shared_batch",
        collapse_rules=dict(first_warning_s=damage.first_warning_s, warning_s=damage.warning_s, recovery_grace_s=damage.grace_s, max_pending_tiles=1, connectivity_preserved=True),
        jev_summary=dict(requests=len(live.rows), delivered=sum("delivered_s" in r for r in live.rows), accepted_plans=sum(sum(r.get("accepted", {}).values()) for r in live.rows), applied_decisions=sum(r.get("jev_request_id") is not None for r in decisions), models=sorted({r["raw"]["model"] for r in live.rows if "raw" in r})),
        seed=seed,
        config=asdict(config),
        capture=capture,
        steps=steps,
        duration=steps * config.dt,
        mujoco_version=mujoco.__version__,
        spawns=spawns,
        outcome=terminal
        or dict(
            status="TIME_LIMIT",
            winner=None,
            alive=[n for n in ducks if n not in referee.eliminated],
            rule="LastDuckAlive",
        ),
        events=events,
        damage=damage.damage,
        body_weight_N=weights,
        body_contact_steps=contact_steps,
        penetration_max_m=maxdepth,
        motor_force_max_Nm=maxmotor,
        arena_motor_force_max_Nm=maxarena,
        final_observations=observations,
        final_alive=[n for n in ducks if n not in referee.eliminated],
        finale_started_s=finale.started,
        finite=finite,
        time_reset=reset,
        external_forces_zero=bool(external_zero),
        solver_warnings=[int(w.number) for w in d.warning],
        final_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        stationary_fraction=sum(
            r["input"]["robot"]["speed_m_s"] < 0.03
            for r in decisions
            if r["intent"] not in ("ELIMINATED", "MATCH_COMPLETE")
        )
        / max(
            1,
            sum(r["intent"] not in ("ELIMINATED", "MATCH_COMPLETE") for r in decisions),
        ),
        stationary_metric="active nonterminal decisions, including initial settle",
        no_safe_option_fraction=sum(r["intent"] == "NO_SAFE_OPTION" for r in decisions)
        / len(decisions),
        roles={n: c.role for n, c in controllers.items()},
        characters=CHARACTERS,
        travelled_m={n: c.travel for n, c in controllers.items()},
        objective_replans={n: c.replans for n, c in controllers.items()},
        contested_s=claim.contested_steps * config.dt,
        points=claim.scores.copy(),
        policy_sha256=policy_snapshot,
        source_sha256=source_snapshot,
        performance=dict(
            real_time_factor=steps * config.dt / (time.monotonic() - start)
        ),
        hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
