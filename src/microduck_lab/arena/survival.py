"""DG-02 loop with auditable scheduler, public warning response and physical referee."""

import gzip
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from .world import build_world
from .episode import STATE, sha
from .tiles import bind_tiles, TileScheduler
from .controller import WarningResponder, observation
from .referee import Referee


def contacts(model, data, tiles):
    pairs = []
    support = set()
    force = np.zeros(6)
    tile_geoms = {model.geom(f"tile_{i}_geom").id: i for i in range(9)}
    feet = {
        model.geom("cream/" + side + "_foot_collision").id for side in ("left", "right")
    }
    for j in range(data.ncon):
        c = data.contact[j]
        a, b = int(c.geom1), int(c.geom2)
        mujoco.mj_contactForce(model, data, j, force)
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
        tile = tile_geoms.get(a, tile_geoms.get(b))
        if tile is not None and (a in feet or b in feet):
            upward = c.frame[2] * (1 if b in feet else -1)
            if force[0] > 0.05 and upward > 0.5 and data.eq_active[tiles[tile].eq_id]:
                support.add(tile)
    return pairs, sorted(support)


def run(out, seed=11, duration=22.0, mode="seeded", brain="warning", dt=0.0005):
    if brain not in ("warning", "hold"):
        raise ValueError("Supported brains: warning, hold")
    if not np.isfinite(duration) or not 7 <= duration <= 48:
        raise ValueError("Duration must be 7–48 s")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, duck = build_world(dt)
    d.qpos[duck.joint_qpos_idx] += np.random.default_rng(seed).uniform(
        -0.005, 0.005, 14
    )
    mujoco.mj_forward(m, d)
    mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)
    tiles = bind_tiles(m)
    scheduler = TileScheduler(seed, mode=mode)
    referee = Referee()
    controller = WarningResponder()
    states = []
    controls = []
    eqs = []
    times = []
    roots = []
    zs = []
    supports = []
    events = []
    decisions = []
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    depths = []
    motor_max = 0.0
    time_reset = False
    locked_drift = 0.0
    energy = []
    m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    # Save the final option flags in the binary used for strict replay.
    mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)
    start = time.monotonic()
    support = []
    terminal_at = None
    with gzip.open(out / "contacts.jsonl.gz", "wt") as f:
        for step in range(round(duration / dt)):
            t = step * dt
            events.extend(scheduler.advance(d, tiles, t))
            if step % round(0.02 / dt) == 0:
                obs = observation(duck, tiles, support, t)
                policy, command, intent = (
                    controller.choose(obs)
                    if brain == "warning"
                    else ("stand", (0.0, 0.0, 0.0), "HOLD")
                )
                if terminal_at is not None:
                    policy, command, intent = "stand", (0.0, 0.0, 0.0), "ELIMINATED"
                duck.active_policy = policy
                duck.set_command(twist=command)
                duck.step()
                decisions.append(
                    dict(
                        time=t,
                        input=obs,
                        policy=policy,
                        command=command,
                        intent=intent,
                        target=controller.target,
                    )
                )
            mujoco.mj_getState(m, d, state, STATE)
            states.append(state.copy())
            controls.append(d.ctrl.copy())
            eqs.append(d.eq_active.copy())
            mujoco.mj_step(m, d)
            pairs, support = contacts(m, d, tiles)
            f.write(
                json.dumps(
                    dict(step=step, t=t, contacts=pairs, supporting_tiles=support)
                )
                + "\n"
            )
            events.extend(event for tile in tiles for event in tile.update(d, t + dt))
            robot = dict(
                z=float(duck.trunk_pos()[2]),
                vz=float(duck.trunk_linvel()[2]),
                supporting_tiles=support,
                contact_step=step,
            )
            new = referee.update(t + dt, {"cream": robot})
            events.extend(new)
            if new and terminal_at is None:
                terminal_at = t + dt
            for tile in tiles:
                if d.eq_active[tile.eq_id]:
                    locked_drift = max(
                        locked_drift, abs(d.qpos[tile.qpos_adr + 2] + 0.025)
                    )
            motor_max = max(motor_max, float(np.max(np.abs(d.actuator_force))))
            time_reset |= abs(d.time - (t + dt)) > 1e-7
            depths.extend(max(0.0, -p[2]) for p in pairs)
            times.append(t)
            roots.append(d.qpos[duck.trunk_qpos_adr : duck.trunk_qpos_adr + 7].copy())
            zs.append([d.qpos[tile.qpos_adr + 2] for tile in tiles])
            supports.append(4 in support)
            energy.append(d.energy.copy())
            # Two physical seconds after elimination preserve landing/failure evidence.
            if terminal_at is not None and t + dt >= terminal_at + 2.0:
                break
    np.savez_compressed(
        out / "trajectory.npz",
        states=states,
        ctrl=controls,
        eq_active=eqs,
        time=times,
        root=roots,
        tile_z=zs,
        support=supports,
        energy=energy,
    )
    with gzip.open(out / "decisions.jsonl.gz", "wt") as f:
        for decision in decisions:
            f.write(json.dumps(decision) + "\n")
    outcome = referee.outcome(timeout=True)
    audit = dict(
        schema="microduck.arena.scheduled.v1",
        seed=seed,
        dt=dt,
        duration=len(states) * dt,
        mujoco_version=mujoco.__version__,
        brain=brain,
        mode=mode,
        outcome=outcome,
        events=events,
        decisions_file="decisions.jsonl.gz",
        oracle_schedule=scheduler.plan,
        penetration_max_m=max(depths, default=0.0),
        penetration_p99_m=float(np.quantile(depths, 0.99)),
        motor_force_max_Nm=motor_max,
        locked_tile_max_drift_m=locked_drift,
        time_reset=time_reset,
        solver_warnings=[int(w.number) for w in d.warning],
        finite=bool(np.isfinite(states).all()),
        external_forces_zero=bool(
            not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        ),
        recording_real_time_factor=len(states) * dt / (time.monotonic() - start),
        policy_sha256={n: sha(p) for n, p in duck.bank.paths.items()},
        source_sha256={
            str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
            for p in Path(__file__).parent.glob("*.py")
        },
        hashes={
            p.name: sha(p)
            for p in [
                out / "scene.mjb",
                out / "trajectory.npz",
                out / "contacts.jsonl.gz",
                out / "decisions.jsonl.gz",
            ]
        },
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
