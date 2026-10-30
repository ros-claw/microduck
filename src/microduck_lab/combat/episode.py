"""Single shared clock, synchronous decisions, auditable motor-only execution."""

from pathlib import Path
from dataclasses import asdict
import gzip
import json
import time
import mujoco
import numpy as np
from .world import build, digest, model_inventory, ASSETS, ROBOT, POLICIES, ROOT
from .contacts import ContactAudit
from .observations import snapshot
from .maneuver import Maneuver, MotorCommand
from .referee import Referee

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def source_hashes():
    paths = list((ROOT / "src/microduck_lab/combat").glob("*.py"))
    paths += [ROOT / "src/microduck_lab/sim" / n for n in ("runtime.py", "composer.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def run(out, config, physics, seed=101, capture=False, probe=None):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Invalid seed")
    if probe is not None and (len(probe) != 3 or not np.isfinite(probe).all()):
        raise ValueError("Probe must be a finite three-dimensional twist")
    if (
        config.candidate_motor_path
        and digest(config.candidate_motor_path) != config.candidate_motor_sha256
    ):
        raise ValueError("Candidate motor digest differs")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = source_hashes()
    m, d, ducks, spawns = build(config, physics, seed)
    if config.candidate_motor_path:
        next(iter(ducks.values())).bank.paths["dash_candidate"] = (
            config.candidate_motor_path
        )
    audit = ContactAudit(m, ducks)
    controllers = {n: Maneuver(n, spawns[n]["role"], config) for n in ducks}
    referee = Referee(ducks, config.down_count_s)
    nsteps = round(config.duration_s / physics.dt)
    nstate = mujoco.mj_stateSize(m, STATE)
    state = np.empty(nstate)
    mujoco.mj_getState(m, d, state, STATE)
    np.savez_compressed(out / "initial_state.npz", state=state)
    inventory = model_inventory(m)
    write_json(out / "model_inventory.json", inventory)
    invariant_arrays = [
        m.body_mass,
        m.body_inertia,
        m.geom_friction,
        m.actuator_gainprm,
        m.actuator_biasprm,
        m.actuator_forcerange,
        m.geom_contype,
        m.geom_conaffinity,
        m.exclude_signature,
    ]
    invariant_bytes = [v.tobytes() for v in invariant_arrays]
    if capture:
        mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)
        states = np.lib.format.open_memmap(
            out / "states.npy", mode="w+", dtype="float64", shape=(nsteps, nstate)
        )
    else:
        states = None
    contact_file = (
        gzip.open(out / "contacts_full.jsonl.gz", "wt", compresslevel=1)
        if capture
        else None
    )
    decision_file = gzip.open(
        out / "tactical_decisions.jsonl.gz", "wt", compresslevel=1
    )
    controls, body_samples, phase_samples = [], [], []
    energy = dict.fromkeys(ducks, 0.0)
    saturation_count = dict.fromkeys(ducks, 0)
    attacks = {}
    inference_times, decision_times = [], []
    maximum_torque = 0.0
    external_zero, finite, clock_ok = True, True, True
    wall_start = time.perf_counter()
    inference_elapsed = 0.0
    completed = 0
    try:
        for step in range(nsteps):
            now = step * physics.dt
            if step % physics.substeps == 0:
                before = time.perf_counter()
                # All controllers receive this same pre-action snapshot. They
                # cannot see controls, phases, future commands or evaluator seed.
                public = snapshot(ducks, audit, config)
                commands = {}
                for n in ducks:
                    other = next(public[o] for o in ducks if o != n)
                    commands[n] = controllers[n].choose(now, public[n], other)
                    if (
                        config.candidate_motor_path
                        and controllers[n].skill == "dash_bump"
                        and commands[n].owner == "maneuver"
                    ):
                        cmd = commands[n]
                        commands[n] = MotorCommand(
                            cmd.owner, "dash_candidate", cmd.twist
                        )
                    if probe is not None:
                        commands[n] = (
                            MotorCommand("command_range_probe", "walk", tuple(probe))
                            if spawns[n]["role"] == "coach_attacker"
                            and 1 <= now < config.duration_s - 1
                            else MotorCommand("command_range_probe")
                        )
                    if n in referee.eliminated:
                        commands[n] = MotorCommand("eliminated_hold")
                    row = dict(
                        sim_step=step,
                        time_s=now,
                        actor=n,
                        observation=asdict(public[n]),
                        opponent=asdict(other),
                        command=asdict(commands[n]),
                        phase=controllers[n].phase,
                        skill=controllers[n].skill,
                        attack_id=controllers[n].active_attack,
                        observation_contact_age_s=physics.dt,
                    )
                    decision_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                decision_times.append(time.perf_counter() - before)
                # Unique motor writer per duck. All inference before physics.
                for n, cmd in commands.items():
                    duck = ducks[n]
                    if cmd.owner == "eliminated_hold":
                        continue
                    duck.active_policy = cmd.policy
                    duck.set_command(twist=cmd.twist, head=cmd.head)
                    start = time.perf_counter()
                    duck.step()
                    elapsed = time.perf_counter() - start
                    inference_times.append(elapsed)
                    inference_elapsed += elapsed
                    # No overrides; last_action is precisely the deployed target offset.
                    if not np.array_equal(
                        d.ctrl[duck.act_ids],
                        (duck.default_pose + duck.last_action).astype(np.float64),
                    ):
                        raise RuntimeError("Motor history differs from executed target")
                controls.append(d.ctrl.copy())
                body_samples.append(
                    [
                        [
                            *public[n].xy,
                            public[n].z,
                            public[n].upright,
                            public[n].speed,
                            public[n].yaw,
                            float(public[n].foot_support),
                        ]
                        for n in ducks
                    ]
                )
                phase_samples.append([controllers[n].phase for n in ducks])
                attacks = {
                    n: controllers[n].active_attack
                    for n in ducks
                    if n not in referee.eliminated
                }
            if states is not None:
                mujoco.mj_getState(m, d, state, STATE)
                states[step] = state
            mujoco.mj_step(m, d)
            rows = audit.read(m, d, step, physics.dt, attacks)
            if contact_file is not None:
                contact_file.write(
                    json.dumps([step, rows], separators=(",", ":")) + "\n"
                )
            maximum_torque = max(
                maximum_torque, float(np.max(np.abs(d.actuator_force)))
            )
            external_zero &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
            clock_ok &= abs(d.time - (now + physics.dt)) < 1e-7
            for n, duck in ducks.items():
                torque = d.actuator_force[duck.act_ids]
                energy[n] += (
                    float(np.sum(np.maximum(0.0, torque * d.qvel[duck.joint_qvel_idx])))
                    * physics.dt
                )
                saturation_count[n] += int(
                    np.count_nonzero(np.abs(torque) >= physics.torque_limit_Nm - 1e-6)
                )
            completed = step + 1
            if completed % physics.substeps == 0:
                finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
                public = snapshot(ducks, audit, config)
                fault = not (finite and clock_ok and external_zero) or bool(
                    any(w.number for w in d.warning)
                )
                referee.update(completed * physics.dt, public, physics_fault=fault)
                if referee.result:
                    break
    finally:
        decision_file.close()
        if contact_file is not None:
            contact_file.close()
    wall_s = time.perf_counter() - wall_start
    if source != source_hashes():
        raise RuntimeError("Source changed during simulation; discard this run")
    if (
        config.candidate_motor_path
        and digest(config.candidate_motor_path) != config.candidate_motor_sha256
    ):
        raise RuntimeError("Candidate motor changed during simulation")
    immutable = all(
        v.tobytes() == old
        for v, old in zip(invariant_arrays, invariant_bytes, strict=True)
    )
    np.savez_compressed(
        out / "controls.npz",
        ctrl=np.array(controls),
        body=np.array(body_samples),
        phases=np.array(phase_samples),
    )
    if states is not None:
        states.flush()
    contacts = audit.summary()
    write_json(out / "contacts_summary.json", contacts)
    all_events = (
        [e for c in controllers.values() for e in c.events]
        + audit.events
        + referee.events
    )
    # Attribute attack results after contact episode consolidation. CONTACTED is
    # deliberately not VALID_STRIKE: its effectiveness remains unqualified.
    linked = {a for e in audit.episodes for a in e["attacks"]}
    for e in list(all_events):
        if e["state"] == "ATTACK_WINDOW_ENDED":
            all_events.append(
                dict(
                    state="CONTACTED_ATTACK" if e["attack_id"] in linked else "MISS",
                    time_s=e["time_s"],
                    actor=e["actor"],
                    attack_id=e["attack_id"],
                )
            )
    all_events.sort(key=lambda e: e.get("time_s", 0.0))
    with open(out / "runtime_events.jsonl", "w") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    metrics = {}
    body_array = np.array(body_samples)
    for i, n in enumerate(ducks):
        v = body_array[:, i]
        metrics[n] = dict(
            displacement_xy_m=(v[-1, :2] - v[0, :2]).tolist(),
            path_length_m=float(
                np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1).sum()
            ),
            speed_max_m_s=float(v[:, 4].max()),
            minimum_upright=float(v[:, 3].min()),
            foot_support_fraction=float(v[:, 6].mean()),
            positive_work_J=energy[n],
            torque_saturation_fraction=saturation_count[n] / (completed * 14),
            down_fraction=float((v[:, 3] < 0.55).mean()),
        )
    write_json(out / "metrics.json", metrics)
    result = referee.finish(completed * physics.dt)
    write_json(
        out / "referee.json",
        dict(result=result, eliminated=referee.eliminated, events=referee.events),
    )
    manifest = dict(
        schema="duckfu.run.v1",
        stage="CF00-03 DEVELOPMENT",
        seed=seed,
        config=config.dict(),
        physics=asdict(physics),
        probe=probe,
        capture=capture,
        mujoco_version=mujoco.__version__,
        source_sha256=source,
        robot_sha256=digest(ASSETS / ROBOT),
        policies_sha256={
            n: digest(ASSETS / "microduck/policies" / p) for n, p in POLICIES.items()
        },
        spawns=spawns,
        steps=completed,
        duration_s=completed * physics.dt,
        state_spec=int(STATE),
        physical=dict(
            finite=finite,
            clock_ok=clock_ok,
            external_forces_zero=external_zero,
            model_unchanged=immutable,
            solver_warnings=[int(w.number) for w in d.warning],
            torque_max_Nm=maximum_torque,
            penetration_max_m=contacts["penetration_max_m"],
            penetrating_p99_m=contacts["penetrating_p99_m"],
        ),
        performance=dict(
            wall_s=wall_s,
            real_time_factor=completed * physics.dt / wall_s,
            policy_call_p50_ms=float(np.quantile(inference_times, 0.5) * 1000),
            policy_call_p95_ms=float(np.quantile(inference_times, 0.95) * 1000),
            policy_inference_s=inference_elapsed,
            decision_p95_ms=float(np.quantile(decision_times, 0.95) * 1000),
        ),
        result=result,
        film_gate="PENDING_HUMAN_REVIEW",
        motor_gate="UNQUALIFIED",
        tactical_backend="public_state_rule_with_fixed_coach",
        notes=[
            "Existing BAM-trained policies transferred to capped position servos; simulation-only, not matched new training or hardware validation.",
            "No new policy trained; contacted attacks are not automatically qualified effective strikes.",
        ],
    )
    manifest["hashes"] = {
        p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()
    }
    write_json(out / "manifest.json", manifest)
    return manifest
