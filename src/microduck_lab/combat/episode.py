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
from .events import attack_outcomes, counter_sequences

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def source_hashes():
    paths = list((ROOT / "src/microduck_lab/combat").glob("*.py"))
    paths += [ROOT / "src/microduck_lab/sim" / n for n in ("runtime.py", "composer.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def run(out, config, physics, seed=101, capture=False, probe=None, training=False):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Invalid seed")
    if probe is not None and (len(probe) != 3 or not np.isfinite(probe).all()):
        raise ValueError("Probe must be a finite three-dimensional twist")
    if (
        config.candidate_motor_path
        and digest(config.candidate_motor_path) != config.candidate_motor_sha256
    ):
        raise ValueError("Candidate motor digest differs")
    if (
        config.evade_motor_path
        and digest(config.evade_motor_path) != config.evade_motor_sha256
    ):
        raise ValueError("Evade motor digest differs")
    if (
        config.walk_motor_path
        and digest(config.walk_motor_path) != config.walk_motor_sha256
    ):
        raise ValueError("Walk motor digest differs")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = source_hashes()
    competition = config.schema in ("duckfu.match.v1", "duckfu.match.v2")
    v2 = config.schema == "duckfu.match.v2"
    recovery_path = getattr(config, "recovery_motor_path", "")
    recovery_sha = getattr(config, "recovery_motor_sha256", "")
    if recovery_path and digest(recovery_path) != recovery_sha:
        raise ValueError("Recovery motor digest differs")
    planners = {}
    teams = None
    if competition:
        from .arena import build_match
        from .tactics import Pilot
        from .learning import FrozenSelector

        m, d, ducks, spawns, teams = build_match(config, physics, seed)
    else:
        m, d, ducks, spawns = build(config, physics, seed)
    if config.candidate_motor_path:
        next(iter(ducks.values())).bank.paths["dash_candidate"] = (
            config.candidate_motor_path
        )
        candidate_metadata = (
            next(iter(ducks.values()))
            .bank.get("dash_candidate")
            .get_modelmeta()
            .custom_metadata_map
        )
    bank = next(iter(ducks.values())).bank
    if recovery_path:
        bank.paths["recovery_candidate"] = recovery_path
        bank.get("recovery_candidate")
    if config.walk_motor_path:
        bank.paths["walk_candidate"] = config.walk_motor_path
        bank.get("walk_candidate")
    if config.evade_motor_path:
        if digest(config.evade_motor_path) != config.evade_motor_sha256:
            raise ValueError("Evade motor digest differs")
        bank.paths["evade_candidate"] = config.evade_motor_path
        bank.get("evade_candidate")
    if competition:
        for path, sha in (
            (config.tactics_path, config.tactics_sha256),
            (config.opponent_path, config.opponent_sha256),
        ):
            if path and digest(path) != sha:
                raise ValueError("Tactical digest differs")
    audit = ContactAudit(m, ducks)
    if competition:
        controllers = {}
        if v2 and config.team_intents:
            from .team_intents import TeamPlanner

            if config.controller_revision == 3:
                from .tactics_v3 import TacticalTeamPlanner as TeamPlanner

            planners = {
                team: TeamPlanner(
                    team,
                    teams,
                    config.team_hold_s,
                    opening_split=config.initial_layout == "peel_course",
                    revision=config.controller_revision,
                )
                for team, enabled in (("RED", config.red_v2), ("BLUE", config.blue_v2))
                if enabled
            }
        for i, n in enumerate(ducks):
            path = config.tactics_path if teams[n] == "RED" else config.opponent_path
            sha = config.tactics_sha256 if teams[n] == "RED" else config.opponent_sha256
            selector = (
                FrozenSelector(
                    path, sha, seed + 10007 * (i + 1), training and teams[n] == "RED"
                )
                if path
                else None
            )
            if v2 and (config.red_v2 if teams[n] == "RED" else config.blue_v2):
                from .tactics_v2 import V2Pilot

                if config.controller_revision == 3:
                    from .tactics_v3 import TacticalPilot as V2Pilot

                controllers[n] = V2Pilot(
                    n, teams[n], teams, config, planners.get(teams[n])
                )
            else:
                if v2 and config.opponent_course != "none" and teams[n] == "BLUE":
                    from .course_coach import SingleDriveCoach, StandingCoach

                    coach = (
                        StandingCoach
                        if config.opponent_course == "stand"
                        else SingleDriveCoach
                    )
                    controllers[n] = coach(n, teams[n], teams, config, selector)
                else:
                    controllers[n] = Pilot(n, teams[n], teams, config, selector)
        if v2 and config.recovery_mode != "legacy":
            from .body_recovery import BodyRecoveryGuard

            for n in controllers:
                controllers[n].guard = BodyRecoveryGuard(
                    n,
                    learned=config.recovery_mode
                    in ("contact_learned", "learned_stabilize"),
                    contact_only=config.recovery_mode != "learned_stabilize",
                )
    else:
        controllers = {n: Maneuver(n, spawns[n]["role"], config) for n in ducks}
    referee = Referee(ducks, config.down_count_s, teams)
    if competition:
        from .strikes import StrikeAudit

        strikes = StrikeAudit(teams)
    nsteps = round(config.duration_s / physics.dt)
    nstate = mujoco.mj_stateSize(m, STATE)
    state = np.empty(nstate)
    mujoco.mj_getState(m, d, state, STATE)
    np.savez_compressed(out / "initial_state.npz", state=state)
    if competition:
        from .arena import match_inventory

        inventory = match_inventory(m, ducks)
    else:
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
                alive = [n for n in ducks if n not in referee.eliminated]
                for planner in planners.values():
                    planner.update(now, public, alive)
                for n in ducks:
                    if v2 and hasattr(controllers[n], "motor_context"):
                        from .transition_contracts import measure_motor_context

                        controllers[n].motor_context = measure_motor_context(
                            ducks[n], audit
                        )
                    other = next(public[o] for o in ducks if o != n)
                    if n in referee.eliminated:
                        commands[n] = MotorCommand("eliminated_hold")
                    else:
                        commands[n] = (
                            controllers[n].choose(now, public, alive)
                            if competition
                            else controllers[n].choose(now, public[n], other)
                        )
                    if (
                        not competition
                        and config.candidate_motor_path
                        and controllers[n].skill == "dash_bump"
                        and commands[n].owner == "maneuver"
                        and (
                            candidate_metadata.get("duckfu_scope") != "forward_active"
                            or controllers[n].phase == "ACTIVE"
                        )
                    ):
                        cmd = commands[n]
                        twist = cmd.twist
                        commands[n] = MotorCommand(
                            cmd.owner,
                            "dash_candidate",
                            twist,
                            cmd.head
                            if candidate_metadata.get("duckfu_head_commands")
                            == "native"
                            else (0.0, 0.0, 0.0, 0.0),
                        )
                    if (
                        not competition
                        and config.evade_motor_path
                        and controllers[n].skill == "evade_step"
                        and controllers[n].phase == "ACTIVE"
                        and commands[n].owner == "maneuver"
                    ):
                        cmd = commands[n]
                        commands[n] = MotorCommand(
                            cmd.owner, "evade_candidate", (0.0, cmd.twist[1], 0.0)
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
                    if config.walk_motor_path and commands[n].policy == "walk":
                        cmd = commands[n]
                        commands[n] = MotorCommand(
                            cmd.owner, "walk_candidate", cmd.twist, cmd.head
                        )
                    row = dict(
                        sim_step=step,
                        time_s=now,
                        actor=n,
                        observation=asdict(public[n]),
                        opponent=(
                            {o: asdict(public[o]) for o in public if o != n}
                            if competition
                            else asdict(other)
                        ),
                        command=asdict(commands[n]),
                        phase=controllers[n].phase,
                        skill=controllers[n].skill,
                        attack_id=controllers[n].active_attack,
                        observation_contact_age_s=physics.dt,
                    )
                    if v2:
                        row["target"] = controllers[n].target
                        row["role"] = getattr(controllers[n], "role", "V1")
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
                before_public = public
                public = snapshot(ducks, audit, config)
                if competition:
                    for event in strikes.observe(
                        completed * physics.dt,
                        audit.episodes,
                        controllers,
                        before_public,
                        public,
                    ):
                        controllers[event["actor"]].pending_reward += 0.3
                    for controller in controllers.values():
                        if controller.decision is not None:
                            controller.pending_reward -= 0.001 / physics.motor_hz
                fault = not (finite and clock_ok and external_zero) or bool(
                    any(w.number for w in d.warning)
                )
                if competition:
                    fault |= any(v["depth_m"] > 0.003 for v in audit.maxima.values())
                previously_eliminated = set(referee.eliminated)
                referee.update(completed * physics.dt, public, physics_fault=fault)
                if competition:
                    for name in set(referee.eliminated) - previously_eliminated:
                        controllers[name].finish(completed * physics.dt, "ELIMINATED")
                        controllers[name].eliminated_at = completed * physics.dt
                        controllers[name].phase = "ELIMINATED"
                if referee.result:
                    break
    finally:
        decision_file.close()
        if contact_file is not None:
            contact_file.close()
    wall_s = time.perf_counter() - wall_start
    if source != source_hashes():
        raise RuntimeError("Source changed during simulation; discard this run")
    if recovery_path and digest(recovery_path) != recovery_sha:
        raise RuntimeError("Recovery motor changed during simulation")
    if (
        config.candidate_motor_path
        and digest(config.candidate_motor_path) != config.candidate_motor_sha256
    ):
        raise RuntimeError("Candidate motor changed during simulation")
    if (
        config.evade_motor_path
        and digest(config.evade_motor_path) != config.evade_motor_sha256
    ):
        raise RuntimeError("Evade motor changed during simulation")
    if (
        config.walk_motor_path
        and digest(config.walk_motor_path) != config.walk_motor_sha256
    ):
        raise RuntimeError("Walk motor changed during simulation")
    if competition:
        for path, sha in (
            (config.tactics_path, config.tactics_sha256),
            (config.opponent_path, config.opponent_sha256),
        ):
            if path and digest(path) != sha:
                raise RuntimeError("Tactical weights changed during simulation")
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
    if competition and (not immutable or not contacts["gate_proposal_pass"]):
        referee.update(
            completed * physics.dt, snapshot(ducks, audit, config), physics_fault=True
        )
    write_json(out / "contacts_summary.json", contacts)
    all_events = (
        [e for c in controllers.values() for e in c.events]
        + audit.events
        + referee.events
    )
    if competition:
        all_events += [e for c in controllers.values() for e in c.guard.events]
        result_for_learning = referee.finish(completed * physics.dt)
        # Each transition spans its actual maneuver duration (semi-MDP).
        # A terminal team result is the only winner reward; no scripted damage.
        for n, c in controllers.items():
            winner = result_for_learning["winner"]
            terminal_reward = (3.0 if winner == teams[n] else -3.0) if winner else 0.0
            if c.decision is not None:
                c.pending_reward += terminal_reward
            elif c.trace:
                c.trace[-1]["reward"] += terminal_reward
                c.trace[-1]["terminal"] = True
                if c.eliminated_at is not None:
                    # An eliminated actor waits for its team's result. Keep
                    # these real seconds in semi-MDP discounting too.
                    waiting = max(0.0, completed * physics.dt - c.eliminated_at)
                    c.trace[-1]["duration_s"] += waiting
                    c.trace[-1]["absorbing_wait_s"] = waiting
            c.finish(completed * physics.dt, "EPISODE_END", final=True)
        write_json(
            out / "transitions.json", {n: c.trace for n, c in controllers.items()}
        )
        all_events = (
            [e for c in controllers.values() for e in c.events]
            + [e for c in controllers.values() for e in c.guard.events]
            + audit.events
            + referee.events
            + strikes.events
            + [e for planner in planners.values() for e in planner.events]
        )
    # Attribute attack results after contact episode consolidation. CONTACTED is
    # deliberately not VALID_STRIKE: its effectiveness remains unqualified.
    all_events += attack_outcomes(all_events, audit.episodes, teams)
    if competition:
        all_events += counter_sequences(all_events)
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
        schema="duckfu.match-run.v1" if competition else "duckfu.run.v1",
        stage=(
            "STANDING_TARGET_ACTION_COURSE"
            if v2 and config.opponent_course == "stand"
            else "SINGLE_DRIVE_TRANSITION_COURSE"
            if v2 and config.opponent_course == "single_drive"
            else "PEEL_PRESSURE_COURSE"
            if v2 and config.initial_layout != "standard"
            else "COMPETITION_DEVELOPMENT"
            if competition
            else "CF00-03 DEVELOPMENT"
        ),
        seed=seed,
        config=config.dict(),
        physics=asdict(physics),
        probe=probe,
        training=training,
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
        motor_gate="SEE_FROZEN_QUALIFICATION_REPORT",
        tactical_backend=(
            "v2_public_rule_composition"
            if v2
            else "learned_categorical"
            if config.tactics_path
            else "public_state_rule"
        )
        if competition
        else "public_state_rule_with_fixed_coach",
        notes=[
            "Upstream motor priors transfer from BAM to capped position servos; new Duck Fu candidates are trained with matched capped position servos. Simulation only; no hardware validation.",
            "CONTACTED_ATTACK alone is not VALID_STRIKE; see frozen motor qualification and separate observed-response events. No health/contact damage.",
        ],
    )
    manifest["hashes"] = {
        p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()
    }
    write_json(out / "manifest.json", manifest)
    return manifest
