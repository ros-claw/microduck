"""Live/rule Neon Escape integration with actuator-only compound maneuvers.

Immutable captures retain skill outcomes, physical contacts and input-replay
proofs. Predictive plans advance clones; they never write poses into the robot.
"""

import argparse, json, hashlib, time, tarfile
from pathlib import Path
from dataclasses import asdict
import mujoco, numpy as np
from ..parkour.level import build_level
from ..parkour.world import foot_support
from ..parkour.hazards import HazardDriver
from ..parkour.skills import Maneuver, lane_command, roll_entry_ready
from ..parkour.jump import CandidateJump, GapAudit
from ..parkour.audit import Audit, PropAudit
from ..parkour.encounters import EncounterBrain
from ..parkour.preview import (
    GapSequence,
    choose_gap_sequence,
    choose_sweeper_route,
    sweeper_command,
    choose_roll_sequence,
    RollSequence,
)
from ..parkour.arcade import ArcadeDriver, EncounterCombo, bowling_command
from ..parkour.victory import VictoryDance


def run(
    policy,
    seed=0,
    seconds=35.0,
    output="artifacts/neon-escape-v2/rehearsal",
    capture=False,
    sweeper_phase=0.0,
    sweeper_mass=0.20,
    brain="rule",
    duck_floor_ref=None,
    hard_contacts=False,
    roll_policy=None,
    sweeper_y=0.0,
    sweeper_height=0.16,
    sweeper_speed=-3.0,
    sweeper_torque=0.10,
    predictive=False,
    difficulty="classic",
    legacy_handoffs=False,
    forecast_entry_roll=False,
    victory_dance=False,
):
    output = Path(output)
    if (output / "audit.json").exists():
        raise FileExistsError(
            "Choose a fresh output directory; existing evidence is immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    source_files = sorted((repository / "src").rglob("*.py")) + sorted(
        (repository / "src/microduck_lab/parkour").glob("*.json")
    )
    source_hashes = {
        str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    with tarfile.open(output / "source.tar.gz", "w:gz") as snapshot:
        for path in source_files:
            snapshot.add(path, arcname=str(path.relative_to(repository)))
    m, d, r, track, level = build_level(
        seed,
        difficulty=difficulty,
        duck_floor_ref=duck_floor_ref,
        hard_contacts=hard_contacts,
        sweeper_phase=sweeper_phase,
        sweeper_mass=sweeper_mass,
        sweeper_y=sweeper_y,
        sweeper_height=sweeper_height,
        sweeper_speed=sweeper_speed,
        sweeper_torque=sweeper_torque,
    )
    r.bank.paths["jump"] = str(Path(policy).resolve())
    if roll_policy:
        r.bank.paths["roulade"] = str(Path(roll_policy).resolve())
    if victory_dance:
        r.bank.paths["victory_hop"] = str(repository / "policies/ropehop_centered.onnx")
    victory = None
    arcade = difficulty == "arcade"
    combo_planning = arcade and not legacy_handoffs
    driver = ArcadeDriver(level.hazards) if arcade else HazardDriver(level.hazards)
    audit = Audit(
        interactive_props=("playball", "pin0", "pin1", "pin2") if arcade else ()
    )
    combo = EncounterCombo() if arcade else None
    props = PropAudit()
    gap_audit = GapAudit()
    skill = None
    tactical = EncounterBrain(m, d, r, track, level, driver, brain)
    wall_start = time.monotonic()
    route_y = track.lane_width
    escape_y = 0.0
    inputs = []
    initial = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy())
    hazards = {h.kind: h for h in reversed(level.hazards)}
    named_hazards = {h.name: h for h in level.hazards}
    stage = "SETTLE"
    events = [dict(type="BOSS_SPAWN", t=0.0)]
    trace = []
    trajectory = []
    results = []
    forecasts = []
    gap_planned = False
    return_stage = None
    finished = False
    launch_stable = 0.0
    launch_settling = False
    launch_pulse_until = 0.0
    launch_brake_until = 0.0
    bowl_watch_started = None
    exit_roll_planned = False
    first_roll_planned = False

    def start(name, target_y=0.0, forward_speed=0.7):
        nonlocal skill
        skill = Maneuver(
            name, float(d.time), target_y=target_y, forward_speed=forward_speed
        )
        skill.launch_position = r.trunk_pos().copy()
        events.append(dict(type="SKILL_START", t=float(d.time), skill=name))

    for i in range(round(seconds * 50)):
        x, y, z = r.trunk_pos()
        if d.time < 2:
            r.active_policy = "stand"
            r.set_command()
            if d.time >= 1.0:
                tactical.prepare("bar", remaining=max(0.0, 2.0 - d.time))
        else:
            if stage == "SETTLE":
                stage = "PLAN_BAR"
            if audit.fallen and stage != "RECOVER":
                return_stage = stage
                stage = "RECOVER"
                start("RECOVER")
            if skill is not None:
                if stage == "ROLL":
                    if hard_contacts:
                        if (
                            skill.phase in ("ALIGN", "STABLE")
                            and d.xmat[r.trunk_body_id, 8] > 0.95
                        ):
                            tactical.prepare(
                                "crate",
                                remaining=0.7,
                                predicted_position=r.trunk_pos()
                                + np.array([0.04, 0.0, 0.0]),
                            )
                    else:
                        tactical.prepare(
                            "crate",
                            remaining=max(0.0, 1.30 - (d.time - skill.started)),
                            predicted_position=skill.launch_position
                            + np.array([0.60, 0.0, 0.0]),
                        )
                elif stage == "RETURN":
                    tactical.prepare(
                        "gap", remaining=max(0.0, 1.20 - (d.time - skill.started))
                    )
                status = (
                    skill.update(m, d, r, gap_audit)
                    if isinstance(skill, (CandidateJump, GapSequence))
                    else skill.update(m, d, r)
                )
                if status != "RUNNING":
                    results.append(
                        dict(
                            stage=stage,
                            displacement_m=(
                                r.trunk_pos() - skill.launch_position
                            ).tolist()
                            if hasattr(skill, "launch_position")
                            else None,
                            skill=skill.name,
                            status=status,
                            t=float(d.time),
                            duration=float(d.time) - skill.started,
                        )
                    )
                    audit.skill_result(skill, float(d.time))
                    skill = None
                    if status == "FAILED":
                        stage = "FAILED"
                    elif stage == "ROLL":
                        stage = "PLAN_CRATE"
                    elif stage == "DODGE":
                        stage = "PASS_CRATE"
                    elif stage == "RETURN":
                        stage = "PLAN_GAP"
                    elif stage == "BRAKE":
                        stage = "GAP_POSITION"
                    elif stage == "JUMP":
                        stage = "SWEEPER"
                        if predictive:
                            escape_y, candidates = choose_sweeper_route(
                                m, d, r, driver, hazards["sweeper"].x
                            )
                            forecasts.append(
                                dict(
                                    t=float(d.time),
                                    kind="sweeper",
                                    candidates=candidates,
                                    accepted=any(c["passed"] for c in candidates),
                                )
                            )
                            events.append(
                                dict(
                                    type="ROUTE_PREVIEW",
                                    t=float(d.time),
                                    target_y=escape_y,
                                    accepted=any(c["passed"] for c in candidates),
                                )
                            )
                    elif stage == "EXIT_ROLL":
                        stage = "FINISH"
                        if victory_dance:
                            victory = VictoryDance(float(d.time))
                            events.append(dict(type="VICTORY_START", t=float(d.time)))
                    elif stage == "RECOVER":
                        stage = return_stage
                        escape_y = (
                            np.sign(r.trunk_pos()[1] or 1.0) * 0.30
                            if abs(hazards["sweeper"].y) < 0.1
                            else -np.sign(hazards["sweeper"].y) * 0.15
                        )
            if skill is None:
                r.joint_vel_delay = 0
                r.active_policy = "run"
                r.command[:3] = lane_command(
                    r,
                    route_y
                    if stage == "PASS_CRATE"
                    else escape_y
                    if stage == "SWEEPER"
                    else 0.0,
                    0.7,
                )
                if predictive and stage == "SWEEPER":
                    r.bank.mirror_run = escape_y < 0
                    r.command[:3] = sweeper_command(r, escape_y)
                if arcade and stage in ("BOWLING", "EXIT_BAR"):
                    r.command[:3] = bowling_command(r, m, d)
                if arcade and stage == "BOWL_OBSERVE":
                    r.bank.mirror_run = False
                    r.set_command()
                    if np.linalg.norm(r.trunk_linvel()[:2]) < 0.04:
                        r.active_policy = "stand"
                if arcade and stage == "RECENTER_CRATE":
                    r.bank.mirror_run = y > 0
                    r.command[:3] = lane_command(r, 0.0, 0.40)
                    if abs(y) < 0.10 and abs(r.trunk_yaw()) < 0.25:
                        stage = "PLAN_CRATE"
                        r.set_command()
                if arcade and stage == "SWEEPER" and x > hazards["sweeper"].x - 0.12:
                    tactical.prepare("bowling", remaining=0.7)
                if (
                    arcade
                    and stage == "BOWL_OBSERVE"
                    and len(driver.bowling.toppled) >= 2
                ):
                    tactical.prepare("exit_bar", remaining=0.7)
                if stage in (
                    "PLAN_BAR",
                    "PLAN_CRATE",
                    "PLAN_GAP",
                    "PLAN_BOWL",
                    "PLAN_EXIT_BAR",
                ):
                    encounter = {
                        "PLAN_BAR": "bar",
                        "PLAN_CRATE": "crate",
                        "PLAN_GAP": "gap",
                        "PLAN_BOWL": "bowling",
                        "PLAN_EXIT_BAR": "exit_bar",
                    }[stage]
                    if arcade and encounter == "crate" and abs(y) > 0.17:
                        stage = "RECENTER_CRATE"
                        events.append(
                            dict(
                                type="HANDOFF_REALIGN",
                                t=float(d.time),
                                entry_y=float(y),
                            )
                        )
                        action = None
                    else:
                        action = tactical.prepare(encounter, handoff=True)
                    r.command[:3] = 0.0
                    if action == "BRAKE_AND_WAIT":
                        events.append(
                            dict(
                                type="TACTICAL_WAIT",
                                t=float(d.time),
                                encounter=encounter,
                            )
                        )
                        tactical.selected.pop(encounter, None)
                    elif action is not None:
                        if encounter == "exit_bar":
                            stage = "EXIT_BAR"
                        elif encounter == "bowling":
                            stage = "BOWLING"
                            events.append(
                                dict(
                                    type="SKILL_START",
                                    t=float(d.time),
                                    skill="PUSH_BALL",
                                )
                            )
                        elif encounter == "bar":
                            stage = "BAR"
                        elif encounter == "gap":
                            stage = "GAP_APPROACH"
                        else:
                            route_y = tactical.route_envelopes[action].get(
                                "target_y_m",
                                track.lane_width
                                if action == "TAKE_LEFT_ROUTE"
                                else -track.lane_width,
                            )
                            stage = "DODGE"
                            start(action, route_y)
                elif (
                    combo_planning
                    and forecast_entry_roll
                    and stage == "BAR"
                    and not first_roll_planned
                    and x >= hazards["push_bar"].x - 0.40
                ):
                    first_roll_planned = True
                    plan, candidates = choose_roll_sequence(
                        m, d, r, driver, hazards["push_bar"].x, strict_self=True
                    )
                    forecasts.append(
                        dict(
                            t=float(d.time),
                            kind="entry_roll",
                            candidates=candidates,
                            accepted=plan is not None,
                        )
                    )
                    if plan is None:
                        stage = "ROLL_ABORT"
                        events.append(
                            dict(type="ROLL_ABORT", t=float(d.time), encounter="bar")
                        )
                    else:
                        stage = "ROLL"
                        skill = RollSequence(float(d.time), plan, r.trunk_pos())
                elif (
                    (not combo_planning or not forecast_entry_roll)
                    and stage == "BAR"
                    and (
                        roll_entry_ready(m, d, r, hazards["push_bar"].x)
                        if hard_contacts
                        else x >= hazards["push_bar"].x - 0.37
                    )
                ):
                    stage = "ROLL"
                    start("ROLL_CENTER")
                elif stage == "PASS_CRATE" and x >= hazards["crate"].x + 0.22:
                    stage = "RETURN"
                    start(
                        "TAKE_RIGHT_ROUTE" if route_y > 0 else "TAKE_LEFT_ROUTE",
                        0.0,
                        forward_speed=0.45 if hard_contacts else 0.7,
                    )
                elif (
                    predictive
                    and stage == "GAP_APPROACH"
                    and not gap_planned
                    and x >= level.gaps[0][0] - 0.30
                ):
                    gap_planned = True
                    plan, candidates = choose_gap_sequence(
                        m, d, r, level.gaps[0][1], expanded=combo_planning
                    )
                    forecasts.append(
                        dict(
                            t=float(d.time),
                            kind="gap",
                            candidates=candidates,
                            accepted=plan is not None,
                        )
                    )
                    events.append(
                        dict(
                            type="PHYSICAL_FORECAST",
                            t=float(d.time),
                            accepted=plan is not None,
                            wall_ms=sum(c["wall_ms"] for c in candidates),
                        )
                    )
                    if plan is not None:
                        stage = "JUMP"
                        skill = GapSequence(
                            float(d.time), level.gaps[0][1], plan, r.trunk_pos()
                        )
                        skill.update(m, d, r, gap_audit)
                elif stage == "GAP_APPROACH" and x >= level.gaps[0][0] - 0.26:
                    stage = "BRAKE"
                    start("BRAKE_AND_WAIT")
                elif stage == "GAP_POSITION":
                    distance = level.gaps[0][0] - x
                    orientation_ready = abs(y) < 0.08 and abs(r.trunk_yaw()) < 0.1
                    speed = float(np.linalg.norm(r.trunk_linvel()[:2]))
                    inner_lo, inner_hi = (
                        (0.040, 0.085) if hard_contacts else (0.060, 0.140)
                    )
                    in_launch_window = (
                        inner_lo <= distance <= inner_hi and orientation_ready
                    )
                    # Short forward pulses, followed by physical braking. Stand
                    # is a balance policy, not a moving-body brake. Sustained
                    # position commands near the ledge can walk into the void.
                    r.set_command()
                    if launch_settling and in_launch_window:
                        r.active_policy = "stand"
                    elif float(d.time) < launch_pulse_until:
                        r.command[0] = 0.45
                    elif speed > 0.025 or float(d.time) < launch_brake_until:
                        r.active_policy = "run"
                    elif in_launch_window:
                        launch_settling = True
                        r.active_policy = "stand"
                    elif inner_lo <= distance <= inner_hi:
                        launch_settling = False
                        r.active_policy = "run"
                    elif distance > inner_hi:
                        launch_settling = False
                        launch_pulse_until = float(d.time) + 0.16
                        launch_brake_until = float(d.time) + 0.80
                        r.command[0] = 0.45
                    else:
                        # No unqualified backward gait at the edge: terminate
                        # this attempt with an explicit safety failure.
                        stage = "UNSAFE_LAUNCH_STATE"
                        events.append(
                            dict(
                                type="LAUNCH_ABORT",
                                t=float(d.time),
                                distance_m=float(distance),
                            )
                        )
                    if not launch_settling:
                        r.command[2] = np.clip(-3 * r.trunk_yaw(), -1.5, 1.5)
                    ready = (
                        in_launch_window
                        and foot_support(m, d)
                        and r.trunk_pos()[2] > 0.10
                        and float(d.xmat[r.trunk_body_id, 8]) > 0.97
                        and np.linalg.norm(r.trunk_linvel()) < 0.04
                        and np.max(np.abs(d.qvel[r.joint_qvel_idx])) < 0.5
                    )
                    launch_stable = launch_stable + 0.02 if ready else 0.0
                    if launch_stable >= 0.12:
                        stage = "JUMP"
                        skill = CandidateJump(
                            float(d.time),
                            level.gaps[0][1],
                            landing_tolerance=track.width / 2
                            - track.duck_width / 2
                            - 0.015,
                        )
                        events.append(
                            dict(
                                type="SKILL_START", t=float(d.time), skill="JUMP_CENTER"
                            )
                        )
                elif stage == "SWEEPER" and x > hazards["sweeper"].x + 0.40:
                    stage = "PLAN_BOWL" if arcade else "FINISH"
                elif (
                    arcade
                    and stage in ("BOWLING", "BOWL_OBSERVE")
                    and driver.bowling.unlocked
                ):
                    stage = "PLAN_EXIT_BAR"
                    events.append(
                        dict(
                            type="SKILL_RESULT",
                            t=float(d.time),
                            skill="PUSH_BALL",
                            status="SUCCESS",
                        )
                    )
                elif (
                    arcade
                    and stage == "BOWLING"
                    and driver.bowling.duck_push_impulse > 0.025
                    and d.qvel[m.jnt_dofadr[m.joint("playball/free").id]] > 0.30
                ):
                    stage = "BOWL_OBSERVE"
                    bowl_watch_started = float(d.time)
                    events.append(dict(type="SHOT_RELEASE", t=float(d.time)))
                elif (
                    arcade
                    and stage == "BOWL_OBSERVE"
                    and d.time - bowl_watch_started > 2.5
                ):
                    stage = "BOWLING_MISS"
                    events.append(dict(type="SHOT_FAILED", t=float(d.time)))
                elif (
                    arcade
                    and stage == "EXIT_BAR"
                    and not exit_roll_planned
                    and x >= named_hazards["exit_bar"].x - 0.36
                ):
                    exit_roll_planned = True
                    plan, candidates = choose_roll_sequence(
                        m,
                        d,
                        r,
                        driver,
                        named_hazards["exit_bar"].x,
                        strict_self=combo_planning,
                    )
                    forecasts.append(
                        dict(
                            t=float(d.time),
                            kind="exit_roll",
                            candidates=candidates,
                            accepted=plan is not None,
                        )
                    )
                    if plan is None:
                        stage = "ROLL_ABORT"
                        events.append(dict(type="ROLL_ABORT", t=float(d.time)))
                    else:
                        stage = "EXIT_ROLL"
                        skill = RollSequence(float(d.time), plan, r.trunk_pos())
                elif stage == "FINISH" and victory is None and x >= level.finish_x:
                    finished = True
                    stage = "CELEBRATE"
                    start("BRAKE_AND_WAIT")
                    events.append(dict(type="FINISH", t=float(d.time)))
                elif stage in (
                    "CELEBRATE",
                    "FAILED",
                    "UNSAFE_LAUNCH_STATE",
                    "BOWLING_MISS",
                    "ROLL_ABORT",
                ):
                    r.active_policy = "stand"
                    r.set_command()
        if victory is not None:
            victory.update(m, d, r)
            if victory.finished_at is not None and not finished:
                finished = True
                stage = "CELEBRATE"
                events.append(dict(type="FINISH", t=victory.finished_at))
        driver.step(m, d, float(x))
        r.step()
        if capture:
            inputs.append(
                (
                    float(d.time),
                    d.ctrl.copy(),
                    d.eq_active.copy(),
                    d.xfrc_applied.copy(),
                    m.geom_rgba.copy(),
                )
            )
        for j in range(100):
            mujoco.mj_step(m, d)
            audit.sample(m, d, r, intentional_rotation=stage in ("ROLL", "EXIT_ROLL"))
            gap_audit.sample(m, d, r)
            props.sample(m, d, r)
            if victory is not None:
                victory.sample(m, d, r)
            if arcade:
                driver.sample(m, d)
            if capture and j % 25 == 0:
                trajectory.append(
                    (
                        float(d.time),
                        d.qpos.copy(),
                        d.qvel.copy(),
                        d.ctrl.copy(),
                        d.eq_active.copy(),
                        m.geom_rgba.copy(),
                    )
                )
        if brain == "jev":
            delay = wall_start + float(d.time) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        trace.append(
            dict(
                t=float(d.time),
                stage=stage,
                pos=r.trunk_pos().tolist(),
                up=float(d.xmat[r.trunk_body_id, 8]),
                yaw=r.trunk_yaw(),
                boss=d.body("boss").xpos.tolist(),
            )
        )
        if arcade:
            combo.sample(m, d, r, audit, gap_audit, driver, results, level)
        if i % 100 == 0:
            print(
                seed,
                round(d.time, 2),
                stage,
                r.trunk_pos().round(3).tolist(),
                flush=True,
            )
        if stage == "UNSAFE_LAUNCH_STATE":
            break
        if r.trunk_pos()[2] < -0.3:
            stage = "VOID_FALL"
            break
        if (
            finished
            and d.time - next(e["t"] for e in events if e["type"] == "FINISH") > 8
        ):
            break
    tactical.close()
    report = dict(
        seed=seed,
        source_hashes=source_hashes,
        source_archive_sha256=hashlib.sha256(
            (output / "source.tar.gz").read_bytes()
        ).hexdigest(),
        predictive=predictive,
        difficulty=difficulty,
        victory=victory.report(m, d, r) if victory else None,
        combination_planning=combo_planning,
        forecast_entry_roll=forecast_entry_roll,
        forecasts=forecasts,
        bowling=driver.bowling.report() if arcade else None,
        encounter_combo=sorted(combo.verified) if arcade else None,
        hard_contacts=hard_contacts,
        roll_policy_sha256=hashlib.sha256(
            Path(r.bank.paths["roulade"]).read_bytes()
        ).hexdigest(),
        ground_contact=dict(
            reference_s=0.002 if hard_contacts else duck_floor_ref,
            impedance=[0.9, 0.95, 0.001, 0.5, 2.0]
            if (duck_floor_ref or hard_contacts)
            else None,
            margin_m=0.0002 if (duck_floor_ref or hard_contacts) else 0.0,
        ),
        publication_status="requires_full_contact_audit_and_matching_input_replay",
        level=asdict(level),
        track=asdict(track),
        brain=brain,
        decisions=tactical.loop.records if tactical.loop else [],
        requests=tactical.requests,
        wall_seconds=time.monotonic() - wall_start,
        policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        policy_hashes={
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in r.bank.paths.items()
        },
        finished=finished,
        stage=stage,
        sweeper_phase=sweeper_phase,
        sweeper_mass=sweeper_mass,
        trace=trace,
        skills=results,
        gap_flights=gap_audit.flights,
        short_same_platform_flights=gap_audit.short_same_platform_flights,
        audit=audit.report(),
        props=dict(contacts=props.contacts, longest_chase_s=props.longest_chase),
        events=sorted(
            events
            + driver.events
            + audit.events
            + props.events
            + tactical.events
            + (driver.bowling.events + combo.events if arcade else [])
            + ([e for e in victory.events if e["type"] != "FINISH"] if victory else []),
            key=lambda e: e["t"],
        ),
        warnings=d.warning.number.tolist(),
    )
    report["component_course_passed"] = bool(
        finished
        and stage == "CELEBRATE"
        and float(d.xmat[r.trunk_body_id, 8]) > 0.95
        and r.trunk_pos()[2] > 0.09
        and gap_audit.crossed
        and not any(s["status"] == "FAILED" for s in results)
        and report["audit"]["hp"] > 0
        and report["audit"]["max_penetration_m"] < 0.0015
        and report["audit"]["p99_penetration_m"] < 0.001
        and not d.warning.number.any()
        and (not arcade or (driver.bowling.unlocked and len(combo.verified) == 6))
        and (not victory_dance or (victory is not None and report["victory"]["passed"]))
    )
    report["passed"] = (
        report["component_course_passed"]
        and max((c["max_penetration_m"] for c in props.contacts.values()), default=0.0)
        < 0.0015
        and props.longest_chase >= 5.0
        and "boss/portal" in props.contacts
        and any(
            s["skill"] == "RECOVER"
            and s["status"] == "SUCCESS"
            and s["duration"] <= 2.0
            for s in results
        )
    )
    if capture:
        mujoco.mj_saveModel(m, str(output / "scene.mjb"), None)
        names = ["time", "qpos", "qvel", "ctrl", "eq_active", "geom_rgba"]
        np.savez_compressed(
            output / "trajectory.npz",
            **{
                name: np.asarray([row[i] for row in trajectory])
                for i, name in enumerate(names)
            },
        )
        np.savez_compressed(
            output / "inputs.npz",
            initial_qpos=initial["qpos"],
            initial_qvel=initial["qvel"],
            **{
                name: np.asarray([row[i] for row in inputs])
                for i, name in enumerate(
                    ["time", "ctrl", "eq_active", "xfrc_applied", "geom_rgba"]
                )
            },
        )
        report["capture"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ["scene.mjb", "trajectory.npz", "inputs.npz"]
        }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "finished",
        finished,
        "passed",
        report["passed"],
        "stage",
        stage,
        "HP",
        report["audit"]["hp"],
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--brain", choices=["rule", "jev"], default="rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=35)
    p.add_argument("--output", default="artifacts/neon-escape-v2/rehearsal")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--sweeper-phase", type=float, default=0.0)
    p.add_argument("--sweeper-mass", type=float, default=0.20)
    p.add_argument("--sweeper-y", type=float, default=0.0)
    p.add_argument("--sweeper-height", type=float, default=0.16)
    p.add_argument("--sweeper-speed", type=float, default=-3.0)
    p.add_argument("--sweeper-torque", type=float, default=0.10)
    p.add_argument("--duck-floor-ref", type=float, default=None)
    p.add_argument("--hard-contacts", action="store_true")
    p.add_argument("--roll-policy")
    p.add_argument("--predictive", action="store_true")
    p.add_argument("--legacy-handoffs", action="store_true")
    p.add_argument("--victory-dance", action="store_true")
    p.add_argument(
        "--forecast-entry-roll",
        action="store_true",
        help="Experimental brake/settle entry preview; default keeps moving phase entry",
    )
    p.add_argument(
        "--difficulty", choices=["classic", "chase", "arcade"], default="classic"
    )
    run(**vars(p.parse_args()))
