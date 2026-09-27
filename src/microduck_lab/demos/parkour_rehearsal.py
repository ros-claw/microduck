"""V2 integration rehearsal. Rule-based development controller, not live Jev.

No hero-run claims are made by this runner. Candidate jumps must be explicitly
supplied; reports preserve contact, skill failure and whole-run outcomes.
"""

import argparse, json, hashlib, time
from pathlib import Path
from dataclasses import asdict
import mujoco, numpy as np
from ..parkour.level import build_level
from ..parkour.hazards import HazardDriver
from ..parkour.skills import Maneuver, lane_command
from ..parkour.jump import CandidateJump, GapAudit
from ..parkour.audit import Audit, PropAudit
from ..parkour.encounters import EncounterBrain


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
    sweeper_y=0.0,
    sweeper_height=0.16,
    sweeper_speed=-3.0,
    sweeper_torque=0.10,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m, d, r, track, level = build_level(
        seed,
        duck_floor_ref=duck_floor_ref,
        sweeper_phase=sweeper_phase,
        sweeper_mass=sweeper_mass,
        sweeper_y=sweeper_y,
        sweeper_height=sweeper_height,
        sweeper_speed=sweeper_speed,
        sweeper_torque=sweeper_torque,
    )
    r.bank.paths["jump"] = str(Path(policy).resolve())
    driver = HazardDriver(level.hazards)
    audit = Audit()
    props = PropAudit()
    gap_audit = GapAudit()
    skill = None
    tactical = EncounterBrain(m, d, r, track, level, driver, brain)
    wall_start = time.monotonic()
    route_y = track.lane_width
    escape_y = 0.0
    inputs = []
    initial = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy())
    hazards = {h.kind: h for h in level.hazards}
    stage = "SETTLE"
    events = [dict(type="BOSS_SPAWN", t=0.0)]
    trace = []
    trajectory = []
    results = []
    return_stage = None
    finished = False

    def start(name, target_y=0.0):
        nonlocal skill
        skill = Maneuver(name, float(d.time), target_y=target_y)
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
                    if isinstance(skill, CandidateJump)
                    else skill.update(m, d, r)
                )
                if status != "RUNNING":
                    results.append(
                        dict(
                            stage=stage,
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
                if stage in ("PLAN_BAR", "PLAN_CRATE", "PLAN_GAP"):
                    encounter = {
                        "PLAN_BAR": "bar",
                        "PLAN_CRATE": "crate",
                        "PLAN_GAP": "gap",
                    }[stage]
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
                        if encounter == "bar":
                            stage = "BAR"
                        elif encounter == "gap":
                            stage = "GAP_APPROACH"
                        else:
                            route_y = (
                                track.lane_width
                                if action == "TAKE_LEFT_ROUTE"
                                else -track.lane_width
                            )
                            stage = "DODGE"
                            start(action, route_y)
                elif stage == "BAR" and x >= hazards["push_bar"].x - 0.37:
                    stage = "ROLL"
                    start("ROLL_CENTER")
                elif stage == "PASS_CRATE" and x >= hazards["crate"].x + 0.22:
                    stage = "RETURN"
                    start("TAKE_RIGHT_ROUTE" if route_y > 0 else "TAKE_LEFT_ROUTE", 0.0)
                elif stage == "GAP_APPROACH" and x >= level.gaps[0][0] - 0.26:
                    stage = "BRAKE"
                    start("BRAKE_AND_WAIT")
                elif stage == "GAP_POSITION":
                    error = level.gaps[0][0] - 0.10 - x
                    r.command[:3] = [
                        0.0
                        if 0.06 <= level.gaps[0][0] - x <= 0.14
                        else np.sign(error) * np.clip(3 * abs(error), 0.25, 0.4),
                        np.clip(-2 * y, -0.1, 0.1),
                        np.clip(-3 * r.trunk_yaw(), -1.5, 1.5),
                    ]
                    if (
                        0.06 <= level.gaps[0][0] - x <= 0.14
                        and abs(y) < 0.04
                        and abs(r.trunk_yaw()) < 0.1
                        and np.linalg.norm(r.trunk_linvel()[:2]) < 0.08
                    ):
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
                    stage = "FINISH"
                elif stage == "FINISH" and x >= level.finish_x:
                    finished = True
                    stage = "CELEBRATE"
                    start("BRAKE_AND_WAIT")
                    events.append(dict(type="FINISH", t=float(d.time)))
                elif stage in ("CELEBRATE", "FAILED"):
                    r.active_policy = "stand"
                    r.set_command()
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
            audit.sample(m, d, r, intentional_rotation=stage == "ROLL")
            gap_audit.sample(m, d, r)
            props.sample(m, d, r)
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
        if i % 100 == 0:
            print(
                seed,
                round(d.time, 2),
                stage,
                r.trunk_pos().round(3).tolist(),
                flush=True,
            )
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
        ground_contact=dict(
            reference_s=duck_floor_ref,
            impedance=[0.9, 0.95, 0.001, 0.5, 2.0] if duck_floor_ref else None,
            margin_m=0.0002 if duck_floor_ref else 0.0,
        ),
        publication_status="requires_full_contact_audit_and_matching_input_replay",
        level=asdict(level),
        track=asdict(track),
        brain=brain,
        decisions=tactical.loop.records if tactical.loop else [],
        requests=tactical.requests,
        wall_seconds=time.monotonic() - wall_start,
        policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        finished=finished,
        stage=stage,
        sweeper_phase=sweeper_phase,
        sweeper_mass=sweeper_mass,
        trace=trace,
        skills=results,
        gap_flights=gap_audit.flights,
        audit=audit.report(),
        props=dict(contacts=props.contacts, longest_chase_s=props.longest_chase),
        events=sorted(
            events + driver.events + audit.events + props.events + tactical.events,
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
    run(**vars(p.parse_args()))
