"""R1/R2 contact-conditioned recovery stress coach, not an autonomous opponent.

The coach uses an upstream rolling motor to make a real collision. Only target
falls with a preceding actual positive-force duck contact enter R1/R2 counts.
All attempts (including no fall and numerical rejection) remain in the report.
"""

import argparse
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics, strict_load
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.world import build, digest
from microduck_lab.combat.episode import source_hashes


def trial(job):
    seed, index, group, moving_delay, first_roll, course, physics_path = job
    rng = np.random.default_rng(seed)
    separation = (
        float(rng.uniform(0.335, 0.350))
        if course == "trip"
        else float(rng.uniform(0.30, 0.36))
    )
    offset = (
        float(rng.uniform(0.012, 0.026))
        if course == "trip"
        else float(rng.uniform(-0.025, 0.025))
    )
    cfg = Config(
        separation_m=separation,
        lateral_offset_m=offset,
        defense="stand",
        heading_rad=(index % 4) * np.pi / 2,
    )
    physics = strict_load(physics_path, Physics) if physics_path else Physics()
    if physics_path:
        cfg = replace(cfg, physics_profile=Path(physics_path).name)
    m, d, ducks, _ = build(cfg, physics, seed)
    frozen_source = source_hashes()
    audit, guard = (
        ContactAudit(m, ducks),
        RecoveryGuard("cream", moving_delay, first_roll, allow_roll=first_roll),
    )
    last_contact, fall, recovered = None, None, None
    body, policies = [], []
    onset = 1.4 if course == "trip" else 1.0 + (index % 3) * 0.2
    maximum_torque, external_zero = 0.0, True
    for step in range(round(20.0 / physics.dt)):
        now = step * physics.dt
        if (fall is None and now >= 10.0) or (fall is not None and now - fall >= 10.0):
            break
        if step % physics.substeps == 0:
            public = snapshot(ducks, audit, cfg)
            b = public["cream"]
            if (
                fall is None
                and b.upright < 0.8
                and last_contact is not None
                and now - last_contact < 0.30
            ):
                fall = now
            if fall is not None and recovered is None:
                ended = [
                    e
                    for e in guard.events
                    if e["state"] == "RECOVERED" and e["time_s"] >= fall
                ]
                if ended:
                    recovered = ended[0]["time_s"] - fall
            coach = ducks["lavender"]
            if fall is not None and group == "R1":
                coach.active_policy = "walk"
                coach.set_command(
                    twist=(-0.3, 0.0, 0.0) if now - fall < 1.0 else (0.0, 0.0, 0.0)
                )
            elif onset <= now < onset + 0.5:
                coach.active_policy = "roulade"
                coach.set_command(twist=(0.0, 0.0, 0.0))
            else:
                coach.active_policy = "walk"
                coach.set_command(
                    twist=(0.4, 0.0, 0.0) if now > 0.8 else (0.0, 0.0, 0.0)
                )
            target = ducks["cream"]
            selected = guard.choose(now, b)
            target.active_policy = selected or (
                "stand" if recovered is not None else "walk"
            )
            lateral = -0.15 if course == "trip" else (0.15 if index % 2 else -0.15)
            target.set_command(
                twist=(0.0, 0.0, 0.0)
                if selected or now < 0.8 or recovered is not None
                else (0.4, lateral, 0.0)
            )
            coach.step()
            target.step()
            body.append([now, *b.xy, b.z, b.upright, b.speed, b.foot_support])
            policies.append([coach.active_policy, target.active_policy])
        mujoco.mj_step(m, d)
        audit.read(m, d, step, physics.dt, {})
        if audit.touching["cream"]:
            last_contact = now
        maximum_torque = max(maximum_torque, float(np.max(np.abs(d.actuator_force))))
        external_zero &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
    summary = audit.summary()
    if frozen_source != source_hashes():
        raise RuntimeError("Actor source changed during the recovery battery")
    physical = bool(
        summary["gate_proposal_pass"]
        and not any(w.number for w in d.warning)
        and np.isfinite(d.qpos).all()
        and maximum_torque <= 0.6405001
        and external_zero
    )
    return dict(
        seed=seed,
        index=index,
        group=group,
        course=course,
        config=cfg.dict(),
        physics=asdict(physics),
        coach="fixed rolling-motor stress course",
        recovery_calibration=dict(
            moving_delay_s=moving_delay, moving_first_roll=first_roll
        ),
        source_sha256=frozen_source,
        observed_until_s=now,
        recovery_timeout_s=10.0,
        contact_conditioned_fall_s=fall,
        recovered_after_fall_s=recovered,
        within_3s=fall is not None and recovered is not None and recovered <= 3.0,
        physical_pass=physical,
        contacts=summary,
        maximum_torque_Nm=maximum_torque,
        guard_events=guard.events,
        body=body,
        policies=policies,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--group", choices=["R1", "R2"], required=True)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--seed", type=int, default=6301)
    p.add_argument("--index-offset", type=int, default=0)
    p.add_argument("--moving-delay", type=float, default=0.8)
    p.add_argument("--first-roll", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--course", choices=["roll", "trip"], default="roll")
    p.add_argument("--physics-profile", default="")
    a = p.parse_args()
    if not 4 <= a.count <= 160:
        p.error("Use a bounded 4–160 attempted trajectories")
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(
                trial,
                [
                    (
                        a.seed + i,
                        i + a.index_offset,
                        a.group,
                        a.moving_delay,
                        a.first_roll,
                        a.course,
                        a.physics_profile,
                    )
                    for i in range(a.count)
                ],
            )
        )
    eligible = [
        r
        for r in rows
        if r["contact_conditioned_fall_s"] is not None and r["physical_pass"]
    ]
    times = [
        r["recovered_after_fall_s"]
        for r in eligible
        if r["recovered_after_fall_s"] is not None
    ]
    report = dict(
        group=a.group,
        attempted=len(rows),
        eligible_contact_falls=len(eligible),
        no_contact_fall=sum(r["contact_conditioned_fall_s"] is None for r in rows),
        numerical_rejections=sum(not r["physical_pass"] for r in rows),
        within_3s=sum(r["within_3s"] for r in eligible),
        unfinished=len(eligible) - len(times),
        completed_only_p50_p90_s=np.percentile(times, [50, 90]).tolist()
        if times
        else None,
        source_sha256=digest(__file__),
        rows=rows,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
