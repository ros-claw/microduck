"""Matched R1/R2 body qualification under actual physical coach collisions.

The rolling coach is a declared stress course, not an autonomous opponent.
Nothing writes a root pose, velocity or external force after initialization.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.episode import source_hashes, write_json
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.world import build, digest


def trial(job):
    seed, index, group, method, motor = job
    frozen_source = source_hashes()
    motor_sha = digest(motor) if method == "recovery_candidate" else None
    rng = np.random.default_rng(seed)
    cfg = Config(
        separation_m=float(rng.uniform(0.30, 0.36)),
        lateral_offset_m=float(rng.uniform(-0.025, 0.025)),
        defense="stand",
        heading_rad=(index % 4) * np.pi / 2,
    )
    physics = Physics(dt=0.000125, native_ccd=False)
    m, d, ducks, _ = build(cfg, physics, seed)
    bank = next(iter(ducks.values())).bank
    if motor_sha:
        bank.paths["recovery_candidate"] = motor
        meta = bank.get("recovery_candidate").get_modelmeta().custom_metadata_map
        assert meta.get("duckfu_scope") == "recorded_contact_recovery"
    bank.paths["walk_candidate"] = "policies/candidates/duckfu_locomotion_v1_200.onnx"
    audit = ContactAudit(m, ducks)
    guard = RecoveryGuard("cream", allow_roll=False)
    last_contact = fall = stable_since = complete = walk_start = None
    walk_origin = walk_yaw = pre_intervention_sha = None
    pre_controls = hashlib.sha256()
    body, policies = [], []
    onset = 1.0 + (index % 3) * 0.2
    maximum_torque, finite, external_zero = 0.0, True, True
    for step in range(round(20.0 / physics.dt)):
        now = step * physics.dt
        if (fall is None and now >= 10.0) or (fall is not None and now - fall >= 10.0):
            break
        if step % physics.substeps == 0:
            b = snapshot(ducks, audit, cfg)["cream"]
            if (
                fall is None
                and b.upright < 0.80
                and last_contact is not None
                and now - last_contact < 0.30
            ):
                fall = now
                state = np.empty(
                    mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
                )
                mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                pre_intervention_sha = hashlib.sha256(state.tobytes()).hexdigest()
            good = (
                b.upright > 0.94
                and b.z > 0.105
                and b.foot_support
                and not b.body_ground
                and b.speed < 0.16
            )
            stable_since = (
                (now if stable_since is None else stable_since)
                if good and fall is not None
                else None
            )
            if (
                complete is None
                and stable_since is not None
                and now - stable_since >= 0.30 - 1e-9
            ):
                complete = now
            coach = ducks["lavender"]
            if fall is not None and group == "R1":
                coach.active_policy = "walk"
                coach.set_command(twist=(-0.3, 0, 0) if now - fall < 1.0 else (0, 0, 0))
            elif onset <= now < onset + 0.5:
                coach.active_policy = "roulade"
                coach.set_command()
            else:
                coach.active_policy = "walk"
                coach.set_command(twist=(0.4, 0, 0) if now > 0.8 else (0, 0, 0))
            target = ducks["cream"]
            if complete is not None and now >= complete + 0.30 - 1e-9:
                if walk_start is None:
                    walk_start, walk_origin, walk_yaw = now, np.array(b.xy), b.yaw
                target.active_policy = "walk_candidate"
                target.set_command(twist=(0.28, 0, 0))
            elif complete is not None:
                target.active_policy = "stand"
                target.set_command()
            elif fall is not None:
                target.active_policy = (
                    (guard.choose(now, b) or "stand") if method == "guard" else method
                )
                target.set_command()
            else:
                target.active_policy = "walk"
                target.set_command(
                    twist=(0.4, 0.15 if index % 2 else -0.15, 0)
                    if now >= 0.8
                    else (0, 0, 0)
                )
            coach.step()
            target.step()
            if fall is None:
                pre_controls.update(d.ctrl.tobytes())
            body.append(
                [
                    now,
                    *b.xy,
                    b.z,
                    b.upright,
                    b.speed,
                    float(b.foot_support),
                    float(b.body_ground),
                ]
            )
            policies.append([coach.active_policy, target.active_policy])
            if walk_start is not None and now - walk_start >= 1.0 - 1e-9:
                break
            if any(c["depth_m"] > 0.003 for c in audit.maxima.values()):
                break
        mujoco.mj_step(m, d)
        audit.read(m, d, step, physics.dt, {})
        if audit.touching["cream"]:
            last_contact = now
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        external_zero &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
        maximum_torque = max(maximum_torque, float(np.abs(d.actuator_force).max()))
    contacts = audit.summary()
    physical = bool(
        finite
        and external_zero
        and not any(w.number for w in d.warning)
        and maximum_torque <= 0.6405001
        and contacts["gate_proposal_pass"]
    )
    walk = [r for r in body if walk_start is not None and r[0] >= walk_start]
    progress = (
        float(
            (np.array(body[-1][1:3]) - walk_origin)
            @ np.array([np.cos(walk_yaw), np.sin(walk_yaw)])
        )
        if walk
        else None
    )
    if frozen_source != source_hashes() or (
        motor_sha is not None and motor_sha != digest(motor)
    ):
        raise RuntimeError("Source or motor changed during physical course")
    duration = complete - fall if complete is not None else None
    return dict(
        seed=seed,
        index=index,
        group=group,
        method=method,
        policy_sha256=motor_sha,
        config=cfg.dict(),
        physics=asdict(physics),
        source_sha256=frozen_source,
        physical_pass=physical,
        contact_conditioned_fall_s=fall,
        recovered_after_fall_s=duration,
        within_3s=duration is not None and duration <= 3,
        recovery_to_walk_pass=bool(
            walk
            and walk[-1][0] - walk_start >= 0.98
            and min(r[4] for r in walk) > 0.90
            and walk[-1][4] > 0.94
            and progress > 0.05
        ),
        walk_forward_progress_m=progress,
        pre_intervention_state_sha256=pre_intervention_sha,
        pre_intervention_controls_sha256=pre_controls.hexdigest(),
        observed_until_s=now,
        maximum_torque_Nm=maximum_torque,
        contacts=contacts,
        guard_events=guard.events,
        body=body,
        policies=policies,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--method",
        action="append",
        choices=["stand", "guard", "recovery_candidate"],
        required=True,
    )
    p.add_argument("--motor")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=150001)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    if not 4 <= a.count <= 40:
        p.error("Use a bounded 4–40 physical source trajectories per group")
    if "recovery_candidate" in a.method and not a.motor:
        p.error("Explicit candidate motor required")
    jobs = [
        (a.seed + i, i, g, method, a.motor)
        for i in range(a.count)
        for g in ("R1", "R2")
        for method in a.method
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for task in as_completed([pool.submit(trial, j) for j in jobs]):
            rows.append(task.result())
    rows.sort(key=lambda r: (r["seed"], r["group"], r["method"]))
    summary = []
    for group in ("R1", "R2"):
        for method in a.method:
            batch = [r for r in rows if r["group"] == group and r["method"] == method]
            valid = [
                r
                for r in batch
                if r["physical_pass"] and r["contact_conditioned_fall_s"] is not None
            ]
            times = [
                r["recovered_after_fall_s"]
                for r in valid
                if r["recovered_after_fall_s"] is not None
            ]
            summary.append(
                dict(
                    group=group,
                    method=method,
                    attempted=len(batch),
                    physical_pass=sum(r["physical_pass"] for r in batch),
                    eligible_falls=len(valid),
                    within_3s=sum(r["within_3s"] for r in valid),
                    completed=len(times),
                    recovery_to_walk=sum(r["recovery_to_walk_pass"] for r in valid),
                    completed_only_median_s=float(np.median(times)) if times else None,
                )
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out,
        dict(
            schema="duckfu.contact-recovery-evaluation.v1",
            split=a.split,
            evaluator_sha256=digest(__file__),
            course="Fixed physical rolling-motor coach. R1 backs off after measured contact fall; R2 continues driving. Not autonomous combat.",
            protocol="Common strict completion: upright>.94, height>.105m, loaded foot, no body-ground, speed<.16m/s for .30s; then stand .30s and walk .28m/s for 1s. Ten-second recovery deadline. All attempted/no-fall/numerical-rejected trials retained.",
            summary=summary,
            rows=rows,
        ),
    )
    print(json.dumps(summary, indent=2))
