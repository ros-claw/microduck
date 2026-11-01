"""Native-robot CPU finishing courses, not autonomous match results.

Initialization varies approach orientation and edge distance. Thereafter only
frozen neural motor targets drive both bodies; no root or collision edits.
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import mujoco
import numpy as np
from microduck_lab.combat.config import Physics
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.arena import build_match
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot, relative
from microduck_lab.combat.body_recovery import BodyRecoveryGuard
from microduck_lab.combat.world import digest
from microduck_lab.sim.composer import _yaw_quat


def trial(job):
    motor, sha, case, bounded = job
    seed, orientation, defense, heading = case
    cfg = MatchConfig(
        players=2,
        radius_m=0.85,
        arena="fixed_octagon",
        duration_s=8.0,
        candidate_motor_path=motor,
        candidate_motor_sha256=sha,
    )
    physics = Physics(dt=0.000125)
    m, d, ducks, spawns, teams = build_match(cfg, physics, seed)
    bank = next(iter(ducks.values())).bank
    bank.paths["walk_candidate"] = "policies/candidates/duckfu_locomotion_v1_200.onnx"
    bank.paths["recovery_candidate"] = "policies/candidates/duckfu_recovery_v2_150.onnx"
    actor, target = ducks["lavender"], ducks["cream"]
    rng = np.random.default_rng(seed)
    target_x = float(rng.uniform(0.48, 0.62))
    gap = float(rng.uniform(0.22, 0.30))
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    for duck, x, yaw in [
        (actor, target_x - gap, heading),
        (target, target_x, heading + orientation),
    ]:
        q = duck.trunk_qpos_adr
        d.qpos[q : q + 2] = rot @ np.array([x, 0.0])
        d.qpos[q + 3 : q + 7] = _yaw_quat(yaw)
    mujoco.mj_forward(m, d)
    audit = ContactAudit(m, ducks)
    guard = BodyRecoveryGuard("cream", learned=True, contact_only=False)
    invariant = [
        v.tobytes()
        for v in (
            m.body_mass,
            m.body_inertia,
            m.geom_contype,
            m.geom_conaffinity,
            m.actuator_forcerange,
        )
    ]
    body = []
    commands = []
    torque = 0.0
    external = True
    finite = True
    first_out = own_out = None
    drive_end = 2.8 if bounded else 3.8
    brake_latched = False
    reason = "TIME_LIMIT"
    origin = target.trunk_pos().copy()
    for k in range(round(7.0 / physics.dt)):
        now = k * physics.dt
        if k % physics.substeps == 0:
            if not np.isfinite(np.r_[d.qpos, d.qvel]).all():
                finite = False
                reason = "NONFINITE"
                break
            public = snapshot(ducks, audit, cfg)
            a, b = public["lavender"], public["cream"]
            if first_out is None and b.z < -0.10:
                first_out = now
            if own_out is None and a.z < -0.10:
                own_out = now
            if bounded and now >= 0.60 and a.boundary_distance < 0.14:
                brake_latched = True
            active = 0.60 <= now < drive_end and first_out is None and not brake_latched
            actor.active_policy = "dash_candidate" if now >= 0.60 else "walk_candidate"
            _, _, angle, _ = relative(a, b)
            actor.set_command(
                twist=(
                    0.4 if active else 0.0,
                    0.0,
                    float(np.clip(2 * angle, -0.5, 0.5)) if active else 0.0,
                ),
                head=(-0.3, -0.15, 0.0, 0.0) if active else (0.0, 0.0, 0.0, 0.0),
            )
            actor.step()
            recovering = guard.choose(now, b)
            target.active_policy = recovering or (
                "stand" if defense == "stand" or now < 0.60 else "walk_candidate"
            )
            twist = (0.0, 0.0, 0.0)
            if (
                recovering is None
                and defense != "stand"
                and now >= 0.60
                and first_out is None
            ):
                point = np.zeros(2) if defense == "inward" else np.array(a.xy)
                delta = point - b.xy
                norm = np.linalg.norm(delta)
                v = delta / max(1e-9, norm) * (0.14 if defense == "inward" else 0.20)
                cy, sy = math.cos(b.yaw), math.sin(b.yaw)
                angle = math.atan2(delta[1], delta[0]) - b.yaw
                twist = (
                    float(cy * v[0] + sy * v[1]),
                    float(-sy * v[0] + cy * v[1]),
                    float(
                        np.clip(
                            2 * math.atan2(math.sin(angle), math.cos(angle)), -1.0, 1.0
                        )
                    ),
                )
            target.set_command(twist=twist)
            target.step()
            body.append(
                [
                    [
                        *v.xy,
                        v.z,
                        v.upright,
                        v.speed,
                        float(v.foot_support),
                        v.boundary_distance,
                    ]
                    for v in [a, b]
                ]
            )
            commands.append([actor.last_action.copy(), target.last_action.copy()])
            if any(v["depth_m"] > 0.003 for v in audit.maxima.values()):
                reason = "PENETRATION"
                break
        mujoco.mj_step(m, d)
        audit.read(m, d, k, physics.dt, {})
        torque = max(torque, float(abs(d.actuator_force).max()))
        external &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
    b = np.array(body)
    tail = b[-20:, 0]
    summary = audit.summary()
    stable = bool(
        len(tail) == 20
        and (tail[:, 3] > 0.94).all()
        and tail[:, 4].max() < 0.18
        and tail[:, 5].mean() >= 0.80
        and tail[:, 6].min() > 0.03
    )
    impulse = sum(e["summed_normal_impulse_Ns"] for e in summary["episodes"])
    outward = rot @ np.array([1.0, 0.0])
    progress = float(((b[:, 1, :2] - origin[:2]) @ outward).max())
    unchanged = invariant == [
        v.tobytes()
        for v in (
            m.body_mass,
            m.body_inertia,
            m.geom_contype,
            m.geom_conaffinity,
            m.actuator_forcerange,
        )
    ]
    numeric = bool(
        finite
        and external
        and unchanged
        and summary["gate_proposal_pass"]
        and not any(w.number for w in d.warning)
        and torque <= 0.6405001
    )
    return dict(
        seed=seed,
        orientation_rad=orientation,
        defense=defense,
        heading_rad=heading,
        motor_sha256=sha,
        stage="INITIALIZATION_ONLY_NATIVE_OPPONENT_COURSE",
        public_edge_braking=bounded,
        drive_limit_s=drive_end - 0.60,
        target_x=target_x,
        gap_m=gap,
        target_out_s=first_out,
        own_out_s=own_out,
        stable_tail=stable,
        supported_finish=bool(
            first_out is not None and own_out is None and stable and impulse > 0.01
        ),
        max_target_progress_m=progress,
        positive_contact_impulse_Ns=impulse,
        actor_min_upright=float(b[:, 0, 3].min()),
        numeric_pass=numeric,
        torque_max_Nm=torque,
        physical=summary,
        finite=finite,
        external_forces_zero=external,
        model_unchanged=unchanged,
        reason=reason,
        body=b.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motor", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=210101)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument(
        "--bounded",
        action="store_true",
        help="Request zero commands at a .14 m edge margin or after 2.2 s, matching the finishing interface; retain unbounded courses separately as stress tests",
    )
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    sha = digest(a.motor)
    cases = [
        (a.seed_base + i, o, defense, (i % 4) * math.pi / 2)
        for i in range(a.seeds)
        for o in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for defense in ["stand", "inward", "pressure"]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed(
            [pool.submit(trial, (a.motor, sha, c, a.bounded)) for c in cases]
        ):
            r = f.result()
            rows.append(r)
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in [
                            "seed",
                            "orientation_rad",
                            "defense",
                            "target_out_s",
                            "supported_finish",
                            "numeric_pass",
                        ]
                    }
                ),
                flush=True,
            )
    groups = {
        s: dict(
            cases=sum(r["defense"] == s for r in rows),
            supported_finishes=sum(
                r["supported_finish"] for r in rows if r["defense"] == s
            ),
            target_exits=sum(
                r["target_out_s"] is not None for r in rows if r["defense"] == s
            ),
            stable=sum(r["stable_tail"] for r in rows if r["defense"] == s),
        )
        for s in ["stand", "inward", "pressure"]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                motor=a.motor,
                sha256=sha,
                public_edge_braking=a.bounded,
                groups=groups,
                all_numeric_pass=all(r["numeric_pass"] for r in rows),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
