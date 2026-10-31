"""Paired body recovery and recovery-to-walk checks from real saved states.

Flat-floor component course: a physical second duck stands far away. Raw
predecessor pose, qvel and last action initialize each trial exactly once.
This does not claim a free-game rescue or edge recovery.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.world import build, digest
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.episode import source_hashes, write_json


@lru_cache(maxsize=1)
def template():
    cfg = Config(defense="absent", radius_m=1.5)
    physics = Physics(dt=0.000125, native_ccd=False)
    m, d, ducks, _ = build(cfg, physics, 23)
    return (
        cfg,
        physics,
        m,
        d.qpos.copy(),
        d.qvel.copy(),
        d.ctrl.copy(),
        next(iter(ducks.values())).bank,
    )


def trial(job):
    path, index, method, motor, timeout = job
    frozen_source = source_hashes()
    motor_digest = digest(motor) if method == "recovery_candidate" else None
    cfg, physics, m, q0, v0, u0, bank = template()
    from microduck_lab.sim.runtime import DuckRuntime

    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = v0
    d.ctrl[:] = u0
    ducks = {
        n: DuckRuntime(m, d, bank, prefix=n + "/", name=n)
        for n in ("lavender", "cream")
    }
    duck, other = ducks.values()
    # Keep the second physical robot outside the entire saved-state envelope.
    # Do not silently evaluate a component course with an interfering opponent.
    oq = other.trunk_qpos_adr
    d.qpos[oq : oq + 2] = [0.9, 1.0]
    if motor:
        bank.paths["recovery_candidate"] = motor
        metadata = bank.get("recovery_candidate").get_modelmeta().custom_metadata_map
        assert metadata.get("duckfu_scope") == "recorded_contact_recovery"
    bank.paths["walk_candidate"] = "policies/candidates/duckfu_locomotion_v1_200.onnx"
    data = np.load(path)
    root, velocity = data["raw_root"][index], data["raw_world_velocity"][index]
    q, v = duck.trunk_qpos_adr, duck.trunk_qvel_adr
    d.qpos[q : q + 7] = root
    d.qvel[v : v + 3] = velocity[:3]
    inverse = np.r_[root[3], -root[4:7]]
    local_angular = np.zeros(3)
    mujoco.mju_rotVecQuat(local_angular, velocity[3:], inverse)
    d.qvel[v + 3 : v + 6] = local_angular
    d.qpos[duck.joint_qpos_idx] = data["joints"][index]
    d.qvel[duck.joint_qvel_idx] = data["joint_velocity"][index]
    duck.last_action[:] = data["last_action"][index]
    d.ctrl[duck.act_ids] = duck.default_pose + duck.last_action
    mujoco.mj_forward(m, d)
    audit = ContactAudit(m, ducks)
    audit.read(m, d, 0, 0.0, {})
    guard = RecoveryGuard("lavender", allow_roll=False)
    stable_since, complete, walk_start, walk_origin, walk_yaw = (
        None,
        None,
        None,
        None,
        None,
    )
    body = []
    maximum_torque = 0.0
    finite = True
    external_zero = True
    for step in range(round((timeout + 2.0) / physics.dt)):
        now = step * physics.dt
        if step % physics.substeps == 0:
            b = snapshot(ducks, audit, cfg)["lavender"]
            good = (
                b.upright > 0.94
                and b.z > 0.105
                and b.foot_support
                and not b.body_ground
                and b.speed < 0.16
            )
            stable_since = (
                (now if stable_since is None else stable_since) if good else None
            )
            if (
                complete is None
                and stable_since is not None
                and now - stable_since >= 0.30 - 1e-9
                and now <= timeout
            ):
                complete = now
            if complete is not None and now >= complete + 0.30 - 1e-9:
                if walk_start is None:
                    walk_start, walk_origin, walk_yaw = now, np.array(b.xy), b.yaw
                duck.active_policy = "walk_candidate"
                duck.set_command(twist=(0.28, 0, 0))
            elif complete is not None:
                duck.active_policy = "stand"
                duck.set_command()
            else:
                duck.active_policy = (
                    (guard.choose(now, b) or "stand") if method == "guard" else method
                )
                duck.set_command()
            duck.step()
            other.active_policy = "stand"
            other.set_command()
            other.step()
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
            if walk_start is not None and now - walk_start >= 1.0 - 1e-9:
                break
            if any(c["depth_m"] > 0.003 for c in audit.maxima.values()):
                break
        mujoco.mj_step(m, d)
        audit.read(m, d, step, physics.dt, {})
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        external_zero &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
        maximum_torque = max(maximum_torque, float(np.abs(d.actuator_force).max()))
    contacts = audit.summary()
    passed = bool(
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
    qualified_walk = bool(
        walk
        and walk[-1][0] - walk_start >= 0.98
        and min(r[4] for r in walk) > 0.90
        and walk[-1][4] > 0.94
        and progress > 0.05
    )
    if frozen_source != source_hashes() or (
        motor_digest is not None and motor_digest != digest(motor)
    ):
        raise RuntimeError("Motor or actor source changed during recovery check")
    return dict(
        state_index=index,
        method=method,
        policy_sha256=motor_digest,
        source_sha256=frozen_source,
        complete_s=complete,
        within_3s=complete is not None and complete <= 3.0,
        recovery_to_walk_pass=qualified_walk,
        walk_forward_progress_m=progress,
        physical_pass=passed,
        maximum_torque_Nm=maximum_torque,
        contacts=contacts,
        body=body,
        guard_events=guard.events,
        other_initial_xy=[0.9, 1.0],
        initial_raw_root=root.tolist(),
        initial_raw_world_velocity=velocity.tolist(),
        initial_last_action=data["last_action"][index].tolist(),
        maximum_penetration_m=contacts["penetration_max_m"],
        warnings=[int(w.number) for w in d.warning],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--method",
        action="append",
        choices=["guard", "stand", "sitstand", "recovery_candidate"],
        required=True,
    )
    p.add_argument("--motor")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--timeout", type=float, default=6.0)
    p.add_argument("--offset", type=float, action="append")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    if "recovery_candidate" in a.method and not a.motor:
        p.error("Candidate requires explicit motor")
    meta = json.loads(Path(a.states).with_suffix(".json").read_text())
    offsets = a.offset if a.offset is not None else [0.0, 0.6]
    indices = [
        i
        for i, r in enumerate(meta["rows"])
        if r["flat_floor_eligible"]
        and any(abs(r["offset_s"] - t) < 1e-6 for t in offsets)
    ]
    jobs = [
        (a.states, i, method, a.motor, a.timeout)
        for i in indices
        for method in a.method
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for task in as_completed([pool.submit(trial, j) for j in jobs]):
            row = task.result()
            row["source"] = meta["rows"][row["state_index"]]
            rows.append(row)
    rows.sort(key=lambda r: (r["state_index"], r["method"]))
    summary = []
    for method in a.method:
        batch = [r for r in rows if r["method"] == method]
        valid = [r for r in batch if r["physical_pass"]]
        times = [r["complete_s"] for r in valid if r["complete_s"] is not None]
        summary.append(
            dict(
                method=method,
                trials=len(batch),
                physical_pass=len(valid),
                within_3s=sum(r["within_3s"] for r in valid),
                completed=sum(r["complete_s"] is not None for r in valid),
                recovery_to_walk=sum(r["recovery_to_walk_pass"] for r in valid),
                completed_only_median_s=float(np.median(times)) if times else None,
                max_depth_m=max((r["maximum_penetration_m"] for r in batch), default=0),
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out,
        dict(
            schema="duckfu.recorded-recovery-evaluation.v1",
            split=meta["split"],
            states_sha256=digest(a.states),
            evaluator_sha256=digest(__file__),
            protocol=dict(
                offsets=offsets,
                timeout_s=a.timeout,
                post_recovery_settle_s=0.30,
                walk_test_s=1.0,
                complete="upright>0.94, trunk height>0.105m, loaded foot, no body-ground contact, XY speed<0.16m/s sustained 0.30s",
                course="Flat 1.5m half-width stage; other duck stands far away. Not a free match or an edge-rescue claim.",
            ),
            summary=summary,
            rows=rows,
        ),
    )
    print(json.dumps(summary, indent=2))
