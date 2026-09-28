"""Measure hard-contact route envelopes conditioned on lateral/velocity entry state."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.world import (
    build_world,
    geom_vertices,
    foot_support,
    duck_bounds,
)
from microduck_lab.parkour.hazards import Hazard
from microduck_lab.parkour.skills import Maneuver, lane_command


def trial(job):
    y0, vcommand, side, seed, *extra = job
    target_offset = extra[0] if extra else 0.20
    m, d, r, tr = build_world(
        hard_contacts=True, hazards=[Hazard("distant", "sweeper", 9.0, z=0.2)]
    )
    d.qpos[r.trunk_qpos_adr + 1] = y0
    d.qpos[r.joint_qpos_idx] += np.random.default_rng(seed).uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    for i in range(175):
        r.active_policy = "stand" if i < 100 else "run"
        r.command[:] = 0.0
        if i >= 100:
            r.command[:3] = lane_command(r, y0, vcommand)
        r.step()
        mujoco.mj_step(m, d, 100)
    start = r.trunk_pos().copy()
    velocity = r.trunk_linvel().copy()
    action = "TAKE_LEFT_ROUTE" if side > 0 else "TAKE_RIGHT_ROUTE"
    skill = Maneuver(action, float(d.time), target_y=side * target_offset)
    geometry = [g for g in range(m.ngeom) if m.geom(g).name.endswith("/hazard_proxy")]
    path = []
    support_path = []
    minup = 1.0
    done = None
    for i in range(160):
        status = skill.update(m, d, r)
        points = np.concatenate([geom_vertices(m, d, g) for g in geometry])
        path.append(
            [
                i * 0.02,
                (points.min(0) - start).tolist(),
                (points.max(0) - start).tolist(),
            ]
        )
        foot_low, foot_high = duck_bounds(m, d, feet_only=True)
        support_path.append(
            [i * 0.02, (foot_low - start).tolist(), (foot_high - start).tolist()]
        )
        minup = min(minup, float(d.xmat[r.trunk_body_id, 8]))
        if status != "RUNNING":
            done = float(d.time) - skill.started
            break
        r.step()
        mujoco.mj_step(m, d, 100)
        if r.trunk_pos()[2] < -0.2:
            break
    return dict(
        target_y_m=side * target_offset,
        entry_y=y0,
        entry_command=vcommand,
        side=side,
        seed=seed,
        action=action,
        start=start.tolist(),
        entry_velocity=velocity.tolist(),
        duration_s=done,
        status=skill.status,
        passed=skill.status == "SUCCESS" and minup > 0.95 and foot_support(m, d),
        min_up=minup,
        end=r.trunk_pos().tolist(),
        path=path,
        support_path=support_path,
    )


def build(rows):
    out = {
        "_profile": dict(
            lane_width_m=0.2,
            duck_floor_reference_s=0.002,
            duck_self_reference_s=0.0012,
            ground_margin_m=0.0002,
            scope="Three development perturbations per measured entry bucket, not unseen-course validation",
        )
    }
    for action in ["TAKE_LEFT_ROUTE", "TAKE_RIGHT_ROUTE"]:
        entries = []
        for y, v in itertools.product(
            sorted({r["entry_y"] for r in rows}),
            sorted({r["entry_command"] for r in rows}),
        ):
            group = [
                r
                for r in rows
                if r["action"] == action
                and r["entry_y"] == y
                and r["entry_command"] == v
            ]
            if len(group) != 3 or not all(r["passed"] for r in group):
                continue
            n = max(len(r["path"]) for r in group)
            path = []
            for i in range(n):
                box = [r["path"][min(i, len(r["path"]) - 1)] for r in group]
                low = np.min([b[1] for b in box], axis=0) - 0.015
                high = np.max([b[2] for b in box], axis=0) + 0.015
                path.append([round(i * 0.02, 3), low.tolist(), high.tolist()])
            support_path = []
            for i in range(n):
                box = [
                    r["support_path"][min(i, len(r["support_path"]) - 1)] for r in group
                ]
                support_path.append(
                    [
                        round(i * 0.02, 3),
                        (np.min([b[1] for b in box], axis=0) - 0.015).tolist(),
                        (np.max([b[2] for b in box], axis=0) + 0.015).tolist(),
                    ]
                )
            entries.append(
                dict(
                    entry_y_m=y,
                    entry_y_tolerance_m=0.035,
                    entry_vx_range_mps=[-0.08, 0.2]
                    if v == 0
                    else [0.15, 0.45]
                    if v == 0.5
                    else [0.35, 0.75],
                    entry_vx_mps=float(
                        np.median([r["entry_velocity"][0] for r in group])
                    ),
                    duration_s=max(r["duration_s"] for r in group),
                    development_seeds=3,
                    transfer_margin_m=0.015,
                    path=path,
                    support_path=support_path,
                )
            )
        targets = {
            r.get("target_y_m", 0.2 if action == "TAKE_LEFT_ROUTE" else -0.2)
            for r in rows
            if r["action"] == action
        }
        if len(targets) != 1:
            raise ValueError("Mixed route targets")
        out[action] = dict(entries=entries, target_y_m=targets.pop())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--entry-ys", nargs="+", type=float, default=[-0.15, 0.0, 0.15])
    p.add_argument("--entry-speeds", nargs="+", type=float, default=[0.0, 0.7])
    p.add_argument("--target-offset", type=float, default=0.2)
    a = p.parse_args()
    target = Path(a.output)
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(
            trial,
            itertools.product(
                a.entry_ys, a.entry_speeds, [-1, 1], range(3), [a.target_offset]
            ),
        ):
            rows.append(row)
            print(
                row["entry_y"],
                row["entry_command"],
                row["side"],
                row["seed"],
                row["passed"],
                row["duration_s"],
                flush=True,
            )
            (target / "trials.json").write_text(json.dumps(rows))
    out = build(rows)
    out["_profile"]["trials_sha256"] = hashlib.sha256(
        (target / "trials.json").read_bytes()
    ).hexdigest()
    (target / "envelopes.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        "qualified entry buckets",
        {a: len(out[a]["entries"]) for a in ["TAKE_LEFT_ROUTE", "TAKE_RIGHT_ROUTE"]},
        flush=True,
    )
