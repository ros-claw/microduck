"""Reuse the existing actuator-only crouch/extend maneuver from a running start.
This is an explicitly procedural candidate, never described as a trained jump.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world, foot_support, duck_bounds
from microduck_lab.skills.jump import JumpSkill


def trial(args):
    gap, edge, vx, depth, crouch = args
    m, d, r, tr = build_world(gaps=((edge, edge + gap),) if gap else ())
    js = JumpSkill(
        r,
        crouch_depth=depth,
        crouch_time=crouch,
        extend_time=0.10,
        flight_time=0.01,
        land_time=0.01,
    )
    peak = 0.0
    upright_peak = 0.0
    flight = 0.0
    support = 0.0
    samples = []
    trigger = None
    for i in range(300):
        t = i * 0.02
        r.command[:] = 0
        if i < 100:
            r.active_policy = "stand"
        elif i < 150:
            r.active_policy = "run"
            r.command[:3] = [vx, 0, np.clip(-3 * r.trunk_yaw(), -1.5, 1.5)]
        else:
            if i == 150:
                js.trigger()
                trigger = r.trunk_pos().tolist()
            r.active_policy = "stand"
            js.update(0.02)
        r.step()
        mujoco.mj_step(m, d, 100)
        up = float(d.xmat[r.trunk_body_id, 8])
        pos = r.trunk_pos()
        if i >= 150:
            has_contact = False
            for c in d.contact:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if (
                        m.geom(a).name.startswith("floor/")
                        and m.body(m.geom_bodyid[b]).name.startswith("duck/")
                        and c.dist <= 0
                    ):
                        has_contact = True
            flight += 0.02 if not has_contact else 0.0
            lo, _ = duck_bounds(m, d, True)
            peak = max(peak, float(lo[2]))
            if up > 0.8:
                upright_peak = max(upright_peak, float(lo[2]))
            support = (
                support + 0.02
                if foot_support(m, d, "floor/1" if gap else None) and up > 0.9
                else 0.0
            )
            samples.append(
                dict(
                    t=round(t - 3 + 0.02, 3),
                    pos=pos.tolist(),
                    up=up,
                    ground_contact=has_contact,
                )
            )
        if pos[2] < -0.3:
            break
    return dict(
        gap=gap,
        edge=edge,
        vx=vx,
        depth=depth,
        crouch=crouch,
        trigger=trigger,
        peak=peak,
        upright_peak=upright_peak,
        airborne_seconds=flight,
        stable_support=support,
        success=bool(support > 0.3 and (not gap or pos[0] > edge + gap)),
        final=samples[-1],
        samples=samples,
    )


if __name__ == "__main__":
    jobs = list(
        itertools.product(
            [0.15],
            [0.30, 0.35, 0.40, 0.45, 0.50],
            [0.5, 0.7, 0.9],
            [0.55, 0.72, 1.0],
            [0.12, 0.22],
        )
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(trial, jobs):
            out.append(r)
            if r["success"]:
                print({k: v for k, v in r.items() if k != "samples"}, flush=True)
    (ROOT / "artifacts/neon-escape-v2/procedural-launch-sweep.json").write_text(
        json.dumps(out)
    )
    print("passes", sum(r["success"] for r in out), "/", len(out), flush=True)
