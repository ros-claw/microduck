"""Rotating-bar impact battery, using the actual V2 finite-torque drive."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, math, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.hazards import Hazard, HazardDriver


def trial(args):
    speed, height, angle = args
    a = math.radians(angle)
    p = np.array([-0.18, -0.18])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = R @ p
    h = Hazard(
        "sweep",
        "sweeper",
        float(p[0]),
        float(p[1]),
        z=height,
        half_width=0.27,
        speed=-speed / 0.27,
    )
    m, d, r, tr = build_world(hazards=[h])
    driver = HazardDriver([h])
    jid = m.joint("sweep/joint").id
    qa = m.jnt_qposadr[jid]
    va = m.jnt_dofadr[jid]
    d.qpos[qa] = a
    mujoco.mj_forward(m, d)
    gid = m.geom("sweep/geom").id
    depth = []
    impulse = 0
    first = None
    up = 1.0
    energy0 = None
    energy1 = None
    vpost = None
    seconds = 2 + 1.6 / abs(h.speed) + 0.8
    for i in range(math.ceil(seconds * 50)):
        r.active_policy = "stand"
        r.step()
        if d.time >= 2:
            driver.step(m, d, 0)
        for _ in range(100):
            mujoco.mj_step(m, d)
            for ci, c in enumerate(d.contact):
                if gid not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == gid else c.geom1
                if not m.body(m.geom_bodyid[other]).name.startswith("duck/"):
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                if f[0] <= 1e-7:
                    continue
                if first is None:
                    first = dict(
                        t=float(d.time),
                        angular_velocity=float(d.qvel[va]),
                        target_body=m.body(m.geom_bodyid[other]).name,
                    )
                    energy0 = float(sum(d.energy))
                depth.append(max(0.0, -float(c.dist)))
                impulse += max(0.0, f[0]) * m.opt.timestep
            up = min(up, float(d.xmat[r.trunk_body_id, 8]))
            if first is not None and d.time >= first["t"] + 0.1 and vpost is None:
                vpost = r.trunk_linvel().tolist()
                energy1 = float(sum(d.energy))
    mx = max(depth, default=None)
    p99 = float(np.quantile(depth, 0.99)) if depth else None
    return dict(
        speed=speed,
        height=height,
        angle=angle,
        first=first,
        count=len(depth),
        max_penetration_m=mx,
        p99_penetration_m=p99,
        normal_impulse_Ns=float(impulse),
        min_upright=up,
        post_impact_velocity=vpost,
        energy_change_100ms=None if energy1 is None else energy1 - energy0,
        warnings=d.warning.number.tolist(),
        passed=bool(
            mx is not None
            and mx < 0.0015
            and p99 < 0.001
            and not d.warning.number.any()
        ),
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = []
        for r in pool.map(
            trial,
            itertools.product([0.2, 0.4, 0.6, 0.8], [0.055, 0.115, 0.23], [0, 30, 60]),
        ):
            results.append(r)
            print(
                r["speed"],
                r["height"],
                r["angle"],
                r["passed"],
                r["max_penetration_m"],
                flush=True,
            )
    (ROOT / "artifacts/neon-escape-v2/impact-rotating.json").write_text(
        json.dumps(
            dict(
                results=results,
                passed=sum(r["passed"] for r in results),
                count=len(results),
            ),
            indent=2,
        )
        + "\n"
    )
