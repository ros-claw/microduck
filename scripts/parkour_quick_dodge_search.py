"""Screen 20 cm lane changes on the hard-contact profile, retaining failures."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.world import build_world, foot_support

PROFILES = [
    (0.7, 8.0, 2.0, 0.9, 0.0),
    (0.9, 8.0, 2.0, 0.9, 0.0),
    (1.1, 8.0, 2.0, 0.9, 0.0),
    (0.7, 12.0, 2.5, 1.1, 0.0),
    (0.9, 12.0, 2.5, 1.1, 0.0),
    (1.1, 12.0, 2.5, 1.1, 0.0),
    (0.7, 8.0, 2.0, 0.9, 0.1),
    (0.9, 8.0, 2.0, 0.9, 0.1),
    (0.7, 12.0, 2.5, 1.1, 0.1),
]


def trial(job):
    profile, seed, direction = job
    vx, kp, kd, yawcap, vymax = profile
    m, d, r, tr = build_world(hard_contacts=True)
    d.qpos[r.joint_qpos_idx] += np.random.default_rng(seed).uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    stable = 0.0
    done = None
    start = None
    target_y = 0.0
    samples = []
    for i in range(325):
        r.active_policy = "stand" if i < 100 else "run"
        r.command[:] = 0.0
        if i == 175:
            start = r.trunk_pos().copy()
            target_y = start[1] + direction * tr.lane_width
            r.bank.mirror_run = direction < 0
        if i >= 100:
            error = target_y - r.trunk_pos()[1]
            yaw = np.clip(kp * error - kd * r.trunk_linvel()[1], -yawcap, yawcap)
            r.command[:3] = [
                vx,
                np.clip(error * 2, -vymax, vymax),
                np.clip(5 * (yaw - r.trunk_yaw()), -3, 3),
            ]
        r.step()
        mujoco.mj_step(m, d, 100)
        if i >= 175:
            p = r.trunk_pos()
            up = float(d.xmat[r.trunk_body_id, 8])
            vel = r.trunk_linvel()
            good = (
                abs(p[1] - target_y) < 0.025
                and abs(r.trunk_yaw()) < 0.15
                and up > 0.95
                and vel[0] > 0.15
                and foot_support(m, d)
            )
            stable = stable + 0.02 if good else 0.0
            if done is None and stable >= 0.12:
                done = (i - 174) * 0.02
            samples.append(
                [
                    float(d.time) - 3.5,
                    *map(float, p - start),
                    up,
                    float(r.trunk_yaw()),
                    *map(float, vel),
                ]
            )
        if r.trunk_pos()[2] < -0.2:
            break
    return dict(
        profile=profile,
        seed=seed,
        direction=direction,
        lane_width_m=tr.lane_width,
        duration_s=done,
        min_up=min(x[4] for x in samples),
        final=samples[-1],
        samples=samples,
        passed=bool(
            done
            and done <= 1.2
            and min(x[4] for x in samples) > 0.9
            and samples[-1][3] > -0.03
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(trial, itertools.product(PROFILES, range(a.seeds), [-1, 1])):
            out.append(r)
            print(
                r["profile"],
                r["seed"],
                r["direction"],
                r["duration_s"],
                r["passed"],
                flush=True,
            )
            Path(a.output).write_text(json.dumps(out))
