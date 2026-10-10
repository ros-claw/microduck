"""Feed-forward lane trajectory screening under unchanged hard-contact physics."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.world import build_world, foot_support
from microduck_lab.parkour.skills import lane_command


def trial(job):
    duration, kp, seed, direction = job
    m, d, r, tr = build_world(hard_contacts=True)
    d.qpos[r.joint_qpos_idx] += np.random.default_rng(seed).uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    stable = 0.0
    done = None
    samples = []
    y0 = 0.0
    vnom = 0.5
    for i in range(300):
        r.active_policy = "stand" if i < 100 else "run"
        r.command[:] = 0.0
        if i == 175:
            y0 = float(r.trunk_pos()[1])
            vnom = max(0.35, float(r.trunk_linvel()[0]))
            r.bank.mirror_run = direction < 0
        if 100 <= i < 175:
            r.command[:3] = lane_command(r, 0.0)
        elif i >= 175:
            t = (i - 175) * 0.02
            u = np.clip(t / duration, 0.0, 1.0)
            dist = direction * tr.lane_width
            y = y0 + dist * (10 * u**3 - 15 * u**4 + 6 * u**5)
            vy = dist / duration * 30 * u * u * (1 - u) ** 2
            ay = dist / duration**2 * 60 * u * (1 - u) * (1 - 2 * u)
            yaw = np.arctan2(vy + kp * (y - r.trunk_pos()[1]), vnom)
            rate = vnom * ay / (vnom * vnom + vy * vy)
            r.command[:3] = [
                0.7,
                0.0,
                np.clip(rate + 5 * (yaw - r.trunk_yaw()), -3.0, 3.0),
            ]
        r.step()
        mujoco.mj_step(m, d, 100)
        if i >= 175:
            p = r.trunk_pos()
            up = float(d.xmat[r.trunk_body_id, 8])
            vel = r.trunk_linvel()
            good = (
                abs(p[1] - (y0 + direction * tr.lane_width)) < 0.025
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
                    *map(float, p),
                    up,
                    float(r.trunk_yaw()),
                    *map(float, vel),
                ]
            )
        if r.trunk_pos()[2] < -0.2:
            break
    return dict(
        duration=duration,
        kp=kp,
        seed=seed,
        direction=direction,
        lane_width_m=tr.lane_width,
        completion_s=done,
        min_up=min(x[4] for x in samples),
        final=samples[-1],
        samples=samples,
        passed=bool(
            done
            and done <= 1.2
            and min(x[4] for x in samples) > 0.9
            and samples[-1][3] > 0.09
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(
            trial, itertools.product([0.7, 0.9, 1.1], [2.0, 4.0], range(3), [-1, 1])
        ):
            out.append(r)
            print(
                r["duration"],
                r["kp"],
                r["seed"],
                r["direction"],
                r["completion_s"],
                r["passed"],
                flush=True,
            )
            Path(a.output).write_text(json.dumps(out))
