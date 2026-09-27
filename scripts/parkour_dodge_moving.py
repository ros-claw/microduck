"""Reproducible development sweep; all failures are retained."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, itertools
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
from concurrent.futures import ProcessPoolExecutor
import mujoco, numpy as np

sys.path.insert(0, str(Path.cwd() / "src"))
from microduck_lab.parkour.world import build_world, geom_vertices
from microduck_lab.parkour.hazards import Hazard


def job(args):
    vx, kp, kd, seed, direction = args
    m, d, r, tr = build_world(hazards=[Hazard("distant", "sweeper", 9.0, z=0.2)])
    rng = np.random.default_rng(seed)
    d.qpos[r.joint_qpos_idx] += rng.uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    stable = 0
    done = None
    samples = []
    envelopes = []
    targety = 0.0
    start = None
    for i in range(325):
        r.command[:] = 0
        r.active_policy = "stand" if i < 100 else "run"
        if i == 175:
            start = r.trunk_pos().copy()
            targety = start[1] + direction * tr.lane_width
            r.bank.mirror_run = direction < 0
        if i >= 100:
            pos = r.trunk_pos()
            vel = r.trunk_linvel()
            target = np.clip(kp * (targety - pos[1]) - kd * vel[1], -0.9, 0.9)
            r.command[:3] = [vx, 0, np.clip(5 * (target - r.trunk_yaw()), -3, 3)]
        r.step()
        mujoco.mj_step(m, d, 100)
        if i >= 175:
            pos = r.trunk_pos()
            up = d.xmat[r.trunk_body_id, 8]
            vel = r.trunk_linvel()
            stable = (
                stable + 0.02
                if abs(pos[1] - targety) < 0.025
                and abs(r.trunk_yaw()) < 0.15
                and up > 0.9
                and vel[0] > 0.15
                else 0
            )
            if done is None and stable >= 0.1:
                done = round((i - 174) * 0.02, 3)
            points = np.concatenate(
                [
                    geom_vertices(m, d, g)
                    for g in range(m.ngeom)
                    if m.geom(g).name.endswith("/hazard_proxy")
                ]
            )
            envelopes.append(
                [
                    round((i - 174) * 0.02, 3),
                    (points.min(0) - start).tolist(),
                    (points.max(0) - start).tolist(),
                ]
            )
            samples.append(
                [round((i - 174) * 0.02, 3), *(pos - start), up, r.trunk_yaw(), *vel]
            )
        if r.trunk_pos()[2] < -0.2:
            break
    return dict(
        vx=vx,
        kp=kp,
        kd=kd,
        seed=seed,
        direction=direction,
        lane_width=tr.lane_width,
        done=done,
        minup=min(x[4] for x in samples),
        final=samples[-1],
        samples=samples,
        envelopes=envelopes,
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = list(
            pool.map(job, itertools.product([0.7], [8], [2], range(10), [-1, 1]))
        )
    Path("artifacts/neon-escape-v2/dodge-proxy-envelopes.json").write_text(
        json.dumps(out)
    )
    for r in sorted(out, key=lambda x: x["done"] or 99)[:10]:
        print(
            {k: v for k, v in r.items() if k not in ("samples", "envelopes")},
            flush=True,
        )
