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
from microduck_lab.parkour.world import build_world


def job(args):
    vx, kp, kd = args
    m, d, r, tr = build_world()
    stable = 0
    done = None
    samples = []
    for i in range(250):
        r.command[:] = 0
        r.active_policy = "stand" if i < 100 else "run"
        if i >= 100:
            pos = r.trunk_pos()
            vel = r.trunk_linvel()
            target = np.clip(kp * (0.2 - pos[1]) - kd * vel[1], -0.9, 0.9)
            r.command[:3] = [vx, 0, np.clip(5 * (target - r.trunk_yaw()), -3, 3)]
        r.step()
        mujoco.mj_step(m, d, 100)
        if i >= 100:
            pos = r.trunk_pos()
            up = d.xmat[r.trunk_body_id, 8]
            vel = r.trunk_linvel()
            stable = (
                stable + 0.02
                if abs(pos[1] - 0.2) < 0.025
                and abs(r.trunk_yaw()) < 0.15
                and up > 0.9
                and vel[0] > 0.15
                else 0
            )
            if done is None and stable >= 0.1:
                done = round((i - 99) * 0.02, 3)
            samples.append([round((i - 99) * 0.02, 3), *pos, up, r.trunk_yaw(), *vel])
        if r.trunk_pos()[2] < -0.2:
            break
    return dict(
        vx=vx,
        kp=kp,
        kd=kd,
        done=done,
        minup=min(x[4] for x in samples),
        final=samples[-1],
        samples=samples,
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = list(
            pool.map(
                job, itertools.product([0.5, 0.6, 0.7], [4, 6, 8], [0.5, 1, 1.5, 2])
            )
        )
    Path("artifacts/neon-escape-v2/dodge-feedback-sweep.json").write_text(
        json.dumps(out)
    )
    for r in sorted(out, key=lambda x: x["done"] or 99)[:12]:
        print({k: v for k, v in r.items() if k != "samples"}, flush=True)
