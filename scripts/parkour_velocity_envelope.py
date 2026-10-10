"""Measure diagonal commands and a feedback lane maneuver with official ONNX."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, math, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world


def trial(args):
    vx, vy, mode = args
    m, d, r, track = build_world()
    samples = []
    stable = 0.0
    completion = None
    start = None
    for i in range(250):
        t = i * 0.02
        r.command[:] = 0
        r.active_policy = "stand" if t < 2 else "run"
        if t >= 2:
            if start is None:
                start = r.trunk_pos().copy()
            error = 0.20 - (r.trunk_pos()[1] - start[1])
            if mode == "diagonal":
                yaw = 0.0
                lateral = vy if t < 3.2 else 0.0
            else:
                yaw = float(np.clip(math.atan2(error, 0.12), -1.0, 1.0))
                lateral = float(np.clip(2 * error, -vy, vy))
            r.command[:3] = [vx, lateral, np.clip(5 * (yaw - r.trunk_yaw()), -3.0, 3.0)]
        r.step()
        mujoco.mj_step(m, d, 100)
        up = float(d.xmat[r.trunk_body_id, 8])
        pos = r.trunk_pos()
        vel = r.trunk_linvel()
        if t >= 2:
            stable = (
                stable + 0.02
                if abs(pos[1] - start[1] - 0.2) < 0.025
                and abs(r.trunk_yaw()) < 0.15
                and up > 0.9
                and vel[0] > 0.15
                else 0.0
            )
            if completion is None and stable >= 0.1:
                completion = round(t - 2 + 0.02, 3)
            samples.append(
                dict(
                    t=round(t - 2 + 0.02, 3),
                    pos=(pos - start).tolist(),
                    up=up,
                    yaw=r.trunk_yaw(),
                    vx=float(vel[0]),
                )
            )
        if pos[2] < -0.2:
            break
    early = [s for s in samples if s["t"] <= 1.2]
    return dict(
        vx=vx,
        vy=vy,
        mode=mode,
        completion_seconds=completion,
        max_shift_1_2s=max(s["pos"][1] for s in early),
        min_up_1_2s=min(s["up"] for s in early),
        final=samples[-1],
        run_speed=float(np.mean([s["vx"] for s in samples[-50:]])),
        samples=samples,
    )


if __name__ == "__main__":
    jobs = list(
        itertools.product([0.3, 0.5, 0.7, 0.9], [0.25, 0.5, 0.8], ["diagonal", "steer"])
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(trial, jobs):
            out.append(r)
            print(
                {k: v for k, v in r.items() if k not in ("samples", "final")},
                flush=True,
            )
    (ROOT / "artifacts/neon-escape-v2/velocity-envelope.json").write_text(
        json.dumps(out)
    )
