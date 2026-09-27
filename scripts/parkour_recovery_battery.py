"""Unassisted recovery from initialized fall poses and a physical force pulse."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, math
from pathlib import Path
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world, foot_support, duck_bounds


def trial(kind, seed):
    m, d, r, tr = build_world()
    rng = np.random.default_rng(seed)
    if kind != "moving_impact":
        axis = (
            np.array([1.0, 0, 0]) if kind.startswith("side") else np.array([0.0, 1, 0])
        )
        angle = {
            "face_up": -math.pi / 2,
            "face_down": math.pi / 2,
            "side_left": math.pi / 2,
            "side_right": -math.pi / 2,
        }[kind]
        angle += rng.uniform(-0.05, 0.05)
        d.qpos[r.trunk_qpos_adr + 3 : r.trunk_qpos_adr + 7] = [
            math.cos(angle / 2),
            *(axis * math.sin(angle / 2)),
        ]
        mujoco.mj_forward(m, d)
        lo, _ = duck_bounds(m, d)
        d.qpos[r.trunk_qpos_adr + 2] += 0.005 - lo[2]
        mujoco.mj_forward(m, d)
        # Settle under gravity, motors holding current joints, before timing recovery.
        mujoco.mj_step(m, d, 1500)
    else:
        for i in range(150):
            r.active_policy = "stand" if i < 100 else "run"
            r.command[0] = 0.6 if i >= 100 else 0
            r.step()
            mujoco.mj_step(m, d, 100)
        d.xfrc_applied[r.trunk_body_id, 1] = 5.0
        for i in range(5):
            r.step()
            mujoco.mj_step(m, d, 100)
        d.xfrc_applied[r.trunk_body_id, :] = 0
    initial = dict(pos=r.trunk_pos().tolist(), up=float(d.xmat[r.trunk_body_id, 8]))
    samples = []
    stable = 0.0
    completion = None
    for i in range(150):
        r.active_policy = "stand"
        r.set_command()
        r.step()
        mujoco.mj_step(m, d, 100)
        up = float(d.xmat[r.trunk_body_id, 8])
        speed = float(np.linalg.norm(r.trunk_linvel()))
        stable = (
            stable + 0.02 if up > 0.95 and foot_support(m, d) and speed < 0.12 else 0.0
        )
        if completion is None and stable >= 0.15:
            completion = round((i + 1) * 0.02, 3)
        samples.append(
            dict(
                t=round((i + 1) * 0.02, 3),
                pos=r.trunk_pos().tolist(),
                up=up,
                speed=speed,
            )
        )
        if r.trunk_pos()[2] < -0.3:
            break
    return dict(
        kind=kind,
        seed=seed,
        initial=initial,
        completion=completion,
        passed=completion is not None and completion <= 2.0,
        samples=samples,
        warnings=d.warning.number.tolist(),
    )


if __name__ == "__main__":
    out = []
    for kind in ["face_up", "face_down", "side_left", "side_right", "moving_impact"]:
        for seed in range(3):
            r = trial(kind, seed)
            out.append(r)
            print(
                kind, seed, r["initial"]["up"], r["completion"], r["passed"], flush=True
            )
    (ROOT / "artifacts/neon-escape-v2/recovery-battery.json").write_text(
        json.dumps(out)
    )
