"""V2 motor-policy batteries on a genuinely split track; no root forcing."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, json, sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world, duck_bounds, foot_support


def run(
    kind,
    gap=0.0,
    edge=0.16,
    duration=0.85,
    vx=0.5,
    vy=0.0,
    grip=False,
    seed=0,
    policy=None,
    joint_delay=0,
):
    m, d, r, track = build_world(gaps=((edge, edge + gap),) if gap else (), grip=grip)
    if policy:
        r.bank.paths["jump"] = str(policy)
    r.joint_vel_delay = joint_delay
    rng = np.random.default_rng(seed)
    # Only initialization writes the root pose. Small spawn perturbations test robustness.
    d.qpos[r.trunk_qpos_adr + 1] += rng.uniform(-0.003, 0.003)
    d.qpos[r.joint_qpos_idx] += rng.uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    samples = []
    supported = 0.0
    airborne = 0.0
    peak = 0.0
    upright_peak = 0.0
    start = None
    first_shift = None
    for i in range(350):
        t = i * 0.02
        elapsed = t - 2
        r.command[:] = 0
        if t < 2:
            r.active_policy = "stand"
        elif kind == "jump":
            r.active_policy = "jump" if elapsed < duration else "stand"
        elif kind == "roulade":
            r.active_policy = "run" if elapsed < 1 or elapsed > 3 else "roulade"
            if r.active_policy == "run":
                r.command[:3] = [vx, 0, np.clip(-3 * r.trunk_yaw(), -1.5, 1.5)]
        else:
            r.active_policy = "run"
            r.command[:3] = [
                vx,
                vy if elapsed < 1.2 else 0,
                np.clip(-3 * r.trunk_yaw(), -1.5, 1.5),
            ]
        r.step()
        mujoco.mj_step(m, d, 100)
        pos = r.trunk_pos()
        up = float(d.xmat[r.trunk_body_id, 8])
        support = foot_support(m, d, "floor/1" if gap else None)
        if t >= 2:
            if start is None:
                start = pos.copy()
            supported = supported + 0.02 if support and up > 0.9 else 0.0
            ground_contact = any(
                (
                    m.geom(c.geom1).name.startswith("floor/")
                    and m.body(m.geom_bodyid[c.geom2]).name.startswith("duck/")
                )
                or (
                    m.geom(c.geom2).name.startswith("floor/")
                    and m.body(m.geom_bodyid[c.geom1]).name.startswith("duck/")
                )
                for c in d.contact
                if c.dist <= 0
            )
            airborne += 0.02 if not ground_contact and up > 0.8 else 0.0
            if i % 2 == 0:
                lo, hi = duck_bounds(m, d, True)
                peak = max(peak, float(lo[2]))
                if up > 0.8:
                    upright_peak = max(upright_peak, float(lo[2]))
            if first_shift is None and abs(pos[1] - start[1]) >= 0.20:
                first_shift = round(elapsed + 0.02, 3)
        samples.append(
            dict(
                t=round(t + 0.02, 4),
                pos=pos.tolist(),
                up=up,
                yaw=r.trunk_yaw(),
                v=r.trunk_linvel().tolist(),
                support=support,
                policy=r.active_policy,
            )
        )
        if pos[2] < -0.3:
            break
    result = dict(
        kind=kind,
        gap=gap,
        edge=edge,
        duration=duration,
        vx=vx,
        vy=vy,
        grip=grip,
        seed=seed,
        track=track.__dict__,
        final=samples[-1],
        displacement=(r.trunk_pos() - start).tolist(),
        peak_entire_sole_clearance=peak,
        upright_sole_clearance=upright_peak,
        upright_airborne_seconds=airborne,
        stable_support_seconds=supported,
        first_20cm_shift_seconds=first_shift,
        run_speed_last_second=float(np.mean([s["v"][0] for s in samples[-50:]])),
        success=bool(
            supported >= 0.3
            and (not gap or (r.trunk_pos()[0] > edge + gap and airborne >= 0.04))
            and abs(r.trunk_pos()[1]) < track.width / 2
        ),
        warning_counts=d.warning.number.tolist(),
        samples=samples,
        policy=str(policy) if policy else "policies/jump.onnx",
        joint_delay=joint_delay,
        jump_sha256=hashlib.sha256(
            Path(policy or ROOT / "policies/jump.onnx").read_bytes()
        ).hexdigest(),
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["jump", "run", "dodge", "roulade"])
    p.add_argument("--gap", type=float, default=0)
    p.add_argument("--edge", type=float, default=0.16)
    p.add_argument("--duration", type=float, default=0.85)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0)
    p.add_argument("--grip", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy")
    p.add_argument("--joint-delay", type=int, default=0)
    p.add_argument("--output", required=True)
    a = vars(p.parse_args())
    output = Path(a.pop("output"))
    result = run(**a)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
