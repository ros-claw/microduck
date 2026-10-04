"""Four physical deployment probes; initial perturbations are not unseen games."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, hashlib
from pathlib import Path
import numpy as np, mujoco
from microduck_lab.parkour.level import build_level
from microduck_lab.parkour.joy import JoyDance


def probe(policy, output):
    policy = Path(policy).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    results = []
    for seed in range(4):
        m, d, r, _, _ = build_level(16, difficulty="arcade", hard_contacts=True)
        r.bank.paths["joy"] = str(policy)
        rng = np.random.default_rng(seed)
        # Perturb only at initialization, then integrate without state edits.
        d.qpos[r.joint_qpos_idx] += rng.uniform(-0.015, 0.015, 14)
        mujoco.mj_forward(m, d)
        for _ in range(100):
            r.active_policy = "stand"
            r.set_command()
            r.step()
            for _ in range(100):
                mujoco.mj_step(m, d)
        joy = JoyDance(float(d.time), finished_at=float(d.time) - 1.6)
        for _ in range(250):
            joy.update(m, d, r)
            r.step()
            joy.condition_targets(d, r)
            joy.control_sample(d, r)
            for _ in range(100):
                mujoco.mj_step(m, d)
                joy.sample(m, d, r)
        row = joy.report(m, d, r) | {
            "initialization_seed": seed,
            "warnings": d.warning.number.tolist(),
        }
        row["passed"] = row["passed"] and not d.warning.number.any()
        results.append(row)
        print(json.dumps(row), flush=True)
    data = dict(
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        probes=results,
        passed=all(x["passed"] for x in results),
        scope="Four small joint-perturbation deployment tuning probes after a 2 s standing settle; not independent unseen-game trials.",
        physics_step_s=0.0002,
        controller_hz=50,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    probe(a.policy, a.output)
