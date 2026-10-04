"""Bounded physical sweeper energy/height study; no interventions on the robot."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from microduck_lab.demos.parkour_rehearsal import run


def job(args):
    target, height, mass, phase = args
    path = Path(target) / f"z{height:.2f}-m{mass:.1f}-p{phase:.1f}"
    r = run(
        "policies/parkour_long_jump_v2.onnx",
        hard_contacts=True,
        brain="rule",
        sweeper_height=height,
        sweeper_mass=mass,
        sweeper_torque=0.2,
        sweeper_phase=phase,
        output=path,
    )
    return dict(
        height=height,
        mass=mass,
        phase=phase,
        passed=r["passed"],
        finished=r["finished"],
        hp=r["audit"]["hp"],
        stage=r["stage"],
        recoveries=[s for s in r["skills"] if s["skill"] == "RECOVER"],
        chase=r["props"]["longest_chase_s"],
        boss_contacts=list(r["props"]["contacts"]),
        hazard_max_mm=r["audit"]["max_penetration_m"] * 1000,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    target = Path(a.output)
    target.mkdir(parents=True, exist_ok=False)
    out = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(
            job, itertools.product([str(target)], [0.16, 0.22], [0.4], [0.0, 0.4, 1.2])
        ):
            out.append(row)
            print("RESULT", json.dumps(row), flush=True)
            (target / "summary.json").write_text(json.dumps(out, indent=2))
