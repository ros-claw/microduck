"""Explicit development phase sweep; retain every failed run and source snapshot."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from microduck_lab.demos.parkour_rehearsal import run


def trial(job):
    target, phase = job
    r = run(
        "policies/parkour_long_jump_v2.onnx",
        hard_contacts=True,
        brain="rule",
        sweeper_phase=phase,
        output=Path(target) / f"phase-{phase:.2f}",
    )
    return dict(
        phase=phase,
        passed=r["passed"],
        finished=r["finished"],
        hp=r["audit"]["hp"],
        stage=r["stage"],
        skills=r["skills"],
        chase=r["props"]["longest_chase_s"],
        boss_contacts=list(r["props"]["contacts"]),
        hazard_max_mm=r["audit"]["max_penetration_m"] * 1000,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument(
        "--phases", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    )
    a = p.parse_args()
    target = Path(a.output)
    target.mkdir(parents=True, exist_ok=False)
    out = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(trial, [(str(target), v) for v in a.phases]):
            out.append(row)
            print("RESULT", json.dumps(row), flush=True)
            (target / "summary.json").write_text(json.dumps(out, indent=2))
