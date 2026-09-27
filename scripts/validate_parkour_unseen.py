"""Frozen-controller evaluation. No retries or selection inside this cohort."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from microduck_lab.demos.parkour_rehearsal import run

ROOT = Path(__file__).resolve().parents[1]


def trial(seed):
    report = run(
        ROOT / "policies/parkour_long_jump_v2.onnx",
        seed=seed,
        brain="jev",
        sweeper_phase=0.4,
        seconds=35,
        output=ROOT / f"artifacts/neon-escape-v2/unseen/{seed}",
    )
    return dict(
        seed=seed,
        passed=report["passed"],
        component_course_passed=report["component_course_passed"],
        stage=report["stage"],
        finished=report["finished"],
        hp=report["audit"]["hp"],
        max_penetration_m=report["audit"]["max_penetration_m"],
        p99_penetration_m=report["audit"]["p99_penetration_m"],
        skills=report["skills"],
        longest_chase_s=report["props"]["longest_chase_s"],
        decisions=[r for r in report["decisions"] if r["status"] == "accepted"],
    )


if __name__ == "__main__":
    target = ROOT / "artifacts/neon-escape-v2/unseen"
    target.mkdir(exist_ok=True)
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *sorted((ROOT / "src/microduck_lab/parkour").glob("*.py")),
            ROOT / "src/microduck_lab/demos/parkour_rehearsal.py",
            ROOT / "policies/parkour_long_jump_v2.onnx",
        ]
    }
    (target / "frozen-controller.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(trial, range(101, 107)))
    result = dict(
        seeds=list(range(101, 107)),
        runs=rows,
        core_passed=sum(r["passed"] for r in rows),
        escape_passed=sum(r["component_course_passed"] for r in rows),
        total=len(rows),
        scope="Unseen level jitter seeds with live Jev; fixed policy/controller and rotor phase; no retries. This is a small cohort, not a deployment safety guarantee.",
    )
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
