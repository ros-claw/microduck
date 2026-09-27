"""Development sweep; never report these tuned seeds as unseen validation."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from microduck_lab.demos.parkour_rehearsal import run


def trial(args):
    mass, phase = args
    r = run(
        "policies/parkour_long_jump_v2.onnx",
        seconds=29,
        output=f"artifacts/neon-escape-v2/course-body-search/m{mass}-p{phase:.3f}",
        sweeper_mass=mass,
        sweeper_phase=phase,
    )
    return {
        k: r[k]
        for k in [
            "sweeper_mass",
            "sweeper_phase",
            "passed",
            "component_course_passed",
            "finished",
            "stage",
            "skills",
        ]
    } | {"hp": r["audit"]["hp"], "max_penetration_m": r["audit"]["max_penetration_m"]}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as p:
        rows = list(
            p.map(trial, itertools.product([0.20], [i * 0.4 for i in range(8)]))
        )
    Path("artifacts/neon-escape-v2/course-body-search.json").write_text(
        json.dumps(rows, indent=2)
    )
