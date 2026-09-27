"""Search valid mechanical impact configurations; retain all successes/failures."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import itertools, json, math
from concurrent.futures import ProcessPoolExecutor
from parkour_trip_battery import trial, ROOT

if __name__ == "__main__":
    jobs = itertools.product(
        [0.05, 0.08, 0.12],
        [0.0, 0.12],
        [-2.2, 2.2],
        [0.10, 0.20],
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(trial, jobs):
            out.append(r)
            if r["recoveries"]:
                print(
                    r["height"],
                    r["y"],
                    r["speed"],
                    r["mass"],
                    round(r["phase"], 2),
                    r["recoveries"],
                    round(r["audit"]["max_penetration_m"] * 1000, 3),
                    r["final"]["pos"],
                    flush=True,
                )
    (ROOT / "artifacts/neon-escape-v2/impact-recovery-search.json").write_text(
        json.dumps(out)
    )
