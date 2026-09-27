"""Compare exported roll policies in the exact hard-contact deployment model."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from parkour_runtime_battery import roll_trial


def trial(job):
    policy, seed = job
    row = roll_trial(seed, None, policy, True)
    row["passed"] = bool(
        row["result"]
        and row["result"]["status"] == "SUCCESS"
        and row["final"]["up"] > 0.95
        and row["final"]["pos"][2] > 0.10
        and row["max_floor_penetration_m"] < 0.0015
        and row["max_self_penetration_m"] < 0.0015
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Choose a new evidence path")
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(trial, itertools.product(a.policies, range(a.seeds))):
            out.append(row)
            print(
                row["policy"],
                row["seed"],
                row["result"],
                row["passed"],
                "floor_mm",
                row["max_floor_penetration_m"] * 1000,
                "self_mm",
                row["max_self_penetration_m"] * 1000,
                flush=True,
            )
            Path(a.output).write_text(json.dumps(out))
