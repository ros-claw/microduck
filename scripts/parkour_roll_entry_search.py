"""Test measured gait-phase entry, without stopping or changing robot state."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from parkour_runtime_battery import roll_trial


def job(args):
    phase, height, seed = args
    return roll_trial(seed, None, None, True, phase, height)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(
            job,
            itertools.product(
                ["left_rising", "right_rising", "left_falling", "right_falling"],
                [0.24, 0.27],
                range(3),
            ),
        ):
            out.append(r)
            print(
                r["entry_phase"],
                r["bar_height"],
                r["seed"],
                r["result"],
                r["final"],
                flush=True,
            )
            Path(a.output).write_text(json.dumps(out))
