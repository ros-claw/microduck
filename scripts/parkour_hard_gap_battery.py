"""Gap transfer battery for the deployed hard ground/self-contact profile."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from parkour_jump_handoff import trial

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--edge", type=float, default=0.06)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Use a fresh result path")
    jobs = [
        ("policies/parkour_long_jump_v2.onnx", g, s, a.edge, None, True)
        for g, s in itertools.product([0.12, 0.15, 0.18, 0.2], range(10))
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(trial, jobs):
            rows.append(row)
            print(row["gap"], row["seed"], row["passed"], row["result"], flush=True)
            Path(a.output).write_text(json.dumps(rows))
