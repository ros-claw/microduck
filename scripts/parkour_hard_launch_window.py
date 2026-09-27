"""Measured hard-contact launch window, preserving all failures."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from parkour_jump_handoff import trial

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Use a new evidence file")
    rows = []
    jobs = [
        ("policies/parkour_long_jump_v2.onnx", 0.15, s, edge, None, True)
        for edge, s in itertools.product([0.04, 0.06, 0.08, 0.10], range(10))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(trial, jobs):
            rows.append(r)
            print(r["edge"], r["seed"], r["passed"], flush=True)
            Path(a.output).write_text(json.dumps(rows))
