"""Reproducible development sweep; all failures are retained."""

import sys, json, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(Path.cwd() / "scripts"))
from probe_parkour import run


def job(params):
    r = run(**params)
    return r


if __name__ == "__main__":
    jobs = [
        dict(kind="jump", gap=g, edge=e, duration=t)
        for g, e, t in itertools.product(
            [0.12, 0.15, 0.18, 0.20],
            [0.08, 0.12, 0.16, 0.20],
            [0.65, 0.85, 1.5, 1.8, 2.0],
        )
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(job, jobs):
            out.append(r)
            print(
                "jump",
                r["gap"],
                r["edge"],
                r["duration"],
                r["success"],
                round(r["peak_entire_sole_clearance"], 3),
                flush=True,
            )
    Path("artifacts/neon-escape-v2/jump-gap-sweep.json").write_text(json.dumps(out))
