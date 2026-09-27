"""Deployment battery for an explicitly supplied, normalized candidate policy."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, argparse, itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_parkour import run


def job(args):
    policy, gap, seed = args
    return run(
        "jump",
        gap=gap,
        edge=0.10,
        duration=2.0,
        policy=policy,
        joint_delay=1,
        seed=seed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = []
        for r in pool.map(
            job,
            itertools.product(
                [str(Path(a.policy).resolve())], [0.12, 0.15, 0.18, 0.20], range(20)
            ),
        ):
            results.append(r)
            print(
                r["gap"],
                r["seed"],
                r["success"],
                round(r["final"]["yaw"], 3),
                flush=True,
            )
    summary = {
        str(gap): dict(
            passed=sum(r["success"] for r in results if r["gap"] == gap), count=20
        )
        for gap in [0.12, 0.15, 0.18, 0.20]
    }
    Path(a.output).write_text(json.dumps(dict(summary=summary, results=results)))
    print(summary, flush=True)
