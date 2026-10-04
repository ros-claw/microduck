"""Controlled ground-contact transfer study; does not change production defaults."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import parkour_runtime_battery as battery

PROFILES = {
    "stiff_002": (0.002, [0.99, 0.999, 0.0005, 0.5, 2], 0.0002),
    "native_imp_002": (0.002, [0.9, 0.95, 0.001, 0.5, 2], 0.0002),
    "native_imp_001": (0.001, [0.9, 0.95, 0.001, 0.5, 2], 0.0002),
    "native_imp_004": (0.004, [0.9, 0.95, 0.001, 0.5, 2], 0.0002),
    "mid_imp_002": (0.002, [0.97, 0.99, 0.0005, 0.5, 2], 0.0002),
    "native_imp_002_zero_margin": (0.002, [0.9, 0.95, 0.001, 0.5, 2], 0.0),
}


def trial(job):
    name, seed = job
    ref, imp, margin = PROFILES[name]
    original = battery.build_world

    def configured(**kw):
        m, d, r, tr = original(**kw)
        for p in range(m.npair):
            names = [m.geom(g).name for g in (m.pair_geom1[p], m.pair_geom2[p])]
            if any(n.startswith("duck/") for n in names) and any(
                n.startswith(("floor/", "rail/")) for n in names
            ):
                m.pair_solimp[p] = imp
                m.pair_margin[p] = margin
        return m, d, r, tr

    battery.build_world = configured
    try:
        row = battery.roll_trial(seed, ref)
        row["profile"] = name
        row["contact_parameters"] = dict(solref=[ref, 1], solimp=imp, margin=margin)
        return row
    finally:
        battery.build_world = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    jobs = [(p, s) for p in PROFILES for s in range(args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(trial, jobs):
            rows.append(r)
            print(
                r["profile"],
                r["seed"],
                r["result"],
                "max_mm",
                r["max_floor_penetration_m"] * 1000,
                flush=True,
            )
            Path(args.output).write_text(json.dumps(rows))
