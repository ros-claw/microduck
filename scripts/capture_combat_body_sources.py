"""Capture frozen-controller body-state sources; no new recovery is deployed."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path

from microduck_lab.combat.config import strict_load, Physics
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.episode import run, write_json


def trial(job):
    path, cfg, physics, seed = job
    result = run(path, cfg, physics, seed, capture=True)
    return dict(
        path=str(path.resolve()),
        seed=seed,
        swap=cfg.swap,
        result=result["result"],
        physical=result["physical"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    base = strict_load(a.config, V2Config)
    physics = strict_load(Path(a.config).parent / base.physics_profile, Physics)
    jobs = [
        (
            out / str(seed),
            replace(base, heading_rad=i % 4 * math.pi / 2, swap=bool(i % 2)),
            physics,
            seed,
        )
        for i, seed in enumerate(a.seeds)
    ]
    write_json(
        out / "protocol.json",
        dict(
            split=a.split,
            seeds=a.seeds,
            configs=[
                dict(seed=s, config=c.dict(), physics=p.__dict__) for _, c, p, s in jobs
            ],
            frozen_body_policies=True,
            source_purpose="Real predecessor states for body training/qualification; no outcome selection.",
        ),
    )
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for task in as_completed([pool.submit(trial, j) for j in jobs]):
            row = task.result()
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(jobs),
                        seed=row["seed"],
                        result=row["result"],
                    )
                ),
                flush=True,
            )
    write_json(out / "report.json", dict(split=a.split, rows=rows))
