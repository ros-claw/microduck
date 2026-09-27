"""Frozen hard-contact static and moving-impact recovery development battery."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from parkour_recovery_battery import trial as static_trial
from parkour_trip_battery import trial as moving_trial


def job(args):
    kind, *parameters = args
    return dict(
        bucket=kind,
        result=static_trial(*parameters, True)
        if kind == "static"
        else moving_trial(parameters),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError("Choose a fresh evidence file")
    jobs = [
        ("static", k, s)
        for k, s in itertools.product(
            ["face_up", "face_down", "side_left", "side_right"], range(3)
        )
    ]
    jobs += [
        ("moving", 0.16, 0.0, -3.0, 0.2, phase, 0.10, True)
        for phase in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            r = row["result"]
            print(
                row["bucket"],
                r.get("kind"),
                r.get("seed"),
                r.get("passed"),
                r.get("recoveries"),
                flush=True,
            )
            Path(a.output).write_text(json.dumps(rows))
