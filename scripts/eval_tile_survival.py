"""Paired current-warning response vs hold; no tournament win-rate claim."""

import argparse
import json
from pathlib import Path
from microduck_lab.arena.survival import run
from microduck_lab.arena.verify import verify

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
rows = []
for seed in range(11, 21):
    for brain in ("warning", "hold"):
        path = out / f"{brain}-{seed}"
        report = run(path, seed, 14.0, "seeded", brain)
        re = verify(path)
        row = {
            k: report[k]
            for k in [
                "seed",
                "brain",
                "outcome",
                "duration",
                "penetration_max_m",
                "penetration_p99_m",
                "motor_force_max_Nm",
                "solver_warnings",
                "finite",
                "time_reset",
                "locked_tile_max_drift_m",
                "external_forces_zero",
                "recording_real_time_factor",
            ]
        }
        row["physics_pass"] = bool(
            row["finite"]
            and not row["time_reset"]
            and not any(row["solver_warnings"])
            and row["penetration_max_m"] <= 0.003
            and row["penetration_p99_m"] <= 0.0015
            and row["motor_force_max_Nm"] <= 0.640500001
            and row["locked_tile_max_drift_m"] < 0.001
            and row["external_forces_zero"]
        )
        row["replay"] = re
        row["events"] = report["events"]
        rows.append(row)
        print(
            seed,
            brain,
            row["outcome"]["status"],
            row["physics_pass"],
            round(row["penetration_max_m"] * 1000, 3),
            flush=True,
        )
        (out / "summary.json").write_text(
            json.dumps(
                dict(
                    runs=rows,
                    total=len(rows),
                    physics_pass=sum(x["physics_pass"] for x in rows),
                ),
                indent=2,
            )
            + "\n"
        )
