"""Timestep calibration plus ten seeded DG-01 release/fall regressions."""

import argparse
import gzip
import json
from pathlib import Path
import mujoco
from microduck_lab.arena.episode import run


def categories(path):
    m = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    result = {}
    with gzip.open(path / "contacts.jsonl.gz", "rt") as f:
        for line in f:
            for a, b, dist, *_ in json.loads(line)["contacts"]:
                names = [
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                    for i in (a, b)
                ]
                category = (
                    "receiver"
                    if "receiver" in names
                    else "duck_tile"
                    if any(n.startswith("tile_") for n in names)
                    and any(n.startswith("cream/") for n in names)
                    else "other"
                )
                result[category] = max(result.get(category, 0), max(0, -dist))
    return result


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default="artifacts/duckverse/battery")
p.add_argument("--dt", type=float, default=0.0005)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
rows = []
for dt in (0.005, 0.002, 0.001, 0.0005):
    path = out / f"dt{dt}"
    report = run(path, 11, dt)
    report["category_max_m"] = categories(path)
    rows.append(report)
(out / "timestep-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
rows = []
for seed in range(11, 21):
    path = out / f"seed{seed}"
    report = run(path, seed, a.dt)
    report["category_max_m"] = categories(path)
    report["gate_pass"] = bool(
        report["finite"]
        and not report["time_reset"]
        and report["motor_force_max_Nm"] <= 0.640500001
        and not any(report["solver_warnings"])
        and report["centre_supported_before_warning"]
        and report["locked_tile_max_drift_m"] < 0.001
        and report["root_final"][2] < -0.4
        and report["penetration_p99_m"] <= 0.0015
        and report["penetration_max_m"] <= 0.003
    )
    rows.append(report)
    print(seed, report["gate_pass"], report["category_max_m"], flush=True)
(out / "summary.json").write_text(
    json.dumps(
        dict(passed=sum(r["gate_pass"] for r in rows), total=len(rows), runs=rows),
        indent=2,
    )
    + "\n"
)
