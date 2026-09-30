"""Re-simulate all inputs and audit every force-bearing contact category."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from microduck_lab.parkour.evidence import verify_capture, contact_gate
from pathlib import Path
import mujoco, numpy as np


def audit(source):
    source = Path(source)
    report = json.loads((source / "audit.json").read_text())
    verify_capture(source, report["capture"])
    m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mujoco.MjData(m)
    u = dict(np.load(source / "inputs.npz"))
    d.qpos[:] = u["initial_qpos"]
    d.qvel[:] = u["initial_qvel"]
    mujoco.mj_forward(m, d)
    records = {}
    depths = {}
    force = np.zeros(6)
    for k in range(len(u["time"])):
        d.ctrl[:] = u["ctrl"][k]
        d.eq_active[:] = u["eq_active"][k]
        d.xfrc_applied[:] = u["xfrc_applied"][k]
        for _ in range(100):
            mujoco.mj_step(m, d)
            for i, c in enumerate(d.contact):
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] <= 1e-7:
                    continue
                bodies = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                duck = sum(n.startswith("duck/") for n in bodies)
                world = "world" in bodies
                category = (
                    "duck_self"
                    if duck == 2
                    else "duck_floor"
                    if duck and world
                    else "duck_hazard"
                    if duck
                    else "prop_floor"
                    if world
                    else "prop_prop"
                )
                depth = max(0.0, -float(c.dist))
                depths.setdefault(category, []).append(depth)
                names = tuple(sorted(m.geom(g).name for g in (c.geom1, c.geom2)))
                key = " × ".join(names)
                rec = records.setdefault(
                    key,
                    dict(
                        category=category,
                        samples=0,
                        max_penetration_m=0.0,
                        normal_impulse_Ns=0.0,
                        peak_time_s=0.0,
                    ),
                )
                rec["samples"] += 1
                rec["normal_impulse_Ns"] += float(force[0]) * m.opt.timestep
                if depth > rec["max_penetration_m"]:
                    rec["max_penetration_m"] = depth
                    rec["peak_time_s"] = float(d.time)
    result = dict(
        categories={
            name: dict(
                samples=len(values),
                max_penetration_m=max(values),
                p99_penetration_m=float(np.quantile(values, 0.99)),
            )
            for name, values in depths.items()
        },
        pairs=records,
        warnings=d.warning.number.tolist(),
        scope="All force-bearing contacts at every 0.2 ms step, from replayed actuator inputs; includes native ground compliance separately from hazard contact.",
    )
    result["capture"] = report["capture"]
    result["publication_gate"] = contact_gate(result["categories"], result["warnings"])
    result["escape_contact_gate"] = contact_gate(
        result["categories"], result["warnings"], require_hazard_contact=False
    )
    (source / "all-contacts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["categories"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    audit(p.parse_args().source)
