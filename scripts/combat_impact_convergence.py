"""Local dt/dt/2 impact replay with identical starting state and motor tape."""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import mujoco
import numpy as np
from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.world import build, digest
from microduck_lab.combat.episode import STATE, write_json
from microduck_lab.combat.contacts import ContactAudit


def compare(directory, out):
    directory = Path(directory)
    path = Path(out)
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["capture"]:
        raise ValueError("Full capture required")
    for n in ("states.npy", "controls.npz", "contacts_summary.json"):
        if digest(directory / n) != manifest["hashes"][n]:
            raise ValueError("Evidence differs")
    contacts = json.loads((directory / "contacts_summary.json").read_text())["episodes"]
    if not contacts:
        raise ValueError("A real impact is required")
    source_step = contacts[0]["start_step"]
    dt = manifest["physics"]["dt"]
    start = max(0, source_step - round(0.08 / dt))
    end = min(manifest["steps"], source_step + round(0.22 / dt))
    states = np.load(directory / "states.npy", mmap_mode="r")
    tape = np.load(directory / "controls.npz")["ctrl"]
    rows = []
    for factor in (1, 2):
        physics = replace(Physics(**manifest["physics"]), dt=dt / factor)
        m, d, ducks, _ = build(Config(**manifest["config"]), physics, manifest["seed"])
        mujoco.mj_setState(m, d, states[start], STATE)
        mujoco.mj_forward(m, d)
        audit = ContactAudit(m, ducks)
        initial = {n: duck.trunk_pos().copy() for n, duck in ducks.items()}
        for k in range((end - start) * factor):
            original_step = start + k // factor
            d.ctrl[:] = tape[original_step // round(0.02 / dt)]
            mujoco.mj_step(m, d)
            audit.read(m, d, k, physics.dt, {})
        summary = audit.summary()
        rows.append(
            dict(
                dt=physics.dt,
                window_s=(end - start) * dt,
                normal_impulse_Ns=sum(
                    e["summed_normal_impulse_Ns"] for e in summary["episodes"]
                ),
                penetration_max_m=summary["penetration_max_m"],
                duck_duck_max_m=summary["categories"]["duck_duck"]["maximum"][
                    "depth_m"
                ],
                contact_episodes=len(summary["episodes"]),
                final={
                    n: dict(
                        position_m=duck.trunk_pos().tolist(),
                        upright=float(-duck.projected_gravity()[2]),
                        displacement_m=(duck.trunk_pos() - initial[n]).tolist(),
                    )
                    for n, duck in ducks.items()
                },
            )
        )
    a, b = rows
    position_error = max(
        np.linalg.norm(
            np.array(a["final"][n]["position_m"]) - b["final"][n]["position_m"]
        )
        for n in a["final"]
    )
    impulse_error = abs(a["normal_impulse_Ns"] - b["normal_impulse_Ns"]) / max(
        b["normal_impulse_Ns"], 1e-12
    )
    result = dict(
        source_run=str(directory),
        source_manifest_sha256=digest(directory / "manifest.json"),
        source_first_contact_step=source_step,
        source_start_step=start,
        source_end_step=end,
        fixed_motor_tape=True,
        rows=rows,
        max_final_position_difference_m=float(position_error),
        relative_impulse_difference=float(impulse_error),
        contact_presence_agrees=all(r["normal_impulse_Ns"] > 0 for r in rows),
        limitations=[
            "Local fixed-input comparison, not a whole-match closed-loop equivalence proof."
        ],
    )
    write_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    compare(a.run, a.out)
