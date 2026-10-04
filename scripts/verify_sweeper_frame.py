"""Recompute the peak-force frame from its hashed, pre-integration state."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, json
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.evidence import verify_capture


def verify(source):
    p = Path(source)
    report = json.loads((p / "audit.json").read_text())
    proof = json.loads((p / "sweeper-clearance.json").read_text())
    verify_capture(p, report["capture"])
    if proof["capture"] != report["capture"]:
        raise ValueError("Capture mismatch")
    dense = p / "sweeper-highrate.npz"
    if hashlib.sha256(dense.read_bytes()).hexdigest() != proof["highrate_sha256"]:
        raise ValueError("Dense-state checksum mismatch")
    z = np.load(dense)
    m = mujoco.MjModel.from_binary_path(str(p / "scene.mjb"))
    d = mujoco.MjData(m)
    k = int(np.argmin(abs(z["time"] - proof["peak_state_time"])))
    for name in ("qpos", "qvel", "ctrl", "eq_active"):
        getattr(d, name)[:] = z[name][k]
    d.time = float(z["time"][k])
    mujoco.mj_forward(m, d)
    rod = m.geom("sweeper/geom").id
    body = m.geom(proof["peak_force_geom"]).id
    f = np.zeros(6)
    forces = []
    for i, c in enumerate(d.contact):
        if {c.geom1, c.geom2} == {rod, body}:
            mujoco.mj_contactForce(m, d, i, f)
            forces.append(float(f[0]))
    distance = float(mujoco.mj_geomDistance(m, d, rod, body, 0.5, np.zeros(6)))
    passed = bool(
        abs(d.time - proof["peak_state_time"]) < 1e-8
        and distance >= 0
        and forces
        and abs(max(forces) - proof["peak_force_N"]) < 1e-7
    )
    result = dict(
        passed=passed,
        capture=report["capture"],
        dense_sha256=proof["highrate_sha256"],
        state_time=d.time,
        geometry_distance_m=distance,
        normal_forces_N=forces,
        expected_peak_N=proof["peak_force_N"],
        method="Reinitialize only the hashed pre-step contact frame; forward dynamics must reproduce its measured normal force. This frame check complements the full uninterrupted input replay.",
    )
    (p / "sweeper-frame-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    verify(p.parse_args().source)
