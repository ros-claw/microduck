"""Strict integration replay; only initial state, motor inputs and weld releases."""

import argparse
import gzip
import json
from pathlib import Path
import mujoco
import numpy as np
from microduck_lab.arena.episode import STATE, sha


def replay(path):
    path = Path(path)
    audit = json.loads((path / "audit.json").read_text())
    for name, digest in audit["hashes"].items():
        if sha(path / name) != digest:
            raise ValueError("Artifact hash mismatch: " + name)
    m = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    if mujoco.__version__ != audit["mujoco_version"]:
        raise ValueError("MuJoCo version mismatch")
    d = mujoco.MjData(m)
    expected = mujoco.MjData(m)
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    mujoco.mj_setState(m, d, z["states"][0], STATE)
    qerr = verr = ferr = 0.0
    with gzip.open(path / "contacts.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= len(z["states"]):
                raise ValueError("Extra contact frames")
            mujoco.mj_setState(m, expected, z["states"][i], STATE)
            qerr = max(qerr, float(np.max(np.abs(d.qpos - expected.qpos))))
            verr = max(verr, float(np.max(np.abs(d.qvel - expected.qvel))))
            d.ctrl[:] = z["ctrl"][i]
            d.eq_active[:] = z["eq_active"][i]
            mujoco.mj_step(m, d)
            sample = json.loads(line)
            if sample["step"] != i or abs(sample["t"] - z["time"][i]) > 1e-9:
                raise ValueError("Contact time mismatch")
            if len(sample["contacts"]) != d.ncon:
                raise ValueError("Contact count mismatch")
            force = np.zeros(6)
            for j, row in enumerate(sample["contacts"]):
                c = d.contact[j]
                mujoco.mj_contactForce(m, d, j, force)
                if list(row[:2]) != [int(c.geom1), int(c.geom2)]:
                    raise ValueError("Contact identity mismatch")
                actual = [
                    c.dist,
                    *force.tolist(),
                    *c.pos.tolist(),
                    *c.frame[:3].tolist(),
                ]
                ferr = max(ferr, float(np.max(np.abs(np.asarray(row[2:]) - actual))))
    if i + 1 != len(z["states"]):
        raise ValueError("Missing contact frames")
    root_adr = int(m.jnt_qposadr[m.joint("cream/trunk_base_freejoint").id])
    qerr = max(
        qerr, float(np.max(np.abs(d.qpos[root_adr : root_adr + 7] - z["root"][-1])))
    )
    final_tiles = np.array(
        [d.qpos[m.jnt_qposadr[m.joint(f"tile_{j}_free").id] + 2] for j in range(9)]
    )
    qerr = max(qerr, float(np.max(np.abs(final_tiles - z["tile_z"][-1]))))
    result = dict(
        frames=len(z["states"]),
        qpos_max_error=qerr,
        qvel_max_error=verr,
        contact_max_error=ferr,
        passed=max(qerr, verr, ferr) < 1e-10,
    )
    (path / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError(str(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    print(json.dumps(replay(p.parse_args().run), indent=2))
