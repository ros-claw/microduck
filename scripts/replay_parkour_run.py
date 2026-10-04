"""Re-simulate recorded actuator inputs; never substitute recorded root states."""

import argparse, hashlib, json
from pathlib import Path
import mujoco, numpy as np


def replay(source):
    source = Path(source)
    report = json.loads((source / "audit.json").read_text())
    for name, expected in report["capture"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Capture hash mismatch: " + name)
    m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mujoco.MjData(m)
    inputs = dict(np.load(source / "inputs.npz"))
    states = dict(np.load(source / "trajectory.npz"))
    d.qpos[:] = inputs["initial_qpos"]
    d.qvel[:] = inputs["initial_qvel"]
    mujoco.mj_forward(m, d)
    index = 0
    max_qpos = max_qvel = 0.0
    for k in range(len(inputs["time"])):
        assert abs(d.time - inputs["time"][k]) < 1e-8
        d.ctrl[:] = inputs["ctrl"][k]
        d.eq_active[:] = inputs["eq_active"][k]
        d.xfrc_applied[:] = inputs["xfrc_applied"][k]
        m.geom_rgba[:] = inputs["geom_rgba"][k]
        for j in range(100):
            mujoco.mj_step(m, d)
            if j % 25 == 0:
                max_qpos = max(
                    max_qpos, float(np.max(np.abs(d.qpos - states["qpos"][index])))
                )
                max_qvel = max(
                    max_qvel, float(np.max(np.abs(d.qvel - states["qvel"][index])))
                )
                assert abs(d.time - states["time"][index]) < 1e-8
                index += 1
    out = dict(
        capture=report["capture"],
        frames=index,
        max_qpos_error=max_qpos,
        max_qvel_error=max_qvel,
        passed=max_qpos < 1e-9 and max_qvel < 1e-8,
        method="Initialize once, then replay ctrl, equality activation and bounded external force at 50 Hz; compare 200 Hz states.",
        warnings=d.warning.number.tolist(),
    )
    (source / "replay.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    replay(p.parse_args().source)
