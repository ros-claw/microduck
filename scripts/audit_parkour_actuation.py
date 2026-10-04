"""Check recorded motor actuation for hidden robot forces or external welds."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.evidence import verify_capture


def audit(source):
    p = Path(source)
    a = json.loads((p / "audit.json").read_text())
    verify_capture(p, a["capture"])
    m = mujoco.MjModel.from_binary_path(str(p / "scene.mjb"))
    u = np.load(p / "inputs.npz")
    bodies = [i for i in range(m.nbody) if m.body(i).name.startswith("duck/")]
    actuators = [i for i in range(m.nu) if m.actuator(i).name.startswith("duck/")]
    forces = float(abs(u["xfrc_applied"][:, bodies, :]).max())
    cross = []
    for i in range(m.neq):
        if (
            m.eq_type[i] not in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT)
            or m.eq_objtype[i] != mujoco.mjtObj.mjOBJ_BODY
        ):
            continue
        if (m.eq_obj1id[i] in bodies) != (m.eq_obj2id[i] in bodies):
            cross.append(m.equality(i).name)
    limit = m.actuator_forcerange[actuators]
    limited = bool(
        m.actuator_forcelimited[actuators].all()
        and np.allclose(limit, [-0.6405, 0.6405], atol=1e-9)
    )
    result = dict(
        passed=forces == 0 and not cross and limited,
        capture=a["capture"],
        robot_body_count=len(bodies),
        motor_count=len(actuators),
        max_robot_applied_external_wrench=forces,
        robot_to_environment_body_equalities=cross,
        all_robot_motor_limits_Nm=limit.tolist(),
        method="Inspect every recorded applied-wrench row, all motor force limits and external body connect/weld declarations. Uninterrupted input replay separately rules out mid-run state edits.",
    )
    (p / "actuation-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    audit(p.parse_args().source)
