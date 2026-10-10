"""Evaluate jump previews on actual approach states, without altering captures."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json
from pathlib import Path
import numpy as np
import mujoco
from microduck_lab.parkour.policies import ParkourPolicyBank
from microduck_lab.parkour.preview import preview_jump
from microduck_lab.sim.runtime import DuckRuntime
from microduck_lab.game.world import ASSETS, REPO

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("--output", required=True)
a = p.parse_args()
out = Path(a.output)
if out.exists():
    raise FileExistsError(out)
source = Path(a.source)
report = json.loads((source / "audit.json").read_text())
m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
d = mujoco.MjData(m)
bank = ParkourPolicyBank(
    {
        **{
            k: str(ASSETS / "microduck/policies" / v)
            for k, v in [
                ("stand", "alpha_stand.onnx"),
                ("run", "alpha_walking.onnx"),
                ("roulade", "roulade.onnx"),
            ]
        },
        "jump": str(REPO / "policies/parkour_long_jump_v2.onnx"),
    }
)
r = DuckRuntime(m, d, bank, prefix="duck/")
inputs = np.load(source / "inputs.npz")
d.qpos[:] = inputs["initial_qpos"]
d.qvel[:] = inputs["initial_qvel"]
mujoco.mj_forward(m, d)
rows = []
for i, t in enumerate(inputs["time"]):
    x = r.trunk_pos()[0]
    if 2.70 < x < 2.73:
        r.last_action = ((d.ctrl[r.act_ids] - r.default_pose) / r.action_scale).astype(
            np.float32
        )
        before = d.qpos.copy()
        for advance in (0.0, 0.08, 0.16, 0.24):
            for brake in (0.2, 0.4, 0.6):
                for settle in (0.2, 0.5):
                    result = preview_jump(
                        m,
                        d,
                        r,
                        3.15,
                        prepare_s=advance,
                        forward_speed=0.45,
                        brake_s=brake,
                        settle_s=settle,
                    )
                    assert np.array_equal(before, d.qpos)
                    rows.append(
                        dict(
                            t=float(t),
                            x=float(x),
                            vx=float(r.trunk_linvel()[0]),
                            **result,
                        )
                    )
                    print(rows[-1], flush=True)
        break
    d.ctrl[:] = inputs["ctrl"][i]
    d.eq_active[:] = inputs["eq_active"][i]
    d.xfrc_applied[:] = inputs["xfrc_applied"][i]
    m.geom_rgba[:] = inputs["geom_rgba"][i]
    mujoco.mj_step(m, d, 100)
    if x > 3.0:
        break
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2) + "\n")
