"""Smoke test: single duck stands with the official alpha_stand policy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime, PHYSICS_DT, SUBSTEPS

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
SCENE = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
POLICIES = ROOT / "microduck/policies"

model = mujoco.MjModel.from_xml_path(str(SCENE))
model.opt.timestep = PHYSICS_DT
data = mujoco.MjData(model)

bank = PolicyBank({
    "stand": str(POLICIES / "alpha_stand.onnx"),
    "walk": str(POLICIES / "alpha_walking.onnx"),
})
duck = DuckRuntime(model, data, bank, prefix="", name="solo")

# stand 3 s
for i in range(3 * 50):
    duck.step()
    mujoco.mj_step(model, data, SUBSTEPS)
print(f"stand 3s: pos={duck.trunk_pos().round(3)} upright={duck.is_upright()}")

# walk forward 2 s
duck.active_policy = "walk"
duck.set_command(twist=(0.15, 0.0, 0.0))
p0 = duck.trunk_pos()
for i in range(2 * 50):
    duck.step()
    mujoco.mj_step(model, data, SUBSTEPS)
p1 = duck.trunk_pos()
print(f"walk 2s: {p0.round(3)} -> {p1.round(3)} upright={duck.is_upright()}")
assert duck.is_upright(), "duck fell!"
print("SMOKE OK")
