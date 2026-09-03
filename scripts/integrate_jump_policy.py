"""Integrate a trained jump policy into the lab.

After training:
  cd ~/workspace/microduck/microduck_rl
  uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <run>  # or --checkpoint

This script: copies the ONNX into the lab's policy bank and runs a quick
single-duck hop check (feet clearance + lands upright).
"""
import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from microduck_lab.sim.composer import compose_world, DuckSpec
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
LAB = ROOT / "rosclaw-microduck"
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=str, help="path to the exported jump ONNX")
    args = ap.parse_args()

    dest = LAB / "policies"
    dest.mkdir(exist_ok=True)
    target = dest / "jump.onnx"
    shutil.copy(args.onnx, target)
    print(f"copied {args.onnx} → {target}")

    # smoke: single duck, hop, measure clearance + landing
    bank = PolicyBank({
        "stand": str(POL / "alpha_stand.onnx"),
        "jump": str(target),
    })
    w = compose_world(str(ROBOT), [DuckSpec("sky", (0, 0))], playground=False)
    d = DuckRuntime(w.model, w.data, bank, prefix="sky/", name="sky")
    for _ in range(2 * 50):
        d.step(); mujoco.mj_step(w.model, w.data, SUBSTEPS)
    # jump: swap to the jump policy with a zero command (behavior contract)
    d.active_policy = "jump"
    d.set_command()
    peak = 0.0
    for _ in range(int(2.5 * 50)):
        d.step(); mujoco.mj_step(w.model, w.data, SUBSTEPS)
        fz = min(d.site_pos("left_foot")[2], d.site_pos("right_foot")[2])
        peak = max(peak, fz)
    d.active_policy = "stand"; d.set_command()
    for _ in range(int(1.5 * 50)):
        d.step(); mujoco.mj_step(w.model, w.data, SUBSTEPS)
    up = d.is_upright()
    print(f"hop: peak both-feet clearance = {peak*100:.1f} cm, landed upright={up}")
    assert peak > 0.03, f"hop too weak ({peak*100:.1f} cm < 3 cm)"
    assert up, "did not land upright"
    print("JUMP POLICY OK — ready for the rope")


if __name__ == "__main__":
    main()
