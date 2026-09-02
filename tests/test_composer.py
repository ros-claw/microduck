"""Unit tests for the multi-duck composer and runtime contracts."""
import os
import pathlib
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from microduck_lab.sim.composer import compose_world, DuckSpec, RopeSpec
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS

ROOT = pathlib.Path(os.environ.get("MICRODUCK_ROOT",
                                   "~/workspace/microduck")).expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"


@pytest.fixture(scope="module")
def world():
    ducks = [DuckSpec("lavender", (-0.25, 0.0), yaw=0.0),
             DuckSpec("cream", (0.25, 0.0), yaw=np.pi),
             DuckSpec("sky", (0.0, -0.3), yaw=np.pi / 2)]
    return compose_world(str(ROBOT), ducks,
                         rope=RopeSpec("lavender", "cream", length=0.55, count=30),
                         playground=False)


def test_three_ducks_one_world(world):
    assert world.model.nu == 42          # 3 × 14 servos
    assert len(world.rope_body_ids) == 31
    assert len(world.rope_eq_ids) == 2
    for name in ("lavender", "cream", "sky"):
        assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"{name}/trunk_base") >= 0


def test_obs_contract(world):
    bank = PolicyBank({"stand": str(POL / "alpha_stand.onnx")})
    d = DuckRuntime(world.model, world.data, bank, prefix="sky/", name="sky")
    assert d.get_obs().shape == (61,)
    assert d.act_ids.shape == (14,)


def test_rope_latch(world):
    """Latch must hold rope ends at the mouths (no silent qpos0 offset)."""
    bank = PolicyBank({"stand": str(POL / "alpha_stand.onnx")})
    rt = {n: DuckRuntime(world.model, world.data, bank, prefix=f"{n}/", name=n)
          for n in ("lavender", "cream", "sky")}
    for _ in range(2 * 50):
        for d in rt.values():
            d.step()
        mujoco.mj_step(world.model, world.data, SUBSTEPS)
    ma = rt["lavender"].site_pos("mouth_tip")
    mb = rt["cream"].site_pos("mouth_tip")
    e0 = world.data.xpos[world.rope_body_ids[0]]
    eN = world.data.xpos[world.rope_body_ids[-1]]
    assert np.linalg.norm(ma - e0) < 0.06, f"rope start off mouth A: {np.linalg.norm(ma-e0):.3f}"
    assert np.linalg.norm(mb - eN) < 0.06, f"rope end off mouth B: {np.linalg.norm(mb-eN):.3f}"


def test_unlatch_drops_rope(world):
    world.set_rope_latched(False)
    for _ in range(int(1.5 * 200)):
        mujoco.mj_step(world.model, world.data, 1)
    zmax = world.rope_kinematics()["max_z"]
    assert zmax < 0.03, "unlatched rope must fall to the floor"
    world.set_rope_latched(True)
