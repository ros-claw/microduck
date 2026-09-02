"""Quick parameter sweep for rope turning: measure stability + loop quality."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import mujoco
from microduck_lab.sim.composer import compose_world, DuckSpec, RopeSpec
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS
from microduck_lab.skills.rope_turn import RopeTurner, RopeTurnPair

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
STAND = str(ROOT / "microduck/policies/alpha_stand.onnx")


def trial(freq, radius, rope_len, sep, rope_density=300.0, seconds=10.0):
    ducks = [
        DuckSpec("a", (-sep / 2, 0.0), yaw=0.0),
        DuckSpec("b", (sep / 2, 0.0), yaw=np.pi),
    ]
    w = compose_world(ROBOT, ducks,
                      rope=RopeSpec("a", "b", length=rope_len, count=28, density=rope_density),
                      playground=False)
    bank = PolicyBank({"stand": STAND})
    rt = {d.name: DuckRuntime(w.model, w.data, bank, prefix=f"{d.name}/", name=d.name)
          for d in ducks}
    for _ in range(2 * 50):  # settle
        for r in rt.values():
            r.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
    pair = RopeTurnPair(RopeTurner(rt["a"], radius), RopeTurner(rt["b"], radius),
                        frequency=freq)
    pair.start()
    min_z, max_z = 1e9, -1e9
    fell = False
    belly_min_after_ramp = 1e9
    for i in range(int(seconds * 50)):
        pair.update(0.02)
        for r in rt.values():
            r.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        rk = w.rope_kinematics()
        min_z = min(min_z, rk["min_z"]); max_z = max(max_z, rk["max_z"])
        if i > 3 * 50:
            belly_min_after_ramp = min(belly_min_after_ramp, rk["min_z"])
        if not (rt["a"].is_upright() and rt["b"].is_upright()):
            fell = True
            break
    return dict(freq=freq, radius=radius, rope_len=rope_len, sep=sep,
                fell=fell, t=i / 50, rope_z=(round(min_z, 3), round(max_z, 3)),
                belly_min=round(belly_min_after_ramp, 3))


if __name__ == "__main__":
    import itertools, sys
    for freq, radius in itertools.product([1.2, 1.5], [0.02, 0.028, 0.036]):
        res = trial(freq, radius, rope_len=0.82, sep=0.72)
        print(res, flush=True)
